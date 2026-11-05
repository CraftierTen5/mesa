# Multi-Space Wildlife Epidemiology Demonstration

A self-contained experimental demonstration in Mesa examining the consequences of combining continuous physical movement (`ContinuousSpace`) with discrete environmental zoning (`OrthogonalMooreGrid`) when spatial representations are maintained independently.

> **Important Note:** This experiment demonstrates a model-level coordination challenge when combining independent spatial abstractions. It is **not** an assertion that Mesa's existing discrete or continuous space APIs are defective. `OrthogonalMooreGrid.find_nearest_cell` is designed as a spatial query; discrete occupancy registration is a separate model-level lifecycle invariant.

---

## 1. Research Question

When a model models animals moving continuously across continuous 2D space while referencing an orthogonal grid representing environmental habitat cells, **does looking up an animal's corresponding grid cell automatically register the animal in that cell? And does continuous position movement automatically update discrete cell membership?**

This demonstration makes the answer concrete by distinguishing between three distinct concepts:
1. **Position Lookup:** Calculating which discrete grid cell contains a continuous coordinate `(x, y)`.
2. **Cell Occupancy:** Maintaining the collection of agents (`cell.agents`) registered inside that discrete cell.
3. **Movement Synchronization:** Updating cell membership whenever continuous coordinates cross discrete cell boundaries.

---

## 2. Spatial Representations in the Simulation

The model integrates two complementary spatial abstractions:
- **Continuous Space (`ContinuousSpace`):** Holds the authoritative physical coordinates `(x, y)` of wildlife individuals, allowing free, fine-grained movement.
- **Discrete Environmental Grid (`OrthogonalMooreGrid`):** Overlaid on the continuous domain, partitioning space into regular cells with habitat metadata (e.g., resource level). Each cell maintains a collection `cell.agents` representing its discrete occupants.

---

## 3. The Three Spatial-Management Modes

The simulation allows the user to switch between three modes to observe the consequences of coordination:

### Mode A: Lookup Only (Deliberately Unsynchronized Baseline)
- **Mechanism:** When placed or moved, the animal queries its containing cell via `grid.find_nearest_cell(position)`. Its continuous coordinates are updated, but the animal is **never registered** in `cell.agents` (`agent.cell = None`).
- **Demonstration:** Proves that spatial lookup alone does not establish discrete cell occupancy.
- **Consequence:** Cell-based disease contact detection iterates over `cell.agents`. Because neither animal is registered, cell-based transmission fails completely—even if an infected and a susceptible animal are physically sitting right next to each other in the same cell.

### Mode B: Register Once, Then Move (Deliberately Stale Occupancy)
- **Mechanism:** The animal registers in its initial containing cell `cell_0` at creation (`agent.cell = cell_0`). When the animal moves continuously into adjacent cells, its continuous coordinates update, but its cell membership is **never updated**.
- **Demonstration:** Proves that updating continuous coordinates does not automatically transfer cell membership.
- **Consequence:** The animal physically leaves `cell_0` and enters `cell_1`, but `cell_0.agents` still reports the animal as occupying it, while `cell_1.agents` reports nothing. In cell `cell_0`, phantom transmissions occur (susceptible animals in `cell_0` are falsely infected by an animal that has already physically departed), while susceptible animals in `cell_1` remain undetected.

### Mode C: Manually Synchronized (Control)
- **Mechanism:** On initial placement and after every continuous movement step, the model queries `new_cell = grid.find_nearest_cell(position)`. If `agent.cell != new_cell`, the agent executes `agent.cell = new_cell` (which safely removes the agent from the previous cell and registers it in the new cell).
- **Demonstration:** Demonstrates correct model-level synchronization.
- **Consequence:** The expected cell and actual registered cell are always identical; `cell.agents` accurately reflects physical presence; cell-based infection transmission occurs where animals physically reside.

---

## 4. Deterministic Demonstration Scenarios

The model includes two deterministic scenarios designed to verify spatial behavior without confounding random factors:

### Scenario 1: Missing Registration
- **Setup:** 1 infected animal at `(1.2, 1.2)` and 1 susceptible animal at `(1.8, 1.8)` in a 4×4 grid. Both animals map to grid cell `(1, 1)`. Movement speed is 0.
- **In Lookup Only Mode:** Neither animal is registered in cell `(1, 1)`. Contact detection queries `cell.agents`, finds no infected occupant, and transmission fails.
- **In Synchronized Mode:** Both animals are registered in cell `(1, 1)`. Transmission occurs on step 1; the susceptible animal turns red (infected).

### Scenario 2: Stale Registration After Movement
- **Setup:** 1 infected animal initially at `(1.5, 1.5)` in cell `(1, 1)` and 1 susceptible animal at `(1.2, 1.2)` in cell `(1, 1)`.
- **Movement:** On step 1, the infected animal moves to `(2.5, 1.5)`, which is physically inside cell `(2, 1)`. The susceptible animal stays in cell `(1, 1)`.
- **In Register Once Mode:** The infected animal's continuous position is `(2.5, 1.5)` in cell `(2, 1)`, but its cell registration remains stale in cell `(1, 1)`. The susceptible animal in cell `(1, 1)` is falsely infected by the phantom presence of the infected animal.
- **In Synchronized Mode:** The infected animal's registration moves to cell `(2, 1)`. Cell `(1, 1)` is cleared of the infected animal, and the susceptible animal in cell `(1, 1)` remains healthy (green).

---

## 5. Interactive Solara Visualization

The interactive dashboard provides:
- **Visual Grid Map:** Overlaid continuous and discrete spaces. Animals are drawn at their **exact floating-point continuous coordinates**.
  - **Green Circle:** Susceptible animal.
  - **Red Circle:** Infected animal.
  - **Orange Dashed Halo:** Unregistered animal (Mode A).
  - **Pink Dashed Line & Cross:** Stale registration indicator pointing back to the previous cell after movement (Mode B).
- **Interactive Controls:** Start / Pause / Step / Reset, Spatial Mode selection, Scenario Presets, and Population sliders.
- **Live Diagnostic Panel:** Live counts of Population, Infected, Susceptible, Occupancy Mismatches, Unregistered Animals, and Stale Memberships, plus an individual animal inspector.

### How to Run the Visualization

Launch the Solara web interface with:

```bash
solara run mesa/examples/experimental/multi_space_epidemiology/app.py
```

Or run the model directly in headless mode:

```bash
python -m mesa.examples.experimental.multi_space_epidemiology.model
```

---

## 6. How to Run the Tests

Execute the automated test suite with pytest (disabling Solara's test plugin under Python 3.14):

```bash
pytest -p no:solara tests/examples/test_multi_space_epidemiology.py
```

To run both this suite and the spatial coordination benchmark matrix:

```bash
pytest -p no:solara tests/examples/test_multi_space_epidemiology.py tests/examples/test_spatial_coordination.py
```

---

## 7. Model Invariants & Limitations

In synchronized mode, the following invariants are enforced:
1. `agent.expected_cell is grid.find_nearest_cell(agent.position)`
2. `agent.actual_cell is agent.expected_cell`
3. `agent in agent.actual_cell.agents`
4. Exactly one cell across the grid contains `agent` in its `cell.agents` collection.
5. Moving within a cell preserves single membership without duplication.
6. Moving across cell boundaries cleans up previous occupancy.
7. Removing an animal cleans up continuous space and discrete cell occupancy.

**Limitations:**
- This is a simplified epidemiological toy model (SEIR/deterministic contact) designed exclusively to isolate spatial coordination correctness, not for real-world disease forecasting.
- Non-torus grid boundaries use half-open ranges `[0, width)`, whereas `ContinuousSpace` includes `width`. Coordinates near the upper bound are clipped to `np.nextafter(width, 0)` to maintain safe domain compatibility.
