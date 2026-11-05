# Multi-space epidemiology: spatial API findings

## Research question

Can a Mesa model keep wildlife positions in `ContinuousSpace` while using an
`OrthogonalMooreGrid` for environmental zones, and what additional work is
needed if cell occupancy is also intended to represent wildlife presence?

## Current API behavior

- `OrthogonalMooreGrid.find_nearest_cell` in `mesa/discrete_space/grid.py`
  floors each coordinate to an integer cell index. It is a containing-cell
  lookup for this grid, despite the method's historical “nearest” name. For a
  non-torus grid, it raises `ValueError` for indices outside the dimensions;
  torus grids wrap indices. Exact interior integer boundaries belong to the
  cell on the positive side. There is no distance tie because the grid uses
  floor arithmetic. See `tests/discrete_space/test_spatial_lookups.py::test_grid_lookups`.
- The grid lookup only returns a `Cell`. The function does not mutate cell
  occupancy. `Cell.agents` in `mesa/discrete_space/cell.py` returns a copy of
  the cell's registered agents; registration is via `Cell.add_agent`.
- `CellAgent.cell` in `mesa/discrete_space/cell_agent.py` adds a new cell first,
  removes the old cell, then changes its reference. Its `remove` method clears
  cell occupancy. `ContinuousSpaceAgent` in
  `mesa/experimental/continuous_space/continuous_space_agents.py` registers
  position and membership in continuous space; its remove method only removes
  continuous/model membership. Neither implementation coordinates the other.
- `ContinuousSpace.get_agents_in_radius` in
  `mesa/experimental/continuous_space/continuous_space.py` compares Euclidean
  distances against the requested radius (inclusive). With `torus=False`, the
  default model behavior uses direct Euclidean distance.
- Grid property layers are arrays attached through `Grid.create_property_layer`
  in `mesa/discrete_space/grid.py`, with per-cell accessors. Moore neighbors
  are the eight surrounding cells in 2D, set up by
  `OrthogonalMooreGrid._connect_cells_2d`.

## Experiment design

`mesa/examples/experimental/multi_space_epidemiology` implements seeded SEIR
individuals, bounded continuous movement, infectious proximity, and a
cell-level environmental risk layer. Three representations are exercised:

- `lookup`: continuous-space agents query their grid cell as needed; they are
  absent from `cell.agents`.
- `synchronized`: continuous-space agents explicitly update `cell` on initial
  placement and after movement, and clear it on removal.
- `grid`: `CellAgent` occupancy and continuous coordinate attributes are used
  without registering individuals in `ContinuousSpace`; proximity is computed
  directly from the small population.

Environmental exposure at risk 1 and proximity exposure at probability 1 are
deterministic. Lower probabilities use the model's seeded RNG. This is a
spatial API investigation, not an epidemiological realism claim.

## Results

The focused tests in `tests/examples/test_multi_space_epidemiology.py` assert
known floor mappings and boundary behavior, absent occupancy in lookup mode,
manual occupancy movement and removal, proximity exposure and SEIR transition,
environmental exposure, and grid occupancy. Commands and results are reported
in the completion summary because they depend on the executing environment.

## Strategy comparison

| Strategy | What it supports | Costs and limits |
| --- | --- | --- |
| Nearest-cell lookup | Environmental layer access from continuous position | No occupancy, cell neighborhood population, or cell-based lifecycle behavior |
| Explicit synchronization | Continuous radius search plus grid cell occupancy | Each placement, move, and removal must update both memberships; ordering matters |
| Cell occupancy with coordinate attributes | Grid population and neighborhoods, plus sub-cell coordinates | No built-in continuous-space queries or index; movement/proximity require model code |

## Failure modes and boundaries

If a continuous position changes without updating `cell`, the cell reference
and occupancy remain at the prior location. Lookup mode avoids that stale copy
because it recomputes the cell, but does not expose occupancy. Cell registration
can also fail on capacity limits; `CellAgent.cell` adds the new cell before
removing the old, so a failed add leaves the prior registration intact. The
continuous space accepts its upper bound (see `in_bounds`), while a non-torus
grid with dimensions `(width, height)` rejects coordinate `width` or `height`;
the experiment clips positions below that endpoint.

The tests establish exact integer-boundary behavior but do not assume a
nearest-center tie policy: this grid does not use nearest-center geometry.
For torus spaces, continuous and grid wrapping must be configured and reasoned
about together; this experiment deliberately uses bounded non-torus spaces.

## API sufficiency assessment

For environmental queries, `find_nearest_cell(position)` is sufficient when
the desired meaning is “which floor-indexed grid zone contains this bounded
position?” It does not provide co-location semantics. If a model needs cell
occupancy, cell capacity, or occupants of neighboring cells, explicit
registration is necessary. The current APIs can support both layers, but the
model owns their synchronization invariant.

## Recommendation

Use current APIs as-is for this use case and document the distinction between
cell lookup and cell membership. If discussion is warranted, clarify the
contract/name of `find_nearest_cell` for regular grids and state explicitly
that it has no occupancy side effect. A dedicated synchronization abstraction
is not established as necessary by this small experiment.

## Limitations and open questions

- The continuous-space API accepts positions on its maximum edge, but the
  regular grid's integer dimensions use a half-open domain. The experiment
  clips around this mismatch; it does not propose a shared bounds contract.
- Torus equivalence across different continuous and discrete extents is not
  evaluated.
- The `grid` strategy's direct pairwise distance calculation is appropriate
  only for this small fixture; it does not inherit continuous-space indexing.
- Epidemiological probabilities and transitions are intentionally simple.

## Discussion-ready summary

In the current checkout, `OrthogonalMooreGrid.find_nearest_cell` floors
coordinates to the containing cell and has no occupancy side effect. A seeded
wildlife experiment can query environmental layers this way while retaining
continuous positions. Cell occupancy is a separate `CellAgent.cell` invariant:
the model must update it during placement, movement, and removal. Existing APIs
are adequate for both approaches, with synchronization responsibilities left
to models that need occupancy. A useful follow-up discussion could clarify the
regular-grid lookup terminology and the half-open grid versus inclusive
continuous-space upper boundary.
