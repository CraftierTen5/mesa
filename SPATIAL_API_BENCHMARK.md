# Multi-space spatial coordination benchmark

## Research question and semantics

For agents with continuous physical positions and discrete grid cells, does
explicit coordination prevent meaningful consistency errors or reduce
implementation complexity enough to justify a future shared API?

The three strategies use these intended semantics:

- **A — lookup-only:** continuous position is authoritative; a regular grid
  cell is derived on demand. No discrete membership exists. Environmental
  properties can be queried, but this strategy cannot answer occupant,
  capacity, or cell-neighborhood membership questions.
- **B — manual synchronization:** continuous position remains authoritative;
  model code updates `CellAgent.cell` during placement/movement and clears it
  during removal. Correctness requires the model to perform each paired update.
- **C — experimental coordinator:** `SpatialCoordinator` owns placement,
  movement, and removal calls for a `SpatialAgent`. Its continuous
  `agent.position` is canonical; the coordinator validates it, derives the
  destination cell, then updates both existing Mesa objects. It is local
  research code, not a Mesa API proposal.

For B and C, the invariant after a successful operation is: the agent is in
the continuous space; it is present exactly once in the cell selected from
its canonical position; and its `cell` reference points to that cell. A uses
the weaker invariant that its continuous membership is live and a fresh
lookup returns the intended environmental zone.

The regular grid maps with floor indexing: `find_nearest_cell` returns the
containing cell for this grid. No mismatch between that rule and geometric
nearest-center mapping is presumed. The independent test reference uses
`floor(position * grid_scale)`.

## Implementation and failure semantics

`spatial_benchmark.py` contains separate `LookupAgent` and
`SpatialAgent(ContinuousSpaceAgent, CellAgent)` classes. The latter uses both
existing agent lifecycles. Manual synchronization calls the existing
`cell` descriptor, which adds the destination before removing the old cell.
The coordinator does not replace either space or agent behavior.

The coordinator validates position shape, continuous bounds, grid mapping,
and destination capacity before changing position. A rejected regular-grid
upper-bound position or full destination leaves both prior records unchanged;
the tests cover these cases. If an unexpected exception occurs after a cell
mutation has begun, the coordinator does **not** claim general atomicity or
rollback. On the standard cell-capacity failure path, the preflight and
Mesa's add-before-remove cell setter preserve the prior occupancy. The
coordinator makes removal idempotent; native repeated removal of a
`ContinuousSpaceAgent` is not idempotent in this checkout because its first
removal clears its `space` reference.

Explicit `grid_scale` supports an application-defined mapping when physical
domain dimensions and grid resolution differ. The coordinator does not infer
that mapping. Matching continuous/grid torus extents are tested; mismatched
torus extents are not.

## Correctness matrix

The deterministic tests in `tests/examples/test_spatial_coordination.py`
cover:

| Case | Expected result |
| --- | --- |
| Lookup placement | Environment cell resolves; agent has no cell membership state |
| Manual/coordinator initial placement | Continuous membership and exactly one matching cell registration |
| Movement within cell | Position changes; cell membership remains singular |
| Boundary and multi-cell movement | Cell becomes the floor-mapped destination |
| Interior integer boundary | Exact boundary maps to the positive-side cell; just below maps to the prior cell |
| Continuous maximum edge | Continuous space accepts its maximum, regular grid rejects it; coordinator rejects before mutation |
| Position below/above non-torus domain | Coordinator rejects before changing either record |
| Toroidal out-of-range coordinate | Continuous position normalizes, grid registration wraps consistently |
| Full-cell update | Manual position-first update can leave position and occupancy inconsistent; coordinator preflight preserves prior state |
| Multiple occupants | One agent leaving does not remove another occupant |
| Different resolution | Explicit scale maps a 4×4 physical domain into a 2×2 grid |
| Removal | Both memberships clear; coordinator accepts repeat removal; native repeated removal raises |

No timing threshold is asserted in unit tests. The matrix establishes the
specified fixture semantics; it does not establish correctness for arbitrary
model-specific transaction requirements.

## Benchmark methodology

Run from the repository root:

```console
python -m mesa.examples.experimental.multi_space_epidemiology.spatial_benchmark --sizes 100 1000 10000 --repeats 5
```

The recorded run used Python **3.14.8**, Mesa **4.0.0a0**, checkout
`811d33027f111fd6f11cf89135faf0a2fa459c60`, seed **17**, and
`time.perf_counter`. Each phase has one warm-up and five measured repetitions;
the module prints every raw sample. Positions are seeded uniformly in a
100×100 non-torus domain with a 100×100 grid. Each movement sample updates
every agent once to a separate deterministic target. There are no epidemiology
calculations.

- **Placement:** create/register all agents into pre-created spaces; B also
  registers cells and C uses the coordinator. Space/grid construction is
  outside the timed interval.
- **Movement:** move every agent once; A sets continuous position, B sets
  position then synchronizes, and C calls the coordinator.
- **Cell query:** A performs a fresh grid lookup; B/C read their maintained
  cell reference. These intentionally measure the lookup cost versus the
  cost of using maintained occupancy.
- **Removal:** construct a fresh fixture outside the timer, then remove all
  agents. Setup is excluded from each removal sample.
- **Total:** sum of the four phase medians, not a separately timed full run.
- **Peak memory:** one `tracemalloc` peak while building a full fixture. The
  fixed 10,000-cell grid is included in all strategies; `tracemalloc` may not
  capture every native/NumPy allocation and should be treated as an estimate.

For implementation complexity, count the source lines in only the explicit
synchronization methods, including their signatures and blank/comment lines:
manual `SpatialAgent.synchronize_cell`, `remove_from_cell`, and `remove` at
`spatial_benchmark.py:51–64` total **12 lines**. Coordinator `_prepare`,
`place`, `move`, and `remove` at `spatial_benchmark.py:90–123` total **31
lines**, in addition to its 13-line initializer at `:76–88`. This is a
transparent local source-size measure, not a general complexity metric. At
the benchmark call site, A has zero occupancy synchronization calls, B has
one synchronization call at placement and one after each position update,
and C has one coordinator call per place/move/remove; C still performs the
underlying position and cell mutations.

## Results

Times below are medians in milliseconds; total is the sum of phase medians.
Memory is the peak `tracemalloc` value in MiB.

| Agents | Strategy | Place | Move | Cell query | Remove | Total | Peak MiB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 100 | A — lookup | 0.686 | 0.637 | 0.358 | 0.431 | 2.112 | 11.48 |
| 100 | B — manual | 1.885 | 1.454 | 0.011 | 0.625 | 3.975 | 11.48 |
| 100 | C — coordinator | 3.998 | 2.075 | 0.006 | 0.679 | 6.758 | 11.48 |
| 1,000 | A — lookup | 14.571 | 5.656 | 5.263 | 5.134 | 30.625 | 11.74 |
| 1,000 | B — manual | 27.658 | 17.147 | 0.103 | 6.577 | 51.486 | 11.78 |
| 1,000 | C — coordinator | 158.867 | 167.920 | 0.121 | 13.572 | 340.480 | 11.78 |
| 10,000 | A — lookup | 220.623 | 103.824 | 85.712 | 58.270 | 468.429 | 14.27 |
| 10,000 | B — manual | 316.293 | 229.622 | 2.525 | 86.281 | 634.721 | 14.62 |
| 10,000 | C — coordinator | 680.054 | 324.661 | 1.814 | 104.095 | 1,110.624 | 14.62 |

The five coordinator placement samples at 1,000 agents varied widely
(36 ms–605 ms), and the 100-agent coordinator placement also had one large
outlier (96 ms versus a 4.0 ms median). Treat those coordinator timing
estimates as noisy. The 10,000-agent coordinator samples were steadier but
still have ordinary machine-load variability. The full command prints raw
samples to allow reruns and inspection.

Lookup-only has the lowest combined time in this fixture, but does not supply
occupancy. Manual synchronization has a higher placement/movement cost than
lookup and a substantially faster repeated occupancy read. The coordinator
adds validation and an operation boundary; it is slower than direct manual
synchronization in these timings, while its tests prevent the specific
position-first capacity inconsistency and upper-bound partial update shown
here. Memory differences between B and C are small compared with the fixed
grid allocation. These measurements do not imply a general performance
ranking across workloads.

## Strategy comparison

| Strategy | Correctness available | Complexity | Measured trade-off |
| --- | --- | --- | --- |
| A — lookup-only | Fresh environmental mapping; no discrete membership invariant | Least spatial coordination code | Lowest total time; repeated cell queries do the mapping work |
| B — manual synchronization | Both records agree after successful paired updates | Model author chooses and maintains operation order | Less overhead than C; position-first failures can leave stale occupancy |
| C — experimental coordinator | Validates normal bounds/capacity failures before mutation; coordinates lifecycle | More helper code and a single explicit lifecycle entry point | Highest measured overhead here; faster cell reads due to maintained occupancy |

## Assessment

**Demonstrated:** lookup is sufficient for environmental zone queries; it
cannot answer occupancy questions. Mesa's existing continuous and cell agent
classes can be combined through multiple inheritance. Manual synchronization
can leave inconsistent state when position changes before a destination-cell
capacity failure. A small local coordinator prevents that tested failure and
the grid-upper-bound partial update, and offers idempotent removal. The
coordinator's validation and call boundary cost time and code in this fixture.

**Uncertain:** whether coordinator checks cover the failure modes important to
real models; whether users would find a common lifecycle coordinator clearer
than a carefully ordered helper; cross-space torus mappings with different
extents; and performance under realistic movement distributions, spatial
queries, and high-churn model lifecycles. Memory values are allocator-limited
estimates, and observed timing varies by host load.

**Assessment of a shared abstraction:** useful as a local pattern when models
need both continuous proximity and cell occupancy, but **inconclusive** as a
case for a unified Mesa API. This prototype centralizes a repeated invariant
but does not reduce source lines versus the narrowly counted manual helper,
and it introduces runtime overhead. The most useful follow-up workload would
be a capacity-limited, high-churn model coordinating continuous position with
multiple discrete layers, where placement failures and lifecycle updates are
common enough to reveal whether one coordinator materially reduces
model-specific bugs.
