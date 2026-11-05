"""Research fixtures for lookup, manual synchronization, and coordination.

Run with ``python -m mesa.examples.experimental.multi_space_epidemiology.spatial_benchmark``.
The coordinator is local experimental code and makes no claim of a Mesa API.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from random import Random
from typing import Literal

import numpy as np

import mesa
from mesa import Model
from mesa.discrete_space import Cell, CellAgent, OrthogonalMooreGrid
from mesa.experimental.continuous_space import ContinuousSpace, ContinuousSpaceAgent

Strategy = Literal["lookup", "manual", "coordinator"]


class BenchmarkModel(Model):
    """Bare Mesa model whose RNG makes benchmark positions reproducible."""

    def __init__(self, seed: int = 17):
        super().__init__(rng=seed)


class LookupAgent(ContinuousSpaceAgent):
    """Continuous-only benchmark agent with no grid occupancy state."""

    def __init__(self, space: ContinuousSpace, model: Model, position):
        super().__init__(space, model)
        self.position = np.asarray(position, dtype=float)


class SpatialAgent(ContinuousSpaceAgent, CellAgent):
    """Continuous agent with explicit, optional cell occupancy operations."""

    def __init__(self, space: ContinuousSpace, model: Model, position):
        super().__init__(space, model)
        self.cell = None
        self.position = np.asarray(position, dtype=float)

    def synchronize_cell(self, grid: OrthogonalMooreGrid, position=None) -> None:
        """Manually synchronize occupancy; the caller owns operation ordering."""
        target = grid.find_nearest_cell(self.position if position is None else position)
        if target is self.cell:
            return
        self.cell = target

    def remove_from_cell(self) -> None:
        if self.cell is not None:
            self.cell = None

    def remove(self) -> None:
        self.remove_from_cell()
        super().remove()


class SpatialCoordinator:
    """Small prototype coordinating an agent's continuous and grid records.

    The continuous-space ``agent.position`` is canonical. Validation and cell
    capacity checks happen before mutation. The destination cell is registered
    before the old one is released, following Mesa's CellAgent setter ordering.
    Unexpected failures during the underlying mutations are not rolled back.
    """

    def __init__(
        self,
        space: ContinuousSpace,
        grid: OrthogonalMooreGrid,
        grid_scale: tuple[float, ...] | None = None,
    ):
        self.space = space
        self.grid = grid
        self.grid_scale = np.asarray(grid_scale or (1.0,) * space.ndims)
        if self.grid_scale.shape != (space.ndims,):
            raise ValueError("grid_scale must have one value per continuous dimension")
        if np.any(self.grid_scale <= 0):
            raise ValueError("grid_scale values must be positive")

    def _prepare(self, position) -> tuple[np.ndarray, Cell]:
        point = np.asarray(position, dtype=float)
        if point.shape != (self.space.ndims,):
            raise ValueError(f"Expected position shape {(self.space.ndims,)}, got {point.shape}")
        if self.space.torus:
            point = self.space.torus_correct(point)
        elif not self.space.in_bounds(point):
            raise ValueError(f"point {point} is outside the continuous space")
        target = self.grid.find_nearest_cell(point * self.grid_scale)
        return point, target

    def place(self, agent: SpatialAgent, position) -> None:
        point, target = self._prepare(position)
        old_position = agent.position.copy()
        old_cell = agent.cell
        if target.capacity is not None and target is not old_cell and target.is_full:
            raise ValueError(f"destination cell {target.coordinate} is full")
        agent.position = point
        try:
            # Use the already validated target rather than repeat the grid lookup.
            agent.cell = target
        except Exception:
            # Cell.add_agent capacity failures occur before occupancy mutation.
            agent.position = old_position
            raise

    def move(self, agent: SpatialAgent, position) -> None:
        self.place(agent, position)

    def remove(self, agent: SpatialAgent) -> None:
        if getattr(agent, "_coordinator_removed", False):
            return
        agent.remove()
        agent._coordinator_removed = True


@dataclass
class Fixture:
    model: BenchmarkModel
    space: ContinuousSpace
    grid: OrthogonalMooreGrid
    agents: list[ContinuousSpaceAgent]
    coordinator: SpatialCoordinator


def make_fixture(size: int, strategy: Strategy, seed: int = 17) -> Fixture:
    model = BenchmarkModel(seed)
    space = ContinuousSpace([[0, 100], [0, 100]], random=model.random, n_agents=size)
    grid = OrthogonalMooreGrid((100, 100), random=model.random)
    coordinator = SpatialCoordinator(space, grid)
    rng = Random(seed)
    agents = []
    for _ in range(size):
        position = [rng.random() * 100, rng.random() * 100]
        agent_type = LookupAgent if strategy == "lookup" else SpatialAgent
        agent = agent_type(space, model, position)
        agents.append(agent)
        if strategy == "manual":
            agent.synchronize_cell(grid)
        elif strategy == "coordinator":
            coordinator.place(agent, position)
    return Fixture(model, space, grid, agents, coordinator)


def _timed(repeats: int, fn) -> list[float]:
    fn()  # one warm-up, excluded from samples
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return samples


def benchmark_case(size: int, repeats: int, seed: int = 17) -> dict:
    """Measure placement, movement, cell query, removal, total, and peak memory."""
    points = np.random.default_rng(seed).uniform(0, 99.999, size=(size, 2))
    movement_workload = np.random.default_rng(seed + 1).uniform(
        0, 99.999, size=(repeats + 1, size, 2)
    )
    results = {}
    for strategy in ("lookup", "manual", "coordinator"):
        def placement_trial(strategy=strategy, points=points, size=size, seed=seed):
            model = BenchmarkModel(seed)
            space = ContinuousSpace([[0, 100], [0, 100]], random=model.random, n_agents=size)
            grid = OrthogonalMooreGrid((100, 100), random=model.random)
            coordinator = SpatialCoordinator(space, grid)
            start = time.perf_counter()
            agents = []
            for point in points:
                agent_type = LookupAgent if strategy == "lookup" else SpatialAgent
                agent = agent_type(space, model, point)
                agents.append(agent)
                if strategy == "manual":
                    agent.synchronize_cell(grid)
                elif strategy == "coordinator":
                    coordinator.place(agent, point)
            elapsed = time.perf_counter() - start
            return elapsed, Fixture(model, space, grid, agents, coordinator)

        placement_trial()  # warm-up; measured interval excludes model/space setup
        placement = [placement_trial()[0] for _ in range(repeats)]
        fixture = make_fixture(size, strategy, seed)

        movement_index = 0
        def move_phase(fixture=fixture, strategy=strategy):
            nonlocal movement_index
            current_points = movement_workload[movement_index]
            movement_index += 1
            for agent, point in zip(fixture.agents, current_points, strict=True):
                if strategy == "lookup":
                    agent.position = point
                elif strategy == "manual":
                    agent.position = point
                    agent.synchronize_cell(fixture.grid)
                else:
                    fixture.coordinator.move(agent, point)

        movement = _timed(repeats, move_phase)
        def query_phase(fixture=fixture, strategy=strategy):
            return sum(
                (fixture.grid.find_nearest_cell(a.position).coordinate[0]
                 if strategy == "lookup" else a.cell.coordinate[0])
                for a in fixture.agents
            )

        query = _timed(repeats, query_phase)

        def removal_trial(strategy=strategy, size=size, seed=seed):
            doomed = make_fixture(size, strategy, seed)
            start = time.perf_counter()
            if strategy == "coordinator":
                for agent in doomed.agents:
                    doomed.coordinator.remove(agent)
            else:
                for agent in doomed.agents:
                    agent.remove()
            return time.perf_counter() - start

        removal_trial()  # warm-up, setup excluded from measured removal interval
        remove = [removal_trial() for _ in range(repeats)]
        tracemalloc.start()
        memory_fixture = make_fixture(size, strategy, seed)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del memory_fixture
        gc.collect()
        phases = {
            "placement_s": placement,
            "movement_s": movement,
            "cell_query_s": query,
            "removal_s": remove,
        }
        results[strategy] = {
            key: {"median": statistics.median(values), "samples": values}
            for key, values in phases.items()
        }
        results[strategy]["total_median_s"] = sum(
            statistics.median(values) for values in (placement, movement, query, remove)
        )
        results[strategy]["peak_tracemalloc_bytes"] = peak
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 1000, 10000])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    print(f"Python: {__import__('sys').version.split()[0]}")
    print(f"Mesa: {getattr(mesa, '__version__', 'unknown')}")
    print(f"Seed: 17; repeats: {args.repeats}; timer: time.perf_counter; warm-up: 1")
    print("Times are seconds; all samples are printed for reproducibility.")
    for size in args.sizes:
        print(f"\nAgents: {size}")
        for strategy, data in benchmark_case(size, args.repeats).items():
            print(strategy, data)


if __name__ == "__main__":
    main()
