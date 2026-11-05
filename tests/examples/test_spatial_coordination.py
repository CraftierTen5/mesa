"""Correctness matrix for the experimental multi-space benchmark fixtures."""

from random import Random

import numpy as np
import pytest

from mesa.discrete_space import OrthogonalMooreGrid
from mesa.examples.experimental.multi_space_epidemiology.spatial_benchmark import (
    BenchmarkModel,
    LookupAgent,
    SpatialAgent,
    SpatialCoordinator,
)
from mesa.exceptions import CellFullException
from mesa.experimental.continuous_space import ContinuousSpace


def setup_spaces(*, width=4, height=4, torus=False, capacity=None, grid_shape=None):
    """Create matching deterministic continuous and discrete spaces."""
    model = BenchmarkModel(23)
    space = ContinuousSpace(
        [[0, width], [0, height]], torus=torus, random=model.random, n_agents=4
    )
    grid = OrthogonalMooreGrid(
        grid_shape or (width, height),
        torus=torus,
        capacity=capacity,
        random=Random(23),
    )
    return model, space, grid, SpatialCoordinator(space, grid)


def new_agent(model, space, position=(0.2, 0.2)):
    """Create an already continuous-space-registered agent."""
    return SpatialAgent(space, model, position)


def assert_synchronized(agent, space, grid, scale=(1.0, 1.0)):
    """Assert live membership and the defined floor-cell mapping agree."""
    assert agent in space.agents
    assert agent.cell is grid.find_nearest_cell(agent.position * scale)
    assert agent in agent.cell.agents
    assert sum(agent in cell.agents for cell in grid.all_cells) == 1


def test_lookup_only_supports_environmental_query_without_occupancy():
    """A lookup resolves a zone but cannot answer cell-occupant queries."""
    model, space, grid, _ = setup_spaces()
    agent = LookupAgent(space, model, (1.2, 2.7))
    assert grid.find_nearest_cell(agent.position).coordinate == (1, 2)
    assert not hasattr(agent, "cell")
    assert agent not in grid.find_nearest_cell(agent.position).agents


@pytest.mark.parametrize("operation", ["manual", "coordinator"])
def test_initial_placement_and_same_cell_movement(operation):
    """Placement registers exactly once; sub-cell movement retains occupancy."""
    model, space, grid, coordinator = setup_spaces()
    agent = new_agent(model, space)
    if operation == "manual":
        agent.synchronize_cell(grid)
    else:
        coordinator.place(agent, agent.position)
    original_cell = agent.cell
    if operation == "manual":
        agent.position = np.array([0.8, 0.8])
        agent.synchronize_cell(grid)
    else:
        coordinator.move(agent, [0.8, 0.8])
    assert agent.cell is original_cell
    assert_synchronized(agent, space, grid)


@pytest.mark.parametrize("destination", [(1.2, 0.2), (3.2, 3.2)])
@pytest.mark.parametrize("operation", ["manual", "coordinator"])
def test_boundary_crossing_and_multi_cell_movement(operation, destination):
    """A move to another cell, including several cells away, updates occupancy."""
    model, space, grid, coordinator = setup_spaces()
    agent = new_agent(model, space)
    if operation == "manual":
        agent.synchronize_cell(grid)
        agent.position = np.array(destination)
        agent.synchronize_cell(grid)
    else:
        coordinator.place(agent, agent.position)
        coordinator.move(agent, destination)
    assert agent.cell.coordinate == tuple(int(value) for value in destination)
    assert_synchronized(agent, space, grid)


@pytest.mark.parametrize("point,expected", [
    ((1.0, 2.0), (1, 2)),
    ((0.999, 2.0), (0, 2)),
])
def test_interior_boundary_mapping_matches_independent_floor_reference(point, expected):
    """Interior boundaries use the positive-side floor cell, independently checked."""
    model, space, grid, coordinator = setup_spaces()
    agent = new_agent(model, space, point)
    coordinator.place(agent, point)
    reference = tuple(int(np.floor(value)) for value in point)
    assert reference == expected
    assert grid.find_nearest_cell(point).coordinate == reference
    assert_synchronized(agent, space, grid)


@pytest.mark.parametrize("operation", ["manual", "coordinator"])
def test_continuous_upper_bound_is_not_a_valid_regular_grid_position(operation):
    """Grid rejection at the continuous maximum is observed without hidden wrapping."""
    model, space, grid, coordinator = setup_spaces()
    agent = new_agent(model, space)
    old_position = agent.position.copy()
    if operation == "manual":
        agent.synchronize_cell(grid)
        agent.position = np.array([4.0, 1.0])  # ContinuousSpace permits its maximum.
        with pytest.raises(ValueError):
            grid.find_nearest_cell(agent.position)
        assert agent.cell.coordinate == (0, 0)
        assert not np.array_equal(agent.position, old_position)
    else:
        coordinator.place(agent, old_position)
        with pytest.raises(ValueError):
            coordinator.move(agent, [4.0, 1.0])
        np.testing.assert_array_equal(agent.position, old_position)
        assert_synchronized(agent, space, grid)


@pytest.mark.parametrize("invalid_point", [(-0.1, 1.0), (4.1, 1.0)])
def test_coordinator_rejects_positions_beyond_non_torus_domain(invalid_point):
    """Out-of-domain positions fail before the canonical position or cell changes."""
    model, space, grid, coordinator = setup_spaces()
    agent = new_agent(model, space)
    coordinator.place(agent, agent.position)
    old_position = agent.position.copy()
    old_cell = agent.cell
    with pytest.raises(ValueError):
        coordinator.move(agent, invalid_point)
    np.testing.assert_array_equal(agent.position, old_position)
    assert agent.cell is old_cell
    assert_synchronized(agent, space, grid)


def test_coordinator_normalizes_torus_position_for_both_spaces():
    """Matching torus extents wrap a coordinate consistently across both spaces."""
    model, space, grid, coordinator = setup_spaces(torus=True)
    agent = new_agent(model, space)
    coordinator.place(agent, [-0.1, 4.1])
    np.testing.assert_allclose(agent.position, [3.9, 0.1])
    assert agent.cell.coordinate == (3, 0)
    assert_synchronized(agent, space, grid)


def test_manual_capacity_failure_leaves_position_updated_but_old_cell_registered():
    """Manual position-first ordering exposes partial state on full-cell failure."""
    model, space, grid, _ = setup_spaces(capacity=1)
    first = new_agent(model, space, (0.2, 0.2))
    second = new_agent(model, space, (1.2, 0.2))
    first.synchronize_cell(grid)
    second.synchronize_cell(grid)
    old_cell = second.cell
    second.position = np.array([0.3, 0.3])
    with pytest.raises(CellFullException):
        second.synchronize_cell(grid)
    assert second.cell is old_cell
    assert second in old_cell.agents
    assert second not in grid.find_nearest_cell(second.position).agents


def test_coordinator_capacity_failure_preserves_position_and_membership():
    """Coordinator preflight rejects a full destination before changing either record."""
    model, space, grid, coordinator = setup_spaces(capacity=1)
    first = new_agent(model, space, (0.2, 0.2))
    second = new_agent(model, space, (1.2, 0.2))
    coordinator.place(first, first.position)
    coordinator.place(second, second.position)
    old_position = second.position.copy()
    old_cell = second.cell
    with pytest.raises(ValueError, match="full"):
        coordinator.move(second, [0.3, 0.3])
    np.testing.assert_array_equal(second.position, old_position)
    assert second.cell is old_cell
    assert_synchronized(second, space, grid)


def test_multiple_agents_can_share_and_leave_a_cell_independently():
    """Occupancy retains the other agent when one of two agents moves away."""
    model, space, grid, coordinator = setup_spaces()
    agents = [new_agent(model, space, (0.2, 0.2)) for _ in range(2)]
    for agent in agents:
        coordinator.place(agent, agent.position)
    common = grid.find_nearest_cell([0.2, 0.2])
    assert common.agents.count(agents[0]) == 1
    assert common.agents.count(agents[1]) == 1
    coordinator.move(agents[0], [2.2, 2.2])
    assert agents[1] in common.agents
    assert agents[0] not in common.agents
    assert_synchronized(agents[0], space, grid)
    assert_synchronized(agents[1], space, grid)


def test_different_grid_resolution_requires_and_supports_explicit_scale():
    """A scale maps a 4 by 4 physical domain into a 2 by 2 grid."""
    model, space, grid, _ = setup_spaces(grid_shape=(2, 2))
    coordinator = SpatialCoordinator(space, grid, grid_scale=(0.5, 0.5))
    agent = new_agent(model, space, (3.2, 2.4))
    coordinator.place(agent, agent.position)
    assert agent.cell.coordinate == (1, 1)
    assert_synchronized(agent, space, grid, scale=(0.5, 0.5))


def test_removal_clears_each_membership_and_coordinator_is_idempotent():
    """First removal clears model/space/cell; coordinator accepts repeated removal."""
    model, space, _grid, coordinator = setup_spaces()
    agent = new_agent(model, space)
    coordinator.place(agent, agent.position)
    old_cell = agent.cell
    coordinator.remove(agent)
    assert agent not in model.agents
    assert agent not in space.agents
    assert agent not in old_cell.agents
    coordinator.remove(agent)
    assert agent not in old_cell.agents


def test_repeated_native_manual_removal_raises_after_first_removal():
    """The Mesa continuous agent lifecycle is not idempotent after clearing space."""
    model, space, grid, _ = setup_spaces()
    agent = new_agent(model, space)
    agent.synchronize_cell(grid)
    old_cell = agent.cell
    agent.remove()
    assert agent not in model.agents
    assert agent not in old_cell.agents
    with pytest.raises(AttributeError):
        agent.remove()

