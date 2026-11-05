"""Automated tests validating multi-space wildlife epidemiology and spatial coordination."""

from random import Random

import numpy as np
import pytest
import reacton.core

from mesa.discrete_space import OrthogonalMooreGrid
from mesa.examples.experimental.multi_space_epidemiology import (
    EpidemiologyScenario,
    HealthState,
    SpatialMode,
    WildlifeEpidemiologyModel,
    get_model_diagnostics,
    validate_synchronized_invariants,
)
from mesa.examples.experimental.multi_space_epidemiology.app import page


def create_model(spatial_mode=SpatialMode.SYNCHRONIZED, **kwargs):
    """Helper to create a deterministic test model."""
    params = {
        "width": 4,
        "height": 4,
        "population_size": 4,
        "initial_infected": 1,
        "movement_speed": 0.0,
        "spatial_mode": spatial_mode,
        "seed": 42,
    }
    params.update(kwargs)
    scenario = EpidemiologyScenario(**params)
    return WildlifeEpidemiologyModel(scenario)


# =============================================================================
# Invariants for Synchronized Mode (Section 7)
# =============================================================================


def test_synchronized_mode_invariants_satisfied_initially():
    """Verify that synchronized mode satisfies all spatial occupancy invariants."""
    model = create_model(SpatialMode.SYNCHRONIZED, population_size=6)
    violations = validate_synchronized_invariants(model)
    assert violations == [], f"Unexpected invariant violations: {violations}"


def test_synchronized_expected_equals_actual_and_in_cell_agents():
    """Assert expected cell equals actual cell, and agent is in cell.agents."""
    model = create_model(SpatialMode.SYNCHRONIZED)
    for agent in model.population:
        expected = model.grid.find_nearest_cell(agent.position)
        assert agent.expected_cell is expected
        assert agent.actual_cell is expected
        assert agent in expected.agents


def test_synchronized_agent_in_exactly_one_cell():
    """Assert each agent appears in exactly one cell's occupancy collection across the grid."""
    model = create_model(SpatialMode.SYNCHRONIZED, population_size=10)
    for agent in model.population:
        cells_with_agent = [c for c in model.grid.all_cells if agent in c.agents]
        assert len(cells_with_agent) == 1
        assert cells_with_agent[0] is agent.expected_cell


def test_synchronized_movement_within_same_cell_no_duplication():
    """Moving within the same discrete cell updates position without duplicating membership."""
    model = create_model(SpatialMode.SYNCHRONIZED, population_size=1)
    agent = model.population[0]
    agent.position = np.array([1.1, 1.1])
    cell_1 = model.cell_for(agent)
    agent.cell = cell_1

    # Move within cell (1, 1)
    agent.move(displacement=(0.3, 0.4))
    assert np.allclose(agent.position, [1.4, 1.5])
    assert agent.cell is cell_1
    assert cell_1.agents.count(agent) == 1
    assert sum(agent in c.agents for c in model.grid.all_cells) == 1


def test_synchronized_movement_across_boundary_updates_occupancy():
    """Moving across cell boundary removes old membership and adds new membership."""
    model = create_model(SpatialMode.SYNCHRONIZED, population_size=1)
    agent = model.population[0]
    agent.position = np.array([1.2, 1.2])
    old_cell = model.grid.find_nearest_cell(agent.position)
    agent.cell = old_cell
    assert agent in old_cell.agents

    # Move into cell (2, 2)
    agent.move(displacement=(1.0, 1.0))
    new_cell = model.grid.find_nearest_cell(agent.position)
    assert new_cell.coordinate == (2, 2)
    assert agent.cell is new_cell
    assert agent not in old_cell.agents
    assert agent in new_cell.agents
    assert sum(agent in c.agents for c in model.grid.all_cells) == 1


def test_synchronized_removal_cleans_up_occupancy_and_space():
    """Removing an agent clears both continuous space and discrete cell occupancy."""
    model = create_model(SpatialMode.SYNCHRONIZED, population_size=2)
    agent = model.population[0]
    cell = agent.cell
    assert agent in cell.agents
    assert agent in model.space.agents

    agent.remove()
    assert agent not in cell.agents
    assert agent not in model.space.agents
    assert agent.cell is None


# =============================================================================
# Baseline Behavior Tests (Section 4 & 7)
# =============================================================================


def test_lookup_only_does_not_register_in_cell():
    """Lookup mode determines containing cell but never registers in cell.agents."""
    model = create_model(SpatialMode.LOOKUP_ONLY, population_size=2)
    for agent in model.population:
        expected = model.grid.find_nearest_cell(agent.position)
        assert agent.expected_cell is expected
        assert agent.actual_cell is None
        assert agent not in expected.agents
        assert agent.is_unregistered is True
        assert agent.is_occupancy_mismatched is True


def test_register_once_retains_stale_membership_after_movement():
    """Register-once mode keeps initial cell membership when agent moves to another cell."""
    model = create_model(SpatialMode.REGISTER_ONCE, population_size=1)
    agent = model.population[0]
    agent.position = np.array([1.2, 1.2])
    initial_cell = model.grid.find_nearest_cell(agent.position)
    agent.cell = initial_cell
    assert initial_cell.coordinate == (1, 1)
    assert agent in initial_cell.agents

    # Move continuously to cell (2, 3)
    agent.move(displacement=(1.0, 2.0))
    expected_cell = model.grid.find_nearest_cell(agent.position)
    assert expected_cell.coordinate == (2, 3)

    # In register-once mode, actual cell remains the initial cell
    assert agent.cell is initial_cell
    assert agent in initial_cell.agents
    assert agent not in expected_cell.agents
    assert agent.is_stale_membership is True
    assert agent.is_occupancy_mismatched is True


def test_diagnostics_detect_mismatches_accurately():
    """Diagnostic system correctly counts mismatches, unregistered agents, and stale agents."""
    # Lookup mode: all agents should be unregistered mismatches
    model_lookup = create_model(SpatialMode.LOOKUP_ONLY, population_size=4)
    diag_lookup = get_model_diagnostics(model_lookup)
    assert diag_lookup.total_animals == 4
    assert diag_lookup.occupancy_mismatches == 4
    assert diag_lookup.unregistered_animals == 4
    assert diag_lookup.stale_memberships == 0

    # Synchronized mode: 0 mismatches
    model_sync = create_model(SpatialMode.SYNCHRONIZED, population_size=4)
    diag_sync = get_model_diagnostics(model_sync)
    assert diag_sync.occupancy_mismatches == 0
    assert diag_sync.unregistered_animals == 0
    assert diag_sync.stale_memberships == 0


# =============================================================================
# Deterministic Scenarios (Section 6)
# =============================================================================


def test_scenario_1_missing_registration_contrast():
    """Scenario 1: Co-located animals in same cell.

    Lookup mode fails to transmit because agents are not registered in cell.agents.
    Synchronized mode succeeds in transmitting infection deterministically.
    """
    # 1. Lookup Only: infection fails because cell.agents is empty
    model_lookup = WildlifeEpidemiologyModel(
        scenario_type="scenario_1",
        spatial_mode=SpatialMode.LOOKUP_ONLY,
    )
    assert len(model_lookup.population) == 2
    inf, susc = model_lookup.population
    assert inf.state == HealthState.INFECTIOUS
    assert susc.state == HealthState.SUSCEPTIBLE
    assert inf.expected_cell == susc.expected_cell  # Co-located in cell (1, 1)

    model_lookup.step()
    # Infection fails because neither is in cell.agents
    assert susc.state == HealthState.SUSCEPTIBLE

    # 2. Synchronized: infection succeeds
    model_sync = WildlifeEpidemiologyModel(
        scenario_type="scenario_1",
        spatial_mode=SpatialMode.SYNCHRONIZED,
    )
    inf_s, susc_s = model_sync.population
    assert inf_s.expected_cell == susc_s.expected_cell
    assert inf_s in inf_s.expected_cell.agents
    assert susc_s in susc_s.expected_cell.agents

    model_sync.step()
    # Transmission succeeds!
    assert susc_s.state == HealthState.INFECTIOUS


def test_scenario_2_stale_registration_contrast():
    """Scenario 2: Infected animal moves to new cell.

    Register-once falsely infects susceptible animal in old cell due to stale membership.
    Synchronized mode correctly updates membership so susceptible animal in old cell is safe.
    """
    # 1. Register Once: stale registration causes false infection in old cell
    model_stale = WildlifeEpidemiologyModel(
        scenario_type="scenario_2",
        spatial_mode=SpatialMode.REGISTER_ONCE,
    )
    inf, susc = model_stale.population
    old_cell = susc.expected_cell

    # Step 1 moves infected animal to cell (2, 1)
    model_stale.step()
    assert inf.expected_cell.coordinate == (2, 1)
    assert inf.actual_cell == old_cell  # Stale in cell (1, 1)!
    # False transmission in cell (1, 1) because stale infected agent is still in old_cell.agents
    assert susc.state == HealthState.INFECTIOUS

    # 2. Synchronized: infected animal's registration is moved to cell (2, 1)
    model_sync = WildlifeEpidemiologyModel(
        scenario_type="scenario_2",
        spatial_mode=SpatialMode.SYNCHRONIZED,
    )
    inf_s, susc_s = model_sync.population

    # Step 1 moves infected animal to cell (2, 1)
    model_sync.step()
    assert inf_s.expected_cell.coordinate == (2, 1)
    assert inf_s.actual_cell == inf_s.expected_cell
    assert inf_s not in susc_s.expected_cell.agents
    # Susceptible animal in cell (1, 1) remains healthy!
    assert susc_s.state == HealthState.SUSCEPTIBLE


# =============================================================================
# Edge Cases (Section 7)
# =============================================================================


def test_boundary_coordinates_mapping():
    """Verify exact floor boundary mapping for discrete grid."""
    grid = OrthogonalMooreGrid((4, 4), random=Random(1))
    assert grid.find_nearest_cell([1.0, 2.0]).coordinate == (1, 2)
    assert grid.find_nearest_cell([0.999, 2.0]).coordinate == (0, 2)
    with pytest.raises(ValueError):
        grid.find_nearest_cell([4.0, 2.0])


def test_movement_back_to_previous_cell():
    """Moving out of a cell and then back maintains clean single occupancy."""
    model = create_model(SpatialMode.SYNCHRONIZED, population_size=1)
    agent = model.population[0]
    agent.position = np.array([1.5, 1.5])
    cell_a = model.grid.find_nearest_cell(agent.position)
    agent.cell = cell_a

    # Move to cell B (2, 1)
    agent.move(displacement=(1.0, 0.0))
    cell_b = agent.cell
    assert cell_b.coordinate == (2, 1)
    assert agent not in cell_a.agents
    assert agent in cell_b.agents

    # Move back to cell A (1, 1)
    agent.move(displacement=(-1.0, 0.0))
    assert agent.cell is cell_a
    assert agent in cell_a.agents
    assert agent not in cell_b.agents
    assert sum(agent in c.agents for c in model.grid.all_cells) == 1


def test_multiple_animals_in_single_cell():
    """Multiple animals occupying the same cell are all tracked without conflict."""
    model = create_model(SpatialMode.SYNCHRONIZED, population_size=3)
    cell = model.grid.find_nearest_cell([1.5, 1.5])
    for agent in model.population:
        agent.position = np.array([1.5, 1.5])
        agent.cell = cell

    assert len(cell.agents) == 3
    for agent in model.population:
        assert agent in cell.agents
        assert agent.actual_cell is cell


def test_upper_world_boundary_clipping():
    """Continuous movement clips below grid dimensions to prevent out-of-bounds error."""
    model = create_model(SpatialMode.SYNCHRONIZED, population_size=1)
    agent = model.population[0]
    agent.position = np.array([3.5, 3.5])
    agent.cell = model.grid.find_nearest_cell(agent.position)

    # Attempt to move far past upper boundary (4.0, 4.0)
    agent.move(displacement=(5.0, 5.0))
    # Coordinates must be strictly < 4.0
    assert agent.position[0] < 4.0
    assert agent.position[1] < 4.0
    # Expected cell must resolve cleanly to (3, 3) without ValueError
    assert agent.expected_cell.coordinate == (3, 3)
    assert agent.actual_cell.coordinate == (3, 3)


def test_visualization_app_imports_cleanly():
    """Verify that app.py imports cleanly and produces a valid SolaraViz component."""
    assert isinstance(page, reacton.core.Element)
