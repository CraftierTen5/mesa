"""Diagnostic and occupancy validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mesa.examples.experimental.multi_space_epidemiology.model import (
        WildlifeEpidemiologyModel,
    )


@dataclass(frozen=True)
class AgentDiagnostic:
    """Diagnostic details for a single agent."""

    agent_id: int
    position: tuple[float, float]
    expected_cell: tuple[int, int]
    actual_cell: tuple[int, int] | None
    state: str
    is_mismatched: bool
    is_unregistered: bool
    is_stale: bool


@dataclass(frozen=True)
class ModelDiagnostics:
    """Summary metrics of simulation state and spatial consistency."""

    step: int
    total_animals: int
    infected: int
    susceptible: int
    recovered: int
    occupancy_mismatches: int
    unregistered_animals: int
    stale_memberships: int
    agents: tuple[AgentDiagnostic, ...]


def get_model_diagnostics(model: WildlifeEpidemiologyModel) -> ModelDiagnostics:
    """Extract live consistency diagnostics from the epidemiology model.

    Args:
        model: The epidemiology model.

    Returns:
        ModelDiagnostics containing aggregate counts and individual agent reports.
    """
    agent_diagnostics: list[AgentDiagnostic] = []
    mismatches = 0
    unregistered = 0
    stale = 0
    infected = 0
    susceptible = 0
    recovered = 0

    for agent in model.population:
        expected_coord = agent.expected_cell.coordinate
        actual_coord = (
            agent.actual_cell.coordinate if agent.actual_cell is not None else None
        )
        mismatch = agent.is_occupancy_mismatched
        unreg = agent.is_unregistered
        is_stale_member = agent.is_stale_membership

        if mismatch:
            mismatches += 1
        if unreg:
            unregistered += 1
        if is_stale_member:
            stale += 1

        state_val = (
            agent.state.value if hasattr(agent.state, "value") else str(agent.state)
        )
        if state_val == "infected":
            infected += 1
        elif state_val == "susceptible":
            susceptible += 1
        elif state_val == "recovered":
            recovered += 1

        pos = (float(agent.position[0]), float(agent.position[1]))
        agent_diagnostics.append(
            AgentDiagnostic(
                agent_id=agent.unique_id,
                position=pos,
                expected_cell=expected_coord,
                actual_cell=actual_coord,
                state=state_val,
                is_mismatched=mismatch,
                is_unregistered=unreg,
                is_stale=is_stale_member,
            )
        )

    return ModelDiagnostics(
        step=model.steps,
        total_animals=len(model.population),
        infected=infected,
        susceptible=susceptible,
        recovered=recovered,
        occupancy_mismatches=mismatches,
        unregistered_animals=unregistered,
        stale_memberships=stale,
        agents=tuple(agent_diagnostics),
    )


def inspect_occupancy(model):
    for agent in model.population:
        expected = agent.expected_cell

        print(f"\nAgent {agent.unique_id}")
        print("Position:", agent.position)
        print("Expected cell:", expected.coordinate)
        print("Actual cell:", agent.cell.coordinate if agent.cell else None)
        print("Present in expected cell:", agent in expected.agents)
        print("Present in actual cell:", agent in agent.cell.agents if agent.cell else False)
        

def validate_synchronized_invariants(model: WildlifeEpidemiologyModel) -> list[str]:
    """Check all correctness invariants for synchronized mode.

    Returns:
        A list of violation descriptions (empty if all invariants hold).
    """
    violations: list[str] = []
    all_grid_cells = model.grid.all_cells

    for agent in model.population:
        expected = agent.expected_cell
        actual = agent.actual_cell

        # Invariant 1: Actual cell equals expected cell
        if actual is None:
            violations.append(
                f"Agent {agent.unique_id} has actual_cell=None, expected cell at {expected.coordinate}"
            )
            continue

        if actual is not expected:
            violations.append(
                f"Agent {agent.unique_id} actual_cell {actual.coordinate} != expected_cell {expected.coordinate}"
            )

        # Invariant 2: Agent must appear in its actual cell's occupancy collection
        if agent not in actual.agents:
            violations.append(
                f"Agent {agent.unique_id} is not present in its actual cell {actual.coordinate}.agents"
            )

        # Invariant 3: Agent must appear in exactly one cell across the entire grid
        occupying_cells = [c for c in all_grid_cells if agent in c.agents]
        if len(occupying_cells) != 1:
            coords = [c.coordinate for c in occupying_cells]
            violations.append(
                f"Agent {agent.unique_id} is registered in {len(occupying_cells)} cells: {coords}"
            )

        # Invariant 4: Agent must be registered in ContinuousSpace
        if agent not in model.space.agents:
            violations.append(
                f"Agent {agent.unique_id} is missing from continuous space agents"
            )

    return violations
