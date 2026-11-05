
"""Spatial management strategies for multi-space coordination."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from mesa.discrete_space import OrthogonalMooreGrid
    from mesa.examples.experimental.multi_space_epidemiology.agents import WildlifeAgent


class SpatialMode(StrEnum):
    """The three spatial management modes."""

    LOOKUP_ONLY = "lookup_only"
    REGISTER_ONCE = "register_once"
    SYNCHRONIZED = "synchronized"


def place_agent(
    mode: SpatialMode | str,
    agent: WildlifeAgent,
    grid: OrthogonalMooreGrid,
    position: np.ndarray,
) -> None:
    """Place an agent in continuous space and optionally register its cell."""
    mode = SpatialMode(mode)

    # ContinuousSpaceAgent manages its space through this position setter.
    agent.position = position

    if mode == SpatialMode.LOOKUP_ONLY:
        # Deliberately do not register in discrete space.
        agent.cell = None
    else:
        # Initial registration for REGISTER_ONCE and SYNCHRONIZED.
        agent.cell = grid.find_nearest_cell(agent.position)


def update_agent_position(
    mode: SpatialMode | str,
    agent: WildlifeAgent,
    grid: OrthogonalMooreGrid,
    new_position: np.ndarray,
) -> None:
    """Move continuously and synchronize discrete occupancy when required."""
    mode = SpatialMode(mode)

    # Always move in continuous space, irrespective of spatial mode.
    agent.position = new_position

    if mode == SpatialMode.SYNCHRONIZED:
        target_cell = grid.find_nearest_cell(agent.position)

        if agent.cell is not target_cell:
            agent.cell = target_cell

    # LOOKUP_ONLY: remains unregistered.
    # REGISTER_ONCE: retains its original cell membership.


def remove_agent_from_spaces(
    mode: SpatialMode | str,
    agent: WildlifeAgent,
) -> None:
    """Remove discrete occupancy when the selected mode uses registration."""
    mode = SpatialMode(mode)

    if (
        mode in (SpatialMode.REGISTER_ONCE, SpatialMode.SYNCHRONIZED)
        and agent.cell is not None
    ):
        agent.cell = None
