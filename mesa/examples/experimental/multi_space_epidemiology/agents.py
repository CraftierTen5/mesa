"""Wildlife agent definitions combining ContinuousSpaceAgent and CellAgent."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from mesa.discrete_space import Cell, CellAgent
from mesa.examples.experimental.multi_space_epidemiology.spatial_modes import (
    place_agent,
    remove_agent_from_spaces,
    update_agent_position,
)
from mesa.experimental.continuous_space import ContinuousSpaceAgent

if TYPE_CHECKING:
    from mesa.examples.experimental.multi_space_epidemiology.model import (
        WildlifeEpidemiologyModel,
    )


class HealthState(StrEnum):
    """Health states for wildlife epidemiology."""

    SUSCEPTIBLE = "susceptible"
    INFECTIOUS = "infected"
    RECOVERED = "recovered"


class WildlifeAgent(ContinuousSpaceAgent, CellAgent):
    """A wildlife individual existing simultaneously in continuous and discrete space.

    Combines:
    - ContinuousSpaceAgent: provides continuous position in R^2 and continuous neighborhood search.
    - CellAgent: provides discrete cell occupancy via the canonical `cell` property.
    """

    def __init__(
        self,
        model: WildlifeEpidemiologyModel,
        state: HealthState = HealthState.SUSCEPTIBLE,
        position: tuple[float, float] | np.ndarray | None = None,
    ):
        """Initialize the wildlife agent.

        Args:
            model: The parent epidemiology model.
            state: Initial health state.
            position: Initial continuous position (if None, sampled randomly within bounds).
        """
        super().__init__(model.space, model)
        self.state = state
        self.state_age = 0
        self.initial_cell: Cell | None = None

        if position is None:
            max_bounds = np.nextafter([model.scenario.width, model.scenario.height], 0)
            position = np.array(
                [
                    model.random.uniform(0, max_bounds[0]),
                    model.random.uniform(0, max_bounds[1]),
                ],
                dtype=float,
            )
        else:
            position = np.asarray(position, dtype=float)

        place_agent(model.spatial_mode, self, model.grid, position)
        self.initial_cell = self.cell

    @property
    def expected_cell(self) -> Cell:
        """The discrete grid cell that corresponds to the animal's continuous position."""
        return self.model.grid.find_nearest_cell(self.position)

    @property
    def actual_cell(self) -> Cell | None:
        """The discrete cell in which the animal is currently registered (or None)."""
        return self.cell

    @property
    def is_occupancy_mismatched(self) -> bool:
        """True if the animal's actual registered cell differs from its expected cell."""
        return self.actual_cell != self.expected_cell

    @property
    def is_unregistered(self) -> bool:
        """True if the animal has no registered cell membership."""
        return self.actual_cell is None

    @property
    def is_stale_membership(self) -> bool:
        """True if the animal is registered in a cell that is not its expected cell."""
        return self.actual_cell is not None and self.actual_cell != self.expected_cell

    def move(
        self, displacement: tuple[float, float] | np.ndarray | None = None
    ) -> None:
        """Move the animal in continuous space.

        Args:
            displacement: Optional explicit (dx, dy). If None, random step within speed limit.
        """
        if displacement is None:
            speed = self.model.scenario.movement_speed
            displacement = np.array(
                [
                    self.model.random.uniform(-speed, speed),
                    self.model.random.uniform(-speed, speed),
                ]
            )
        else:
            displacement = np.asarray(displacement, dtype=float)

        limits = np.nextafter(
            [self.model.scenario.width, self.model.scenario.height], 0
        )
        new_position = np.clip(self.position + displacement, 0, limits)

        update_agent_position(
            self.model.spatial_mode,
            self,
            self.model.grid,
            new_position,
        )

    def remove(self) -> None:
        """Safely remove the agent from model, continuous space, and discrete cell."""
        remove_agent_from_spaces(self.model.spatial_mode, self)
        super().remove()
