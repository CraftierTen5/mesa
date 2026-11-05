"""Wildlife epidemiology model combining continuous and discrete spaces."""

from __future__ import annotations

from mesa import Model
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.examples.experimental.multi_space_epidemiology.agents import (
    HealthState,
    WildlifeAgent,
)
from mesa.examples.experimental.multi_space_epidemiology.diagnostics import (
    ModelDiagnostics,
    get_model_diagnostics,
)
from mesa.examples.experimental.multi_space_epidemiology.spatial_modes import (
    SpatialMode,
)
from mesa.experimental.continuous_space import ContinuousSpace
from mesa.experimental.scenarios import Scenario


class EpidemiologyScenario(Scenario):
    """Configuration parameters for the multi-space wildlife epidemiology model."""

    width: int = 4
    height: int = 4
    population_size: int = 8
    initial_infected: int = 2
    movement_speed: float = 0.3
    spatial_mode: SpatialMode = SpatialMode.SYNCHRONIZED
    scenario_type: str = "default"  # "default", "scenario_1", "scenario_2"
    recovery_steps: int = 0  # 0 disables recovery
    environmental_risk: float = 0.0


class WildlifeEpidemiologyModel(Model):
    """Ecosystem simulation investigating spatial coordination and cell occupancy.

    Combines:
    - ContinuousSpace: authoritative continuous coordinates (R^2).
    - OrthogonalMooreGrid: discrete environmental grid providing zoning metadata and cell occupancy.
    """

    def __init__(
        self,
        scenario: EpidemiologyScenario | None = None,
        **kwargs,
    ):
        """Initialize the multi-space wildlife epidemiology model.

        Args:
            scenario: Optional EpidemiologyScenario instance.
            **kwargs: Scenario parameter overrides.
        """
        if "spatial_mode" in kwargs and isinstance(kwargs["spatial_mode"], str):
            kwargs["spatial_mode"] = SpatialMode(kwargs["spatial_mode"])

        if scenario is None:
            scenario = EpidemiologyScenario(**kwargs)
        elif kwargs:
            # Recreate frozen scenario with overrides
            fields = [
                "width",
                "height",
                "population_size",
                "initial_infected",
                "movement_speed",
                "spatial_mode",
                "scenario_type",
                "recovery_steps",
                "environmental_risk",
                "seed",
            ]
            values = {f: getattr(scenario, f) for f in fields if hasattr(scenario, f)}
            values.update(kwargs)
            scenario = EpidemiologyScenario(**values)

        super().__init__(scenario=scenario)
        self.steps = 0
        self.spatial_mode: SpatialMode = SpatialMode(scenario.spatial_mode)

        # Set up discrete environmental grid (torus=False for bounded world)
        self.grid = OrthogonalMooreGrid(
            (scenario.width, scenario.height),
            torus=False,
            random=self.random,
        )
        # Environmental metadata layer (resource level / habitat quality)
        self.grid.create_property_layer("resource_level", 1.0)

        # Set up continuous space
        self.space = ContinuousSpace(
            [[0, scenario.width], [0, scenario.height]],
            torus=False,
            random=self.random,
            n_agents=scenario.population_size,
        )

        self.population: list[WildlifeAgent] = []
        self._initialize_population()

    def _initialize_population(self) -> None:
        """Create a controlled infection pair and fill the remaining population."""
        n = self.scenario.population_size

        if n < 2:
            raise ValueError("population_size must be at least 2.")

        # Keep at least one infected and one susceptible animal.
        n_inf = min(max(1, self.scenario.initial_infected), n - 1)

        # First two animals form the controlled transmission experiment.
        infected = WildlifeAgent(
            self,
            state=HealthState.INFECTIOUS,
            position=(1.2, 1.2),
        )
        susceptible = WildlifeAgent(
            self,
            state=HealthState.SUSCEPTIBLE,
            position=(1.8, 1.8),
        )

        self.population = [infected, susceptible]

        # Fill the remaining population using the requested infection count.
        remaining_states = (
            [HealthState.INFECTIOUS] * (n_inf - 1)
            + [HealthState.SUSCEPTIBLE] * (n - n_inf - 1)
        )
        self.random.shuffle(remaining_states)

        for state in remaining_states:
            self.population.append(WildlifeAgent(self, state=state))
    def cell_for(self, agent: WildlifeAgent):
        """Return the discrete grid cell corresponding to the agent's continuous coordinates."""
        return self.grid.find_nearest_cell(agent.position)

    def set_spatial_mode(self, mode: SpatialMode | str) -> None:
        """Switch spatial mode dynamically."""
        self.spatial_mode = SpatialMode(mode)

    def step(self) -> None:
        """Execute one simulation step: movement, deterministic cell infection, and aging."""
        self.steps += 1
        for agent in self.population:
            agent.move()

        # 2. Infection transmission phase (Deterministic cell-occupancy rule)
        # An infected animal transmits infection to susceptible animals registered in the same cell.
        newly_infected: list[WildlifeAgent] = []
        for cell in self.grid.all_cells:
            # Look up agents registered in this cell's discrete occupancy collection
            registered_agents = cell.agents
            has_infected_occupant = any(
                a.state == HealthState.INFECTIOUS for a in registered_agents
            )
            if has_infected_occupant:
                for a in registered_agents:
                    if a.state == HealthState.SUSCEPTIBLE and a not in newly_infected:
                        newly_infected.append(a)

        for agent in newly_infected:
            agent.state = HealthState.INFECTIOUS
            agent.state_age = 0

        # 3. Aging / Recovery phase (if enabled)
        if self.scenario.recovery_steps > 0:
            for agent in self.population:
                if agent.state == HealthState.INFECTIOUS:
                    agent.state_age += 1
                    if agent.state_age >= self.scenario.recovery_steps:
                        agent.state = HealthState.RECOVERED
                        agent.state_age = 0

    def get_diagnostics(self) -> ModelDiagnostics:
        """Retrieve live consistency and epidemiology diagnostics."""
        return get_model_diagnostics(self)


if __name__ == "__main__":
    test_model = WildlifeEpidemiologyModel()
    for _ in range(5):
        test_model.step()
    diag = test_model.get_diagnostics()
    print(
        f"Step {diag.step}: Infected={diag.infected}, Susceptible={diag.susceptible}, Mismatches={diag.occupancy_mismatches}"
    )
