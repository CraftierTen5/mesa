"""Direct SpaceRendererComponent visualization for multi-space wildlife epidemiology."""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import solara
from mesa.examples.experimental.multi_space_epidemiology.agents import (
    HealthState,
    WildlifeAgent,
)
from mesa.examples.experimental.multi_space_epidemiology.diagnostics import (
    get_model_diagnostics,
)
from mesa.examples.experimental.multi_space_epidemiology.model import (
    EpidemiologyScenario,
    WildlifeEpidemiologyModel,
)
from mesa.examples.experimental.multi_space_epidemiology.spatial_modes import (
    SpatialMode,
)
from mesa.visualization import SpaceRenderer
from mesa.visualization.components import AgentPortrayalStyle
from mesa.visualization.solara_viz import ModelController, SpaceRendererComponent
from mesa.visualization.utils import force_update, update_counter


class MultiSpaceRenderer(SpaceRenderer):
    """Custom SpaceRenderer that visualizes both ContinuousSpace and OrthogonalMooreGrid.

    - Authoritative spatial domain: ContinuousSpace (ensuring agents are plotted at
      their true continuous coordinates in all three spatial modes).
    - Environmental overlay: Discrete grid boundaries, cell coordinate labels, and
      indicators for unregistered and stale cell registrations.
    """

    def __init__(self, model: WildlifeEpidemiologyModel, backend: str = "matplotlib"):
        self._model = model
        super().__init__(model, backend=backend)

    @property
    def space(self):
        """Authoritative space is the continuous space holding physical agent positions."""
        return self._model.space

    @space.setter
    def space(self, _):
        """Prevent SpaceRendererComponent from forcing DiscreteSpace as the primary space."""
        pass

    def draw_structure(self, **kwargs):
        """Draw continuous boundary and overlay the discrete environmental grid."""
        while self.canvas.texts:
            self.canvas.texts[0].remove()

        mesh = super().draw_structure(**kwargs)
        ax = self.canvas
        width = self._model.scenario.width
        height = self._model.scenario.height

        # Draw discrete grid lines
        for x in range(width + 1):
            ax.axvline(x, color="#dddddd", linestyle="-", linewidth=1.0, zorder=1)
        for y in range(height + 1):
            ax.axhline(y, color="#dddddd", linestyle="-", linewidth=1.0, zorder=1)

        # Label cell coordinates at cell centers
        for x in range(width):
            for y in range(height):
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    f"({x}, {y})",
                    color="#cccccc",
                    fontsize=8,
                    ha="center",
                    va="center",
                    zorder=1,
                )

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect("equal")
        ax.set_xticks(range(width + 1))
        ax.set_yticks(range(height + 1))
        return mesh

    def draw_agents(self, **kwargs):
        """Draw animals at continuous coordinates with occupancy diagnostic markers."""
        mesh = super().draw_agents(**kwargs)
        ax = self.canvas

        for agent in self._model.population:
            x, y = float(agent.position[0]), float(agent.position[1])
            actual = agent.actual_cell
            expected = agent.expected_cell

            # Mode B: Stale registration vector connecting continuous pos to old cell center
            if actual is not None and actual != expected:
                stale_x = actual.coordinate[0] + 0.5
                stale_y = actual.coordinate[1] + 0.5
                ax.plot(
                    [x, stale_x],
                    [y, stale_y],
                    color="#e377c2",
                    linestyle="--",
                    linewidth=1.8,
                    zorder=4,
                )
                ax.scatter(
                    [stale_x],
                    [stale_y],
                    marker="x",
                    s=90,
                    color="#e377c2",
                    linewidths=2,
                    zorder=5,
                )

            # Mode A: Orange warning halo for completely unregistered animal
            if actual is None:
                ax.scatter(
                    [x],
                    [y],
                    s=260,
                    facecolors="none",
                    edgecolors="#ff7f0e",
                    linestyle="--",
                    linewidths=2.0,
                    zorder=4,
                )

            # Text label above each animal
            label = "I" if agent.state == HealthState.INFECTIOUS else "S"
            ax.text(
                x,
                y + 0.12,
                f"#{agent.unique_id} ({label})",
                fontsize=8,
                ha="center",
                va="bottom",
                fontweight="bold",
                color="#222222",
                zorder=6,
            )

        mode_name = self._model.spatial_mode.value.replace("_", " ").title()
        ax.set_title(
            f"Mode: {mode_name} | Step: {self._model.steps}",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        return mesh


def wildlife_portrayal(agent: WildlifeAgent) -> AgentPortrayalStyle:
    """Portray wildlife individuals using continuous coordinates and infection status."""
    color = "#d62728" if agent.state == HealthState.INFECTIOUS else "#2ca02c"
    return AgentPortrayalStyle(
        color=color,
        size=120,
        marker="o",
        zorder=3,
    )


def create_renderer(model: WildlifeEpidemiologyModel) -> MultiSpaceRenderer:
    """Instantiate and initialize the dual-space renderer."""
    renderer = MultiSpaceRenderer(model, backend="matplotlib")
    renderer.setup_structure().setup_agents(wildlife_portrayal)
    renderer.draw_structure()
    renderer.draw_agents()
    return renderer


# Initial default simulation
initial_model = WildlifeEpidemiologyModel(
    EpidemiologyScenario(spatial_mode=SpatialMode.SYNCHRONIZED)
)
initial_renderer = create_renderer(initial_model)


@solara.component
def DiagnosticPanel(model: WildlifeEpidemiologyModel):
    """Display real-time spatial consistency metrics and inspect selected agents."""
    update_counter.get()
    diag = get_model_diagnostics(model)

    selected_id, set_selected_id = solara.use_state(
        diag.agents[0].agent_id if diag.agents else 0
    )

    with solara.Column():
        solara.Markdown("### Spatial Consistency & Diagnostics")

        # Aggregate metric cards
        with solara.Row(style={"flex-wrap": "wrap", "gap": "10px", "margin-bottom": "15px"}):
            with solara.Card("Step", style={"min-width": "100px"}):
                solara.Text(f"{diag.step}", style={"font-size": "22px", "font-weight": "bold"})
            with solara.Card("Population", style={"min-width": "110px"}):
                solara.Text(f"{diag.total_animals}", style={"font-size": "22px", "font-weight": "bold"})
            with solara.Card("Infected", style={"min-width": "110px"}):
                solara.Text(
                    f"{diag.infected}",
                    style={"font-size": "22px", "font-weight": "bold", "color": "#d62728"},
                )
            with solara.Card("Susceptible", style={"min-width": "110px"}):
                solara.Text(
                    f"{diag.susceptible}",
                    style={"font-size": "22px", "font-weight": "bold", "color": "#2ca02c"},
                )

        with solara.Row(style={"flex-wrap": "wrap", "gap": "10px", "margin-bottom": "15px"}):
            with solara.Card("Occupancy Mismatches", style={"min-width": "160px"}):
                color = "#d62728" if diag.occupancy_mismatches > 0 else "#2ca02c"
                solara.Text(
                    f"{diag.occupancy_mismatches}",
                    style={"font-size": "22px", "font-weight": "bold", "color": color},
                )
            with solara.Card("Unregistered Animals", style={"min-width": "160px"}):
                solara.Text(f"{diag.unregistered_animals}", style={"font-size": "22px", "font-weight": "bold"})
            with solara.Card("Stale Memberships", style={"min-width": "160px"}):
                solara.Text(f"{diag.stale_memberships}", style={"font-size": "22px", "font-weight": "bold"})

        # Inspector section
        with solara.Card("Selected Animal Inspector"):
            agent_ids = [a.agent_id for a in diag.agents]
            effective_id = (
                selected_id if selected_id in agent_ids else (agent_ids[0] if agent_ids else None)
            )

            if agent_ids and effective_id is not None:
                solara.Select(
                    label="Inspect Animal ID:",
                    value=effective_id,
                    values=agent_ids,
                    on_value=set_selected_id,
                )

                selected_agent = next((a for a in diag.agents if a.agent_id == effective_id), None)
                if selected_agent is not None:
                    status_text = (
                        "MATCH (Consistent)"
                        if not selected_agent.is_mismatched
                        else (
                            "UNREGISTERED (Lookup only)"
                            if selected_agent.is_unregistered
                            else "STALE (Old cell)"
                        )
                    )
                    status_color = "#2ca02c" if not selected_agent.is_mismatched else "#d62728"
                    actual_str = (
                        f"{selected_agent.actual_cell}"
                        if selected_agent.actual_cell is not None
                        else "None (Unregistered)"
                    )

                    solara.Markdown(f"""
- **Agent ID:** `#{selected_agent.agent_id}`
- **Health State:** `{selected_agent.state.upper()}`
- **Continuous Coordinates:** `({selected_agent.position[0]:.3f}, {selected_agent.position[1]:.3f})`
- **Expected Grid Cell:** `{selected_agent.expected_cell}`
- **Actual Registered Cell:** `{actual_str}`
- **Consistency Status:** <span style="color: {status_color}; font-weight: bold;">{status_text}</span>
""")
            else:
                solara.Text("No active agents in model.")

        with solara.Card("Visual Legend"):
            solara.Markdown("""
- **Green circle**: Susceptible animal (healthy)
- **Red circle**: Infected animal (infectious)
- **Orange dashed halo**: Animal physically present but completely unregistered in cell (Mode A)
- **Pink dashed line & cross**: Stale registration pointing to previous cell after movement (Mode B)
""")


@solara.component
def Page():
    """Main application component using direct SpaceRendererComponent rendering."""
    reactive_model = solara.use_reactive(initial_model)
    reactive_renderer = solara.use_reactive(initial_renderer)

    play_interval = solara.use_reactive(150)
    render_interval = solara.use_reactive(1)
    use_threads = solara.use_reactive(False)

    # Interactive parameter state
    current_params = solara.use_reactive({
        "spatial_mode": "synchronized",
        "scenario_type": "scenario_1",
        "population_size": 2,
        "initial_infected": 1,
        "movement_speed": 0.3,
    })

    def update_model_params(key, value):
        new_params = {**current_params.value, key: value}
        current_params.value = new_params
        new_model = WildlifeEpidemiologyModel(**new_params)
        reactive_model.value = new_model
        reactive_renderer.value = create_renderer(new_model)
        force_update()

    with solara.AppBar():
        solara.AppBarTitle("Multi-Space Wildlife Epidemiology Demonstration")

    with solara.Sidebar():
        with solara.Card("Simulation Controls"):
            ModelController(
                reactive_model,
                renderer=reactive_renderer,
                model_parameters=current_params,
                play_interval=play_interval,
                render_interval=render_interval,
                use_threads=use_threads,
            )
            solara.SliderInt(
                label="Play Interval (ms)",
                value=play_interval,
                min=20,
                max=500,
                step=10,
            )

        with solara.Card("Experiment Mode"):
            solara.Select(
                label="Choose behaviour:",
                value=current_params.value["spatial_mode"],
                values=["synchronized", "lookup_only"],
                on_value=lambda val: update_model_params("spatial_mode", val),
            )

            mode_desc = {
                "synchronized": (
                    "CORRECT: Animals are registered in their corresponding "
                    "grid cells. Infection spreads between co-located animals."
                ),
                "lookup_only": (
                    "INCORRECT: Animals exist in continuous space but are not "
                    "registered in grid cells. Cell-based infection fails."
                ),
            }

            solara.Markdown(
                mode_desc[current_params.value["spatial_mode"]]
            )
        with solara.Card("Model Configuration"):
            solara.SliderInt(
                label="Population Size",
                value=current_params.value["population_size"],
                min=2,
                max=20,
                on_value=lambda val: update_model_params("population_size", val),
            )
            solara.SliderInt(
                label="Initial Infected",
                value=current_params.value["initial_infected"],
                min=1,
                max=5,
                on_value=lambda val: update_model_params("initial_infected", val),
            )
            solara.SliderFloat(
                label="Movement Speed",
                value=current_params.value["movement_speed"],
                min=0.0,
                max=1.0,
                step=0.05,
                on_value=lambda val: update_model_params("movement_speed", val),
            )

    # Main dashboard view
    with solara.Row(style={"padding": "15px", "gap": "20px"}):
        with solara.Column(style={"flex": "1", "min-width": "480px"}):
            with solara.Card("Continuous Wildlife & Discrete Environmental Grid"):
                SpaceRendererComponent(
                    model=reactive_model.value,
                    renderer=reactive_renderer.value,
                )

        with solara.Column(style={"flex": "1", "min-width": "420px"}):
            DiagnosticPanel(reactive_model.value)


page = Page()
