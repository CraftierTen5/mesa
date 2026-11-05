"""Experimental wildlife epidemiology model spanning continuous and cell spaces."""

from __future__ import annotations

from mesa.examples.experimental.multi_space_epidemiology.agents import (
    HealthState,
    WildlifeAgent,
)
from mesa.examples.experimental.multi_space_epidemiology.diagnostics import (
    AgentDiagnostic,
    ModelDiagnostics,
    get_model_diagnostics,
    validate_synchronized_invariants,
)
from mesa.examples.experimental.multi_space_epidemiology.model import (
    EpidemiologyScenario,
    WildlifeEpidemiologyModel,
)
from mesa.examples.experimental.multi_space_epidemiology.spatial_modes import (
    SpatialMode,
)

# Compatibility aliases
EpidemiologyModel = WildlifeEpidemiologyModel
SpatialStrategy = SpatialMode

__all__ = [
    "AgentDiagnostic",
    "EpidemiologyModel",
    "EpidemiologyScenario",
    "HealthState",
    "ModelDiagnostics",
    "SpatialMode",
    "SpatialStrategy",
    "WildlifeAgent",
    "WildlifeEpidemiologyModel",
    "get_model_diagnostics",
    "validate_synchronized_invariants",
]
