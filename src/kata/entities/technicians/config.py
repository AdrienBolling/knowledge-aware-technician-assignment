"""Pydantic configuration models for Technician entities."""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class TechnicianConfig(BaseModel):
    """Configuration for a GymTechnician.

    Fields cover identity, fatigue dynamics, and knowledge-grid parameters.

    A config entry may also be expanded from a named template by
    providing a ``template`` key.  The named template is loaded from
    :mod:`kata.EntityFactories.technician_factory` and any other fields
    in the entry override the template's defaults.

    Example (in JSON)::

        {"template": "expert", "name": "alice"}
    """

    name: str = Field(
        default="technician",
        description="Human-readable name of the technician.",
    )

    # ------------------------------------------------------------------
    # Fatigue parameters
    # ------------------------------------------------------------------
    fatigue_lambda: float = Field(
        default=0.01,
        gt=0.0,
        description=(
            "Fatigue accumulation rate. Controls how quickly fatigue grows "
            "during work (exponential accumulation model)."
        ),
    )
    fatigue_mu: float = Field(
        default=0.05,
        gt=0.0,
        description=(
            "Fatigue recovery rate. Controls how quickly fatigue decays "
            "during idle time (exponential recovery model)."
        ),
    )

    # ------------------------------------------------------------------
    # Knowledge-grid parameters (used by GymTechnician / KnowledgeGrid)
    # ------------------------------------------------------------------
    knowledge_k_shape: tuple[int, ...] = Field(
        default=(10, 10),
        description="Shape of the knowledge grid (dimensionality of the embedding space).",
    )
    knowledge_propagation_sigma: float = Field(
        default=1.0,
        gt=0.0,
        description="Sigma for Gaussian propagation of knowledge across the grid.",
    )
    knowledge_transmission_factor: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Fraction of knowledge transmitted when sharing.",
    )
    knowledge_learning_rate: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Learning rate for knowledge updates.",
    )

    @model_validator(mode="before")
    @classmethod
    def _expand_template(cls, data: Any) -> Any:
        """Expand a ``template`` key by merging the named template's fields."""
        if not isinstance(data, dict):
            return data
        template_name = data.get("template")
        if template_name is None:
            return data
        from kata.EntityFactories.technician_factory import get_template

        merged: dict[str, Any] = get_template(template_name)
        for key, value in data.items():
            if key == "template":
                continue
            merged[key] = value
        return merged


# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

default_technician = TechnicianConfig()

expert_technician = TechnicianConfig(
    name="expert_technician",
    fatigue_lambda=0.005,
    fatigue_mu=0.08,
    knowledge_k_shape=(10, 10),
    knowledge_propagation_sigma=1.5,
    knowledge_transmission_factor=0.7,
    knowledge_learning_rate=0.15,
)

junior_technician = TechnicianConfig(
    name="junior_technician",
    fatigue_lambda=0.02,
    fatigue_mu=0.03,
    knowledge_k_shape=(10, 10),
    knowledge_propagation_sigma=0.5,
    knowledge_transmission_factor=0.3,
    knowledge_learning_rate=0.05,
)

# Repository of named default technician configurations
technician_config_registry: dict[str, TechnicianConfig] = {
    "default": default_technician,
    "expert": expert_technician,
    "junior": junior_technician,
}
