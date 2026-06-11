"""SNN + Transformer Hybrid Architecture."""
from __future__ import annotations
from typing import Optional
from .base import HybridArchitecture


class SNNTransformerHybrid(HybridArchitecture):
    """
    Hybrid combining SNN and Transformer.

    SNN encodes input → Transformer processes sequence.
    """

    def __init__(
        self,
        snn_model,
        transformer_model,
        model_id: Optional[str] = None,
        version: str = "1.0.0",
    ):
        """Initialize SNN+Transformer hybrid."""
        components = {
            "snn_encoder": snn_model,
            "transformer": transformer_model,
        }

        connections = [
            ("snn_encoder", "transformer"),
        ]

        super().__init__(
            components=components,
            connections=connections,
            model_id=model_id or "snn_transformer_hybrid",
            version=version,
            description="SNN + Transformer hybrid",
            output_component="transformer"
        )
