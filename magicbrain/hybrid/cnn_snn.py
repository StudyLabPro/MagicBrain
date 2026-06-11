"""CNN + SNN Hybrid for Vision."""
from __future__ import annotations
from typing import Optional
from .base import HybridArchitecture


class CNNSNNHybrid(HybridArchitecture):
    """
    Hybrid for vision: CNN features → SNN processing.

    CNN extracts features → SNN performs neuromorphic classification.
    """

    def __init__(
        self,
        cnn_model,
        snn_model,
        model_id: Optional[str] = None,
        version: str = "1.0.0",
    ):
        """Initialize CNN+SNN hybrid."""
        components = {
            "cnn_features": cnn_model,
            "snn_classifier": snn_model,
        }

        connections = [
            ("cnn_features", "snn_classifier"),
        ]

        super().__init__(
            components=components,
            connections=connections,
            model_id=model_id or "cnn_snn_hybrid",
            version=version,
            description="CNN feature extractor + SNN classifier",
            output_component="snn_classifier"
        )
