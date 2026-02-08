"""SNN + DNN Hybrid Architecture."""
from __future__ import annotations
from typing import Optional
from .base import HybridArchitecture
from ..models.snn import SNNTextModel
from ..models.dnn import DNNModel


class SNNDNNHybrid(HybridArchitecture):
    """
    Hybrid combining SNN and DNN.

    SNN processes input → DNN performs classification/processing.
    """

    def __init__(
        self,
        snn_model: SNNTextModel,
        dnn_model: DNNModel,
        model_id: Optional[str] = None,
        version: str = "1.0.0",
    ):
        """
        Initialize SNN+DNN hybrid.

        Args:
            snn_model: SNN component
            dnn_model: DNN component
            model_id: Model ID
            version: Version
        """
        components = {
            "snn_encoder": snn_model,
            "dnn_decoder": dnn_model,
        }

        connections = [
            ("snn_encoder", "dnn_decoder"),
        ]

        super().__init__(
            components=components,
            connections=connections,
            model_id=model_id or "snn_dnn_hybrid",
            version=version,
            description="SNN encoder + DNN decoder hybrid",
            output_component="dnn_decoder"
        )


def create_snn_dnn_hybrid(
    snn_genome: str,
    vocab_size: int,
    dnn_module,
    model_id: Optional[str] = None
) -> SNNDNNHybrid:
    """Quick factory for SNN+DNN hybrid."""
    snn = SNNTextModel(snn_genome, vocab_size, model_id="snn_enc")
    dnn = DNNModel(dnn_module, model_id="dnn_dec")
    return SNNDNNHybrid(snn, dnn, model_id=model_id)
