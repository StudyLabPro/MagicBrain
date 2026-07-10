"""Cross-process wiring of the MagicBrain twin service into the MAGIC bus.

Subscribes to KnowledgeBaseAI student-progress events and folds each score into
the student's :class:`NeuralDigitalTwin` as an observation (шов 6, L4→L2). The
twin's substrate self-surprise can be published upward (шов 4 producer).

The ``studyninja_magic`` SDK is an optional dependency; if it (or the bus) is
absent the wiring degrades to a no-op, keeping MagicBrain runnable standalone.
See docs/REFLEXIVE_VERTICAL_SEAMS.md.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from magicbrain.integration.neural_digital_twin import NeuralDigitalTwin


def substrate_metrics_from_twin(twin: NeuralDigitalTwin) -> Dict[str, float]:
    """Read the twin's brain self-surprise (energy/debt/firing) for шов 4.

    Meaningful only when the twin's ``TextBrain`` tracks energy and trains;
    otherwise returns the quiet/healthy defaults (zero penalty downstream).
    """
    brain = getattr(twin, "brain", None)
    if brain is None:
        return {}
    firing = float(brain.firing_rate()) if hasattr(brain, "firing_rate") else 0.0
    return {
        "energy": float(getattr(brain, "energy", 0.0)),
        "delta_energy": float(getattr(brain, "delta_energy", 0.0)),
        "lr_penalty": float(getattr(brain, "_lr_penalty", 1.0)),
        "firing_rate": firing,
    }


def default_twin_registry() -> Tuple[Dict[str, NeuralDigitalTwin], Callable[[str], NeuralDigitalTwin]]:
    """A simple in-memory twin registry: ``student_id`` → lazily-created twin."""
    twins: Dict[str, NeuralDigitalTwin] = {}

    def get_twin(student_id: str) -> NeuralDigitalTwin:
        sid = student_id or "anonymous"
        if sid not in twins:
            twins[sid] = NeuralDigitalTwin(sid)
        return twins[sid]

    return twins, get_twin


async def wire_twin_service(
    bus: Any,
    get_twin: Callable[[str], NeuralDigitalTwin],
    *,
    publish_substrate_on_update: bool = False,
) -> None:
    """Subscribe the twin service to KB progress events (шов 6, L4→L2).

    On each ``magic.kb.student_progress_updated`` event the KB mastery score is
    folded into the student's twin as an observation. Optionally re-publishes
    the twin's substrate self-surprise upward (шов 4) via the SDK helper.
    """
    publish_substrate = None
    if publish_substrate_on_update:
        try:
            from studyninja_magic.wiring import publish_substrate as _ps
            publish_substrate = _ps
        except Exception:  # pragma: no cover - SDK optional
            publish_substrate = None

    async def _handler(event: Any) -> None:
        payload = getattr(event, "payload", {}) or {}
        student_id = payload.get("student_id", "") or getattr(event, "student_id", "")
        topic_id = payload.get("topic_id", "") or getattr(event, "topic_id", "")
        score = float(payload.get("score", getattr(event, "score", 0.0)))
        twin = get_twin(student_id)
        twin.process_interaction_event(
            {"type": "student_progress_updated", "topic_id": topic_id, "score": score}
        )
        if publish_substrate is not None:
            metrics = substrate_metrics_from_twin(twin)
            if metrics:
                await publish_substrate(bus, twin_id=student_id or topic_id, **metrics)

    await bus.subscribe("magic.kb.student_progress_updated", _handler)
