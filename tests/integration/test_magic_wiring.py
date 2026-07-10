"""Cross-process wiring of the MagicBrain twin service (шов 6, L4→L2).

A KB progress event published on one bus instance is folded into the student's
twin by the twin service subscribed on another — proving delivery across
"processes" via the shared in-process exchange. The SDK is optional.
"""
import asyncio

import pytest

pytest.importorskip("studyninja_magic")

from studyninja_magic import InMemoryEventBus, publish_progress, reset_shared_bus

from magicbrain.integration.magic_wiring import default_twin_registry, wire_twin_service


def test_twin_service_folds_kb_progress_over_the_bus():
    async def scenario():
        reset_shared_bus()
        kb = InMemoryEventBus(shared=True)
        svc = InMemoryEventBus(shared=True)
        await kb.connect()
        await svc.connect()

        twins, get_twin = default_twin_registry()
        await wire_twin_service(svc, get_twin)

        # KB (another "process") reports mastery for a topic.
        await publish_progress(kb, topic_id="algebra", student_id="s1", score=0.8, prior_mastery=0.0)
        return twins

    twins = asyncio.run(scenario())
    assert "s1" in twins                                        # twin created on demand
    assert twins["s1"].mastery_scores.get("algebra", 0.0) > 0.0  # KB score folded in
