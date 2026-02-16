"""
Tests for Neural Digital Twin cognitive extensions.

Verifies:
- process_interaction_event with answer_submitted and lesson_completed
- get_enhanced_cognitive_state returns all expected fields
- get_optimal_difficulty for known and unknown topics
- Batch events accumulate correctly
"""
import pytest
import numpy as np
from datetime import datetime

from magicbrain.integration.neural_digital_twin import NeuralDigitalTwin


class TestProcessInteractionEvent:

    def _make_twin_with_topic(self, topic_id="math_101"):
        """Helper: create a twin with one registered topic."""
        twin = NeuralDigitalTwin(
            student_id="test_student_001",
            learning_style="adaptive",
        )
        twin.register_topic(topic_id, "Math 101", n_neurons=10)
        return twin

    def test_process_interaction_event_answer_submitted(self):
        """Verify mastery update and prediction returned on answer_submitted."""
        twin = self._make_twin_with_topic("math_101")
        initial_mastery = twin.mastery_scores["math_101"]

        event = {
            "type": "answer_submitted",
            "topic_id": "math_101",
            "is_correct": True,
            "response_time_ms": 1500,
            "difficulty": 0.6,
            "timestamp": datetime.now().isoformat(),
        }

        result = twin.process_interaction_event(event)

        # Mastery should increase on correct answer
        assert twin.mastery_scores["math_101"] > initial_mastery

        # Result structure must include cognitive_prediction
        assert "cognitive_prediction" in result
        pred = result["cognitive_prediction"]
        assert "predicted_confusion" in pred
        assert "predicted_attention" in pred
        assert "optimal_difficulty" in pred
        assert "recommended_action" in pred
        assert pred["recommended_action"] in (
            "continue", "review", "break", "switch_format"
        )

        # Result must include neural_metrics and mastery_scores
        assert "neural_metrics" in result
        assert "mastery_scores" in result
        assert "math_101" in result["mastery_scores"]

    def test_answer_submitted_incorrect_decreases_mastery(self):
        """Verify incorrect answer decreases mastery."""
        twin = self._make_twin_with_topic("math_101")
        twin.mastery_scores["math_101"] = 0.5

        event = {
            "type": "answer_submitted",
            "topic_id": "math_101",
            "is_correct": False,
            "difficulty": 0.3,
        }

        twin.process_interaction_event(event)

        assert twin.mastery_scores["math_101"] < 0.5

    def test_answer_submitted_unregistered_topic_no_crash(self):
        """Answer for unregistered topic should not crash."""
        twin = self._make_twin_with_topic("math_101")

        event = {
            "type": "answer_submitted",
            "topic_id": "unknown_topic",
            "is_correct": True,
            "difficulty": 0.5,
        }

        # Should not raise
        result = twin.process_interaction_event(event)
        assert "cognitive_prediction" in result

    def test_process_interaction_event_lesson_completed(self):
        """Verify lesson_completed records learning event and updates mastery."""
        twin = NeuralDigitalTwin(
            student_id="test_student_002",
            learning_style="visual",
        )

        event = {
            "type": "lesson_completed",
            "topic_id": "physics_201",
            "difficulty": 0.4,
            "timestamp": datetime.now().isoformat(),
        }

        initial_event_count = len(twin.learning_events)
        result = twin.process_interaction_event(event)

        # Learning event should be appended
        assert len(twin.learning_events) == initial_event_count + 1
        last_event = twin.learning_events[-1]
        assert last_event["type"] == "lesson_completed"
        assert last_event["topic_id"] == "physics_201"

        # Topic should be auto-registered and have mastery > 0
        assert "physics_201" in twin.mastery_scores
        assert twin.mastery_scores["physics_201"] > 0.0

        # Result should have prediction
        assert "cognitive_prediction" in result

    def test_session_start_event_recorded(self):
        """Session start events should be recorded without mastery changes."""
        twin = self._make_twin_with_topic("math_101")
        initial_mastery = twin.mastery_scores["math_101"]

        event = {
            "type": "session_start",
            "topic_id": "math_101",
        }

        result = twin.process_interaction_event(event)

        # Mastery should not change on session_start
        assert twin.mastery_scores["math_101"] == initial_mastery
        assert len(twin.learning_events) > 0
        assert "cognitive_prediction" in result

    def test_event_always_appended(self):
        """Every event type should be appended to learning_events."""
        twin = self._make_twin_with_topic("math_101")

        for event_type in ["session_start", "session_end", "answer_submitted", "lesson_completed"]:
            event = {"type": event_type, "topic_id": "math_101", "difficulty": 0.5}
            twin.process_interaction_event(event)

        assert len(twin.learning_events) >= 4


class TestGetEnhancedCognitiveState:

    def test_get_enhanced_cognitive_state(self):
        """Verify all expected fields are present in enhanced state."""
        twin = NeuralDigitalTwin(
            student_id="test_student_003",
            learning_style="adaptive",
        )
        twin.register_topic("topic_a", "Topic A")
        twin.register_topic("topic_b", "Topic B")
        twin.mastery_scores["topic_a"] = 0.6
        twin.mastery_scores["topic_b"] = 0.3

        state = twin.get_enhanced_cognitive_state()

        # Base cognitive state fields
        assert "student_id" in state
        assert state["student_id"] == "test_student_003"
        assert "overall_mastery" in state
        assert "topics_learned" in state
        assert "learning_velocity" in state
        assert "total_learning_events" in state
        assert "neural_metrics" in state
        assert "last_updated" in state

        # Enhanced fields
        assert "predicted_performance_next" in state
        assert isinstance(state["predicted_performance_next"], float)
        assert 0.0 <= state["predicted_performance_next"] <= 1.0

        assert "recommended_break_in_minutes" in state
        # Can be None or float
        assert state["recommended_break_in_minutes"] is None or isinstance(
            state["recommended_break_in_minutes"], float
        )

        assert "topic_readiness" in state
        assert isinstance(state["topic_readiness"], dict)
        assert "topic_a" in state["topic_readiness"]
        assert "topic_b" in state["topic_readiness"]
        for readiness in state["topic_readiness"].values():
            assert 0.0 <= readiness <= 1.0

        assert "session_events_count" in state
        assert isinstance(state["session_events_count"], int)
        assert state["session_events_count"] >= 0

    def test_enhanced_state_no_topics(self):
        """Enhanced state with no topics should return sensible defaults."""
        twin = NeuralDigitalTwin(student_id="empty_student")
        state = twin.get_enhanced_cognitive_state()

        assert state["predicted_performance_next"] == 0.0
        assert state["topic_readiness"] == {}
        assert state["session_events_count"] == 0

    def test_enhanced_state_break_recommendation(self):
        """Verify break recommendation when many recent events."""
        twin = NeuralDigitalTwin(student_id="busy_student")
        twin.register_topic("topic_x", "Topic X")

        # Add many events with recent timestamps
        for i in range(25):
            event = {
                "type": "answer_submitted",
                "topic_id": "topic_x",
                "is_correct": True,
                "difficulty": 0.5,
                "timestamp": datetime.now().isoformat(),
            }
            twin.process_interaction_event(event)

        state = twin.get_enhanced_cognitive_state()

        # Should recommend a break after 20+ events
        assert state["recommended_break_in_minutes"] is not None
        assert state["recommended_break_in_minutes"] > 0


class TestGetOptimalDifficulty:

    def test_get_optimal_difficulty_known_topic(self):
        """Verify returns float 0-1 for a known topic."""
        twin = NeuralDigitalTwin(student_id="test_student_004")
        twin.register_topic("algebra", "Algebra")
        twin.mastery_scores["algebra"] = 0.7

        difficulty = twin.get_optimal_difficulty("algebra")

        assert isinstance(difficulty, float)
        assert 0.0 <= difficulty <= 1.0
        # Expected: 0.7 * 0.8 + 0.1 = 0.66
        expected = 0.7 * 0.8 + 0.1
        assert abs(difficulty - expected) < 1e-6

    def test_get_optimal_difficulty_unknown_topic(self):
        """Verify returns default (0.5) for an unknown topic."""
        twin = NeuralDigitalTwin(student_id="test_student_005")

        difficulty = twin.get_optimal_difficulty("nonexistent_topic")

        assert isinstance(difficulty, float)
        # Default mastery is 0.5, so: 0.5 * 0.8 + 0.1 = 0.5
        expected = 0.5 * 0.8 + 0.1
        assert abs(difficulty - expected) < 1e-6

    def test_get_optimal_difficulty_floor(self):
        """Verify difficulty floor at 0.1."""
        twin = NeuralDigitalTwin(student_id="test_student_006")
        twin.register_topic("easy", "Easy Topic")
        twin.mastery_scores["easy"] = 0.0

        difficulty = twin.get_optimal_difficulty("easy")

        # 0.0 * 0.8 + 0.1 = 0.1 (floor)
        assert difficulty == pytest.approx(0.1, abs=1e-6)

    def test_get_optimal_difficulty_cap(self):
        """Verify difficulty cap at 0.9."""
        twin = NeuralDigitalTwin(student_id="test_student_007")
        twin.register_topic("hard", "Hard Topic")
        twin.mastery_scores["hard"] = 1.0

        difficulty = twin.get_optimal_difficulty("hard")

        # 1.0 * 0.8 + 0.1 = 0.9 (cap)
        assert difficulty == pytest.approx(0.9, abs=1e-6)


class TestBatchEvents:

    def test_batch_events_cumulative(self):
        """Verify multiple events accumulate correctly."""
        twin = NeuralDigitalTwin(student_id="test_student_008")
        twin.register_topic("chemistry", "Chemistry")

        initial_mastery = twin.mastery_scores["chemistry"]
        initial_event_count = len(twin.learning_events)

        # Send a batch of correct answers
        events = [
            {
                "type": "answer_submitted",
                "topic_id": "chemistry",
                "is_correct": True,
                "difficulty": 0.5,
                "timestamp": datetime.now().isoformat(),
            }
            for _ in range(5)
        ]

        results = []
        for event in events:
            result = twin.process_interaction_event(event)
            results.append(result)

        # All 5 events should be recorded
        assert len(twin.learning_events) == initial_event_count + 5

        # Mastery should have increased cumulatively
        assert twin.mastery_scores["chemistry"] > initial_mastery

        # Each result should be valid
        for result in results:
            assert "cognitive_prediction" in result
            assert "mastery_scores" in result
            assert "neural_metrics" in result

    def test_batch_mixed_events_accumulate(self):
        """Verify mixed correct/incorrect events accumulate properly."""
        twin = NeuralDigitalTwin(student_id="test_student_009")
        twin.register_topic("biology", "Biology")
        twin.mastery_scores["biology"] = 0.5

        # Mix of correct and incorrect
        events = [
            {"type": "answer_submitted", "topic_id": "biology", "is_correct": True, "difficulty": 0.5},
            {"type": "answer_submitted", "topic_id": "biology", "is_correct": True, "difficulty": 0.5},
            {"type": "answer_submitted", "topic_id": "biology", "is_correct": False, "difficulty": 0.5},
            {"type": "answer_submitted", "topic_id": "biology", "is_correct": True, "difficulty": 0.5},
        ]

        for event in events:
            twin.process_interaction_event(event)

        # With 3 correct and 1 incorrect, mastery should still increase overall
        # (gain per correct > loss per incorrect at default difficulty)
        assert twin.mastery_scores["biology"] > 0.5

    def test_batch_mastery_scores_monotonic_on_correct(self):
        """Verify mastery increases monotonically with consecutive correct answers."""
        twin = NeuralDigitalTwin(student_id="test_student_010")
        twin.register_topic("history", "History")

        mastery_history = [twin.mastery_scores["history"]]

        for _ in range(10):
            event = {
                "type": "answer_submitted",
                "topic_id": "history",
                "is_correct": True,
                "difficulty": 0.5,
            }
            twin.process_interaction_event(event)
            mastery_history.append(twin.mastery_scores["history"])

        # Each mastery should be >= previous
        for i in range(1, len(mastery_history)):
            assert mastery_history[i] >= mastery_history[i - 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
