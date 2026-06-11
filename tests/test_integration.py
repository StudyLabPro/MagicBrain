"""
Tests for KnowledgeBaseAI integration.
"""
import pytest
import tempfile
from magicbrain.integration import NeuralDigitalTwin, KnowledgeBaseClient


def test_neural_twin_creation():
    """Test creating a Neural Digital Twin."""
    twin = NeuralDigitalTwin(
        student_id="student_123",
        learning_style="visual",
    )

    assert twin.student_id == "student_123"
    assert twin.learning_style == "visual"
    assert len(twin.genome) == 25
    assert twin.brain is not None


def test_genome_generation():
    """Test genome generation is consistent for same student."""
    twin1 = NeuralDigitalTwin("student_123")
    twin2 = NeuralDigitalTwin("student_123")

    # Same student should get same genome
    assert twin1.genome == twin2.genome

    # Different students get different genomes
    twin3 = NeuralDigitalTwin("student_456")
    assert twin1.genome != twin3.genome


def test_topic_registration():
    """Test registering topics."""
    twin = NeuralDigitalTwin("student_123")

    twin.register_topic("math_algebra", "Algebra Basics", n_neurons=10)

    assert "math_algebra" in twin.topic_neurons
    assert len(twin.topic_neurons["math_algebra"]) == 10
    assert twin.mastery_scores["math_algebra"] == 0.0


def test_learning():
    """Test learning a topic."""
    twin = NeuralDigitalTwin("student_123")
    twin.register_topic("math", "Math Topic")

    # Learn
    result = twin.learn_topic(
        topic_id="math",
        learning_data="2 + 2 = 4, 3 + 3 = 6, 4 + 4 = 8",
        steps=50,
        difficulty=0.3,
    )

    assert "mastery_change" in result
    assert result["new_mastery"] >= 0.0
    assert twin.mastery_scores["math"] > 0.0


def test_mastery_assessment():
    """Test mastery assessment."""
    twin = NeuralDigitalTwin("student_123")
    twin.register_topic("topic1", "Topic 1")

    # Initial assessment
    assessment = twin.assess_mastery("topic1")

    assert assessment["mastery"] == 0.0
    assert assessment["needs_review"] == True
    assert "confidence" in assessment


def test_performance_prediction():
    """Test performance prediction."""
    twin = NeuralDigitalTwin("student_123")
    twin.register_topic("topic1", "Topic 1")

    # Set some mastery
    twin.mastery_scores["topic1"] = 0.7

    prediction = twin.predict_performance("topic1", difficulty=0.5)

    assert "success_probability" in prediction
    assert "estimated_time_seconds" in prediction
    assert "recommendation" in prediction
    assert 0.0 <= prediction["success_probability"] <= 1.0


def test_cognitive_state():
    """Test getting cognitive state."""
    twin = NeuralDigitalTwin("student_123")
    twin.register_topic("topic1", "Topic 1")
    twin.register_topic("topic2", "Topic 2")

    state = twin.get_cognitive_state()

    assert "student_id" in state
    assert "overall_mastery" in state
    assert "topics_learned" in state
    assert state["topics_learned"] == 2


def test_save_load():
    """Test saving and loading twin state."""
    twin = NeuralDigitalTwin("student_123", learning_style="kinesthetic")
    twin.register_topic("math", "Math")
    twin.mastery_scores["math"] = 0.6

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        filepath = f.name

    # Save
    twin.save_state(filepath)

    # Load
    loaded_twin = NeuralDigitalTwin.load_state(filepath)

    assert loaded_twin.student_id == "student_123"
    assert loaded_twin.learning_style == "kinesthetic"
    assert "math" in loaded_twin.mastery_scores
    assert loaded_twin.mastery_scores["math"] == 0.6

    # Cleanup
    import os
    os.unlink(filepath)
    os.unlink(filepath.replace(".json", "_brain.npz"))


def test_knowledgebase_client():
    """Test KnowledgeBaseClient initialization."""
    client = KnowledgeBaseClient(
        base_url="http://localhost:8000",
        api_key="test_key",
    )

    assert client.base_url == "http://localhost:8000"
    assert client.api_key == "test_key"


def test_client_get_or_create_twin():
    """Test getting or creating twin via client."""
    client = KnowledgeBaseClient()

    twin = client.get_or_create_twin("student_123", learning_style="adaptive")

    assert twin.student_id == "student_123"
    assert "student_123" in client.twins

    # Getting again should return same twin
    twin2 = client.get_or_create_twin("student_123")
    assert twin2 is twin


def test_learning_recommendations():
    """Test getting learning recommendations."""
    import asyncio

    client = KnowledgeBaseClient()
    twin = client.get_or_create_twin("student_123")

    # Register some topics with different mastery
    twin.register_topic("topic1", "Topic 1")
    twin.mastery_scores["topic1"] = 0.5  # Learning zone

    twin.register_topic("topic2", "Topic 2")
    twin.mastery_scores["topic2"] = 0.9  # Mastered

    twin.register_topic("topic3", "Topic 3")
    twin.mastery_scores["topic3"] = 0.2  # Struggling

    # Get recommendations
    recommendations = asyncio.run(client.get_learning_recommendations(
        "student_123",
        ["topic1", "topic2", "topic3", "topic4"]  # topic4 is new
    ))

    assert len(recommendations) == 4
    # Should be sorted by priority
    assert recommendations[0]["priority"] >= recommendations[-1]["priority"]


def test_forgetting():
    """Test forgetting over time."""
    from datetime import datetime, timedelta

    twin = NeuralDigitalTwin("student_123")
    twin.register_topic("topic1", "Topic 1")

    # Set mastery and last practice
    twin.mastery_scores["topic1"] = 0.8
    twin.last_practice["topic1"] = datetime.now() - timedelta(days=30)

    # Assess - should show forgetting
    assessment = twin.assess_mastery("topic1")

    # Mastery should be lower due to forgetting
    assert assessment["mastery"] < 0.8
    assert assessment["days_since_practice"] == 30
