"""Tests for communication layer."""
import pytest
import numpy as np

from magicbrain.platform.communication import (
    Message,
    MessageType,
    MessagePriority,
    MessageBus,
    Topic,
    TypeConverter,
    ConverterRegistry,
    SpikesToDenseConverter,
    DenseToSpikesConverter,
)
from magicbrain.platform import OutputType


class TestMessage:
    """Tests for Message class."""

    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(
            source="model1",
            target="model2",
            data=np.array([1, 2, 3]),
            message_type=MessageType.DATA
        )

        assert msg.source == "model1"
        assert msg.target == "model2"
        assert msg.message_type == MessageType.DATA
        assert msg.priority == MessagePriority.NORMAL
        assert msg.data_type == "numpy"
        assert msg.data_shape == (3,)

    def test_message_to_dict(self):
        """Test converting message to dict."""
        msg = Message(
            source="model1",
            target="model2",
            data="test"
        )

        d = msg.to_dict()
        assert d["source"] == "model1"
        assert d["target"] == "model2"
        assert d["message_type"] == "data"

    def test_control_message(self):
        """Test control message."""
        from magicbrain.platform.communication.message import ControlMessage

        msg = ControlMessage(
            source="orchestrator",
            target="model1",
            command="start"
        )

        assert msg.is_control_message()
        assert msg.command == "start"


class TestMessageBus:
    """Tests for MessageBus."""

    def test_create_topic(self):
        """Test creating a topic."""
        bus = MessageBus()
        topic = bus.create_topic("test_topic", "Test topic")

        assert topic.name == "test_topic"
        assert topic.description == "Test topic"

    def test_subscribe_and_publish(self):
        """Test subscribing and publishing."""
        bus = MessageBus()
        received_messages = []

        def callback(msg):
            received_messages.append(msg)

        # Subscribe
        bus.subscribe("subscriber1", "topic1", callback)

        # Publish
        msg = Message(
            source="model1",
            target="model2",
            data="test",
            topic="topic1"
        )
        bus.publish(msg)

        # Check received
        assert len(received_messages) == 1
        assert received_messages[0].data == "test"

    def test_direct_routing(self):
        """Test direct point-to-point routing."""
        bus = MessageBus()
        received_messages = []

        def callback(msg):
            received_messages.append(msg)

        # Create route
        bus.route("model1", "model2", callback)

        # Send message
        msg = Message(
            source="model1",
            target="model2",
            data="direct"
        )
        bus.publish(msg)

        # Check received
        assert len(received_messages) == 1
        assert received_messages[0].data == "direct"

    def test_unsubscribe(self):
        """Test unsubscribing."""
        bus = MessageBus()
        received_count = [0]

        def callback(msg):
            received_count[0] += 1

        bus.subscribe("sub1", "topic1", callback)

        # Publish first message
        bus.publish(Message(source="m1", target="m2", data="msg1", topic="topic1"))
        assert received_count[0] == 1

        # Unsubscribe
        bus.unsubscribe("sub1", "topic1")

        # Publish second message (should not be received)
        bus.publish(Message(source="m1", target="m2", data="msg2", topic="topic1"))
        assert received_count[0] == 1


class TestTypeConverters:
    """Tests for type converters."""

    def test_spikes_to_dense_rate(self):
        """Test spike to dense conversion (rate method)."""
        converter = SpikesToDenseConverter(method="rate", time_window=10)

        # Create spike train (T=20, N=5)
        spikes = np.random.rand(20, 5) < 0.3
        spikes = spikes.astype(np.float32)

        # Convert
        dense = converter.convert(spikes)

        assert dense.shape == (5,)
        assert np.all(dense >= 0)
        assert np.all(dense <= 1)

    def test_spikes_to_dense_sum(self):
        """Test spike to dense conversion (sum method)."""
        converter = SpikesToDenseConverter(method="sum")

        spikes = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        dense = converter.convert(spikes)

        assert dense.shape == (3,)
        assert dense[0] == 2  # Two spikes
        assert dense[1] == 2
        assert dense[2] == 1

    def test_dense_to_spikes(self):
        """Test dense to spikes conversion."""
        converter = DenseToSpikesConverter(method="threshold", duration=5)

        dense = np.array([0.8, 0.3, 0.6])
        spikes = converter.convert(dense, threshold=0.5)

        assert spikes.shape == (5, 3)
        # First and third neurons should spike (above threshold)
        assert spikes[0, 0] == 1.0
        assert spikes[0, 1] == 0.0
        assert spikes[0, 2] == 1.0

    def test_converter_registry(self):
        """Test converter registry."""
        registry = ConverterRegistry()

        # Check default converters registered
        assert registry.has_converter(OutputType.SPIKES, OutputType.DENSE)
        assert registry.has_converter(OutputType.DENSE, OutputType.SPIKES)
        assert registry.has_converter(OutputType.LOGITS, OutputType.PROBABILITY)

    def test_registry_convert(self):
        """Test converting through registry."""
        registry = ConverterRegistry()

        # Convert spikes to dense
        spikes = np.random.rand(10, 5) < 0.5
        spikes = spikes.astype(np.float32)

        dense = registry.convert(spikes, OutputType.SPIKES, OutputType.DENSE)

        assert isinstance(dense, np.ndarray)
        assert dense.shape == (5,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
