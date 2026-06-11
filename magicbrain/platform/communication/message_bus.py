"""
Message bus for pub/sub communication between models.

Enables asynchronous, decoupled messaging between models in the platform.
"""
from __future__ import annotations
from collections import defaultdict, deque
from typing import Callable, Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
import asyncio
import threading
from datetime import datetime

from .message import Message, MessageType, MessagePriority


@dataclass
class Topic:
    """
    Message topic for pub/sub.

    Groups related messages and manages subscribers.
    """
    name: str
    description: str = ""
    message_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    # Statistics
    total_messages: int = 0
    last_message_time: Optional[datetime] = None

    def __repr__(self) -> str:
        return f"Topic(name={self.name}, messages={self.total_messages})"


class MessageBus:
    """
    Message bus for inter-model communication.

    Implements pub/sub pattern with topics, priorities, and async support.
    """

    def __init__(self, max_queue_size: int = 1000):
        """
        Initialize message bus.

        Args:
            max_queue_size: Maximum messages per subscriber queue
        """
        self.max_queue_size = max_queue_size

        # Topics
        self._topics: Dict[str, Topic] = {}

        # Subscribers: topic -> set of (subscriber_id, callback)
        self._subscribers: Dict[str, List[tuple[str, Callable]]] = defaultdict(list)

        # Message queues: subscriber_id -> deque of messages
        self._queues: Dict[str, deque[Message]] = defaultdict(
            lambda: deque(maxlen=max_queue_size)
        )

        # Direct routes: source -> target -> callback
        self._routes: Dict[str, Dict[str, Callable]] = defaultdict(dict)

        # Message history (for debugging)
        self._history: deque[Message] = deque(maxlen=100)

        # Statistics
        self._total_published = 0
        self._total_delivered = 0

        # Thread safety
        self._lock = threading.RLock()

        # Async support
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    def create_topic(self, name: str, description: str = "") -> Topic:
        """
        Create a new topic.

        Args:
            name: Topic name
            description: Topic description

        Returns:
            Created topic
        """
        with self._lock:
            if name in self._topics:
                return self._topics[name]

            topic = Topic(name=name, description=description)
            self._topics[name] = topic
            return topic

    def get_topic(self, name: str) -> Optional[Topic]:
        """
        Get topic by name.

        Args:
            name: Topic name

        Returns:
            Topic or None if not found
        """
        return self._topics.get(name)

    def list_topics(self) -> List[Topic]:
        """
        List all topics.

        Returns:
            List of topics
        """
        return list(self._topics.values())

    def subscribe(
        self,
        subscriber_id: str,
        topic: str,
        callback: Callable[[Message], None]
    ):
        """
        Subscribe to a topic.

        Args:
            subscriber_id: Unique subscriber ID
            topic: Topic name
            callback: Callback function for messages
        """
        with self._lock:
            # Create topic if doesn't exist
            if topic not in self._topics:
                self.create_topic(topic)

            # Add subscriber
            self._subscribers[topic].append((subscriber_id, callback))

    def unsubscribe(self, subscriber_id: str, topic: str):
        """
        Unsubscribe from a topic.

        Args:
            subscriber_id: Subscriber ID
            topic: Topic name
        """
        with self._lock:
            if topic in self._subscribers:
                self._subscribers[topic] = [
                    (sid, cb) for sid, cb in self._subscribers[topic]
                    if sid != subscriber_id
                ]

    def unsubscribe_all(self, subscriber_id: str):
        """
        Unsubscribe from all topics.

        Args:
            subscriber_id: Subscriber ID
        """
        with self._lock:
            for topic in self._subscribers:
                self.unsubscribe(subscriber_id, topic)

    def publish(self, message: Message):
        """
        Publish a message to a topic or direct route.

        Args:
            message: Message to publish
        """
        with self._lock:
            self._total_published += 1
            self._history.append(message)

            # Direct routing (point-to-point)
            if message.target and message.target in self._routes.get(message.source, {}):
                callback = self._routes[message.source][message.target]
                self._deliver_message(message, callback)
                return

            # Topic-based routing (pub/sub)
            if message.topic:
                if message.topic in self._subscribers:
                    # Update topic stats
                    topic = self._topics.get(message.topic)
                    if topic:
                        topic.total_messages += 1
                        topic.last_message_time = datetime.now()

                    # Deliver to all subscribers
                    for subscriber_id, callback in self._subscribers[message.topic]:
                        # Add to subscriber queue
                        self._queues[subscriber_id].append(message)
                        self._deliver_message(message, callback)

            # Broadcast if no topic (deliver to all routes)
            elif message.target == "*":
                for target_routes in self._routes.values():
                    for callback in target_routes.values():
                        self._deliver_message(message, callback)

    def _deliver_message(self, message: Message, callback: Callable):
        """
        Deliver message to callback.

        Args:
            message: Message to deliver
            callback: Callback function
        """
        try:
            self._total_delivered += 1

            # Async callback
            if asyncio.iscoroutinefunction(callback):
                if self._event_loop:
                    asyncio.run_coroutine_threadsafe(callback(message), self._event_loop)
                else:
                    # Create task in current event loop if available
                    try:
                        asyncio.create_task(callback(message))
                    except RuntimeError:
                        # No event loop, call synchronously
                        asyncio.run(callback(message))
            # Sync callback
            else:
                callback(message)

        except Exception as e:
            # Log error but don't crash
            print(f"Error delivering message: {e}")

    def route(
        self,
        source: str,
        target: str,
        callback: Callable[[Message], None]
    ):
        """
        Create direct route from source to target.

        Args:
            source: Source model ID
            target: Target model ID
            callback: Callback for messages
        """
        with self._lock:
            self._routes[source][target] = callback

    def unroute(self, source: str, target: str):
        """
        Remove direct route.

        Args:
            source: Source model ID
            target: Target model ID
        """
        with self._lock:
            if source in self._routes:
                self._routes[source].pop(target, None)

    def send(self, message: Message):
        """
        Send message (alias for publish).

        Args:
            message: Message to send
        """
        self.publish(message)

    def get_queue(self, subscriber_id: str) -> deque[Message]:
        """
        Get message queue for subscriber.

        Args:
            subscriber_id: Subscriber ID

        Returns:
            Message queue
        """
        return self._queues[subscriber_id]

    def clear_queue(self, subscriber_id: str):
        """
        Clear message queue for subscriber.

        Args:
            subscriber_id: Subscriber ID
        """
        self._queues[subscriber_id].clear()

    def get_history(self) -> List[Message]:
        """
        Get message history.

        Returns:
            List of recent messages
        """
        return list(self._history)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get message bus statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            return {
                "total_published": self._total_published,
                "total_delivered": self._total_delivered,
                "topics_count": len(self._topics),
                "subscribers_count": sum(len(subs) for subs in self._subscribers.values()),
                "routes_count": sum(len(routes) for routes in self._routes.values()),
                "queued_messages": sum(len(q) for q in self._queues.values()),
            }

    def reset(self):
        """Reset message bus (clear all state)."""
        with self._lock:
            self._topics.clear()
            self._subscribers.clear()
            self._queues.clear()
            self._routes.clear()
            self._history.clear()
            self._total_published = 0
            self._total_delivered = 0

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """
        Set event loop for async message delivery.

        Args:
            loop: Event loop
        """
        self._event_loop = loop

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"MessageBus(topics={stats['topics_count']}, "
            f"subscribers={stats['subscribers_count']}, "
            f"published={stats['total_published']})"
        )


class PriorityMessageBus(MessageBus):
    """
    Message bus with priority-based delivery.

    Higher priority messages are delivered first.
    """

    def __init__(self, max_queue_size: int = 1000):
        """
        Initialize priority message bus.

        Args:
            max_queue_size: Maximum messages per subscriber queue
        """
        super().__init__(max_queue_size)

        # Priority queues: subscriber_id -> priority -> deque
        self._priority_queues: Dict[str, Dict[int, deque[Message]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=max_queue_size))
        )

    def publish(self, message: Message):
        """
        Publish message with priority handling.

        Args:
            message: Message to publish
        """
        # Use priority queues for topic-based messages
        if message.topic:
            with self._lock:
                self._total_published += 1
                self._history.append(message)

                if message.topic in self._subscribers:
                    topic = self._topics.get(message.topic)
                    if topic:
                        topic.total_messages += 1
                        topic.last_message_time = datetime.now()

                    for subscriber_id, callback in self._subscribers[message.topic]:
                        # Add to priority queue
                        priority = message.priority.value
                        self._priority_queues[subscriber_id][priority].append(message)

                        # Deliver highest priority message
                        self._deliver_priority_message(subscriber_id, callback)
        else:
            # Fall back to regular delivery for direct routes
            super().publish(message)

    def _deliver_priority_message(self, subscriber_id: str, callback: Callable):
        """
        Deliver highest priority message to subscriber.

        Args:
            subscriber_id: Subscriber ID
            callback: Callback function
        """
        # Find highest priority non-empty queue
        priorities = sorted(self._priority_queues[subscriber_id].keys(), reverse=True)
        for priority in priorities:
            queue = self._priority_queues[subscriber_id][priority]
            if queue:
                message = queue.popleft()
                self._deliver_message(message, callback)
                break


# Global message bus instance
_global_bus = MessageBus()


def get_global_bus() -> MessageBus:
    """
    Get global message bus instance.

    Returns:
        Global message bus
    """
    return _global_bus
