"""
Neural Digital Twin - cognitive state modeling for students.

Each student has a unique spiking neural network that models their
cognitive state, knowledge mastery, and learning dynamics.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import hashlib

from ..brain import TextBrain
from ..learning_rules import STDPBrain
from ..diagnostics import LiveMonitor


class NeuralDigitalTwin:
    """
    Neural Digital Twin for a student.

    Models student's cognitive state using a spiking neural network.
    Knowledge mastery is encoded in neural activity patterns.
    """

    def __init__(
        self,
        student_id: str,
        learning_style: str = "adaptive",
        use_stdp: bool = False,
        initial_mastery: Optional[Dict[str, float]] = None,
    ):
        """
        Create Neural Digital Twin for a student.

        Args:
            student_id: Unique student identifier
            learning_style: Learning style (adaptive, visual, kinesthetic, etc.)
            use_stdp: Use STDP learning instead of dopamine
            initial_mastery: Initial mastery levels for topics
        """
        self.student_id = student_id
        self.learning_style = learning_style
        self.use_stdp = use_stdp
        self.created_at = datetime.now()
        self.last_updated = datetime.now()

        # Generate student-specific genome from ID
        self.genome = self._generate_student_genome(student_id, learning_style)

        # Create brain (start with small vocabulary, expand as needed)
        vocab_size = 100  # Will expand as student learns
        if use_stdp:
            self.brain = STDPBrain(self.genome, vocab_size, stdp_type="triplet")
        else:
            self.brain = TextBrain(self.genome, vocab_size)

        # Monitor for tracking cognitive state
        self.monitor = LiveMonitor(log_every=10)

        # Topic to neuron mapping
        self.topic_neurons: Dict[str, List[int]] = {}

        # Mastery scores (topic_id -> mastery level 0-1)
        self.mastery_scores: Dict[str, float] = initial_mastery or {}

        # Learning history
        self.learning_events: List[Dict] = []

        # Forgetting parameters
        self.forgetting_rate = 0.01  # Per day without practice
        self.last_practice: Dict[str, datetime] = {}

    def _generate_student_genome(self, student_id: str, learning_style: str) -> str:
        """
        Generate unique genome for student based on ID and learning style.

        Args:
            student_id: Student identifier
            learning_style: Learning style preference

        Returns:
            Genome string
        """
        # Hash student ID to get consistent genome
        hash_obj = hashlib.sha256(student_id.encode())
        hash_bytes = hash_obj.digest()

        # Convert to base-4 (0-3) genome string
        genome_chars = []
        for byte in hash_bytes[:25]:  # Use first 25 bytes
            # Convert byte (0-255) to two base-4 digits
            genome_chars.append(str((byte >> 4) % 4))
            genome_chars.append(str(byte % 4))

        genome = ''.join(genome_chars[:25])  # Take first 25 characters

        # Modify based on learning style
        if learning_style == "visual":
            # Visual learners: more neurons, slower dynamics
            genome = '3' + genome[1:]  # Increase N
        elif learning_style == "kinesthetic":
            # Kinesthetic: faster learning, more plasticity
            genome = genome[:4] + '3' + genome[5:]  # Increase learning rate
        elif learning_style == "auditory":
            # Auditory: balanced dynamics
            pass  # Use default

        return genome

    def register_topic(self, topic_id: str, topic_name: str, n_neurons: int = 10):
        """
        Register a topic and assign neurons to it.

        Args:
            topic_id: Unique topic identifier
            topic_name: Human-readable topic name
            n_neurons: Number of neurons to assign to this topic
        """
        # Randomly select neurons for this topic
        available_neurons = set(range(self.brain.N)) - set(
            neuron for neurons in self.topic_neurons.values() for neuron in neurons
        )

        if len(available_neurons) < n_neurons:
            # Need more neurons, expand brain (would need reinitialization)
            n_neurons = len(available_neurons)

        topic_neurons = list(np.random.choice(
            list(available_neurons),
            size=n_neurons,
            replace=False
        ))

        self.topic_neurons[topic_id] = topic_neurons
        self.mastery_scores[topic_id] = 0.0
        self.last_practice[topic_id] = datetime.now()

    def learn_topic(
        self,
        topic_id: str,
        learning_data: str,
        steps: int = 100,
        difficulty: float = 0.5,
    ) -> Dict:
        """
        Student learns about a topic.

        Args:
            topic_id: Topic being learned
            learning_data: Text representation of learning material
            steps: Number of learning steps
            difficulty: Task difficulty (0=easy, 1=hard)

        Returns:
            Learning result with mastery change
        """
        if topic_id not in self.topic_neurons:
            raise ValueError(f"Topic {topic_id} not registered")

        # Build vocab from learning data
        from ..tasks.text_task import build_vocab, train_loop_with_history

        stoi, itos = build_vocab(learning_data)

        # Update brain vocab size if needed
        if len(stoi) > self.brain.vocab_size:
            # Would need to expand brain (simplified here)
            pass

        # Train on this material
        initial_loss = None
        losses = []

        for step in range(steps):
            # Sample from learning data
            idx = step % (len(learning_data) - 1)
            if learning_data[idx] in stoi and learning_data[idx + 1] in stoi:
                x = stoi[learning_data[idx]]
                y = stoi[learning_data[idx + 1]]

                probs = self.brain.forward(x)
                loss = self.brain.learn(y, probs)

                if initial_loss is None:
                    initial_loss = loss

                losses.append(loss)

                # Record metrics
                if step % 10 == 0:
                    self.monitor.record(self.brain, loss, step)

        final_loss = np.mean(losses[-10:]) if losses else 0.0

        # Update mastery score based on learning
        old_mastery = self.mastery_scores.get(topic_id, 0.0)

        # Mastery increases based on learning progress
        if initial_loss and losses:
            learning_gain = (initial_loss - final_loss) / (initial_loss + 1e-6)
            learning_gain = np.clip(learning_gain, -0.2, 0.5)
            # Adjust by difficulty
            learning_gain *= (1.0 - difficulty * 0.5)
        else:
            # Minimal gain even if no effective learning occurred
            learning_gain = 0.05

        # Ensure minimum progress for any learning activity
        learning_gain = max(learning_gain * 0.1, 0.01)

        new_mastery = np.clip(old_mastery + learning_gain, 0.0, 1.0)
        self.mastery_scores[topic_id] = new_mastery

        # Update last practice time
        self.last_practice[topic_id] = datetime.now()
        self.last_updated = datetime.now()

        # Record learning event
        event = {
            "timestamp": datetime.now(),
            "topic_id": topic_id,
            "steps": steps,
            "difficulty": difficulty,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "mastery_change": new_mastery - old_mastery,
            "new_mastery": new_mastery,
        }
        self.learning_events.append(event)

        return event

    def assess_mastery(self, topic_id: str) -> Dict:
        """
        Assess current mastery level for a topic.

        Args:
            topic_id: Topic to assess

        Returns:
            Mastery assessment
        """
        if topic_id not in self.topic_neurons:
            return {"mastery": 0.0, "confidence": 0.0, "needs_review": True}

        # Get current mastery score
        mastery = self.mastery_scores.get(topic_id, 0.0)

        # Apply forgetting
        mastery = self._apply_forgetting(topic_id, mastery)

        # Compute confidence based on neural activity
        topic_neurons = self.topic_neurons[topic_id]
        neuron_activity = self.brain.a[topic_neurons]
        confidence = float(np.mean(neuron_activity > 0))

        # Needs review if mastery low or not practiced recently
        days_since_practice = (datetime.now() - self.last_practice.get(
            topic_id, datetime.now()
        )).days
        needs_review = mastery < 0.7 or days_since_practice > 7

        return {
            "topic_id": topic_id,
            "mastery": float(mastery),
            "confidence": confidence,
            "needs_review": needs_review,
            "days_since_practice": days_since_practice,
            "neural_activity": float(np.mean(neuron_activity)),
        }

    def _apply_forgetting(self, topic_id: str, current_mastery: float) -> float:
        """
        Apply forgetting curve to mastery score.

        Args:
            topic_id: Topic ID
            current_mastery: Current mastery level

        Returns:
            Adjusted mastery after forgetting
        """
        if topic_id not in self.last_practice:
            return current_mastery

        days_since_practice = (datetime.now() - self.last_practice[topic_id]).days

        # Exponential forgetting
        forgotten_amount = current_mastery * (1.0 - np.exp(-self.forgetting_rate * days_since_practice))

        new_mastery = current_mastery - forgotten_amount

        # Update stored mastery
        self.mastery_scores[topic_id] = max(0.0, new_mastery)

        return new_mastery

    def predict_performance(
        self,
        topic_id: str,
        difficulty: float = 0.5,
    ) -> Dict:
        """
        Predict student performance on a task.

        Args:
            topic_id: Topic of the task
            difficulty: Task difficulty

        Returns:
            Performance prediction
        """
        mastery_info = self.assess_mastery(topic_id)
        mastery = mastery_info["mastery"]

        # Predict success probability
        # Higher mastery, lower difficulty -> higher success
        base_success = mastery
        difficulty_penalty = difficulty * 0.3
        success_probability = np.clip(base_success - difficulty_penalty, 0.0, 1.0)

        # Predict time needed (inverse of mastery)
        base_time = 60  # seconds
        time_estimate = base_time * (2.0 - mastery) * (1.0 + difficulty)

        return {
            "topic_id": topic_id,
            "success_probability": float(success_probability),
            "estimated_time_seconds": float(time_estimate),
            "current_mastery": mastery,
            "confidence": mastery_info["confidence"],
            "recommendation": self._get_recommendation(mastery, difficulty),
        }

    def _get_recommendation(self, mastery: float, difficulty: float) -> str:
        """Get learning recommendation based on mastery and difficulty."""
        if mastery < 0.3:
            return "review_basics"
        elif mastery < 0.6:
            if difficulty > 0.7:
                return "too_difficult"
            else:
                return "continue_practice"
        elif mastery < 0.8:
            return "ready_for_challenge"
        else:
            return "mastered"

    def get_cognitive_state(self) -> Dict:
        """
        Get complete cognitive state snapshot.

        Returns:
            Cognitive state summary
        """
        # Overall mastery
        overall_mastery = np.mean(list(self.mastery_scores.values())) if self.mastery_scores else 0.0

        # Neural metrics
        neural_metrics = {
            "firing_rate": float(np.mean(self.brain.a)),
            "mean_theta": float(np.mean(self.brain.theta)),
            "mean_weight": float(np.mean(np.abs(self.brain.w_slow + self.brain.w_fast))),
        }

        # Learning velocity (recent mastery changes)
        recent_events = self.learning_events[-10:] if len(self.learning_events) > 0 else []
        learning_velocity = np.mean([
            e["mastery_change"] for e in recent_events
        ]) if recent_events else 0.0

        return {
            "student_id": self.student_id,
            "overall_mastery": float(overall_mastery),
            "topics_learned": len(self.mastery_scores),
            "learning_velocity": float(learning_velocity),
            "total_learning_events": len(self.learning_events),
            "neural_metrics": neural_metrics,
            "last_updated": self.last_updated.isoformat(),
        }

    def save_state(self, filepath: str):
        """Save twin state to file."""
        from ..io import save_model
        import json

        # Save brain
        brain_path = filepath.replace(".json", "_brain.npz")
        stoi = {str(i): i for i in range(self.brain.vocab_size)}
        itos = {i: str(i) for i in range(self.brain.vocab_size)}
        save_model(self.brain, stoi, itos, brain_path)

        # Save metadata
        metadata = {
            "student_id": self.student_id,
            "learning_style": self.learning_style,
            "genome": self.genome,
            "topic_neurons": {k: [int(n) for n in v] for k, v in self.topic_neurons.items()},
            "mastery_scores": {k: float(v) for k, v in self.mastery_scores.items()},
            "learning_events": [
                {
                    "timestamp": e["timestamp"].isoformat(),
                    "topic_id": e["topic_id"],
                    "mastery_change": float(e["mastery_change"]),
                }
                for e in self.learning_events[-100:]  # Last 100 events
            ],
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load_state(cls, filepath: str) -> 'NeuralDigitalTwin':
        """Load twin state from file."""
        import json
        from ..io import load_model

        # Load metadata
        with open(filepath, 'r') as f:
            metadata = json.load(f)

        # Create twin
        twin = cls(
            student_id=metadata["student_id"],
            learning_style=metadata["learning_style"],
        )

        # Load brain
        brain_path = filepath.replace(".json", "_brain.npz")
        brain, stoi, itos = load_model(brain_path)
        twin.brain = brain
        twin.genome = metadata["genome"]

        # Restore state
        twin.topic_neurons = {k: list(v) for k, v in metadata["topic_neurons"].items()}
        twin.mastery_scores = metadata["mastery_scores"]
        twin.created_at = datetime.fromisoformat(metadata["created_at"])
        twin.last_updated = datetime.fromisoformat(metadata["last_updated"])

        return twin
