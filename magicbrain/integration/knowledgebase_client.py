"""
KnowledgeBaseAI integration client.

Connects Neural Digital Twin with KnowledgeBaseAI for mastery tracking.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import httpx
from .neural_digital_twin import NeuralDigitalTwin


class KnowledgeBaseClient:
    """
    Client for integrating with KnowledgeBaseAI service.

    Provides methods to sync Neural Digital Twin state with
    KnowledgeBase for student mastery tracking.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
    ):
        """
        Initialize client.

        Args:
            base_url: KnowledgeBaseAI service URL
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.twins: Dict[str, NeuralDigitalTwin] = {}

    def get_or_create_twin(
        self,
        student_id: str,
        learning_style: str = "adaptive",
    ) -> NeuralDigitalTwin:
        """
        Get existing twin or create new one for student.

        Args:
            student_id: Student identifier
            learning_style: Learning style preference

        Returns:
            Neural Digital Twin
        """
        if student_id in self.twins:
            return self.twins[student_id]

        # Try to load from KnowledgeBase
        twin = self._load_twin_from_kb(student_id)

        if twin is None:
            # Create new twin
            twin = NeuralDigitalTwin(
                student_id=student_id,
                learning_style=learning_style,
            )

        self.twins[student_id] = twin
        return twin

    def _load_twin_from_kb(self, student_id: str) -> Optional[NeuralDigitalTwin]:
        """
        Load twin state from KnowledgeBase API.

        Makes synchronous HTTP call to retrieve stored twin state.
        Uses graceful degradation - returns None on any error.

        Args:
            student_id: Student ID

        Returns:
            Twin or None if not found/error
        """
        try:
            # Use httpx for synchronous HTTP call
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            with httpx.Client(timeout=5.0) as client:
                response = client.get(
                    f"{self.base_url}/api/v1/neural-twins/{student_id}",
                    headers=headers,
                )

                if response.status_code == 404:
                    # Twin not found - normal case, will create new
                    return None

                response.raise_for_status()
                data = response.json()

                # Reconstruct twin from API data
                twin = NeuralDigitalTwin(
                    student_id=student_id,
                    learning_style=data.get("learning_style", "adaptive"),
                )

                # Restore state
                if "mastery_scores" in data:
                    twin.mastery_scores = data["mastery_scores"]

                if "topic_neurons" in data:
                    twin.topic_neurons = data["topic_neurons"]

                # Restore last practice times
                if "last_practice" in data:
                    from datetime import datetime
                    twin.last_practice = {
                        k: datetime.fromisoformat(v)
                        for k, v in data["last_practice"].items()
                    }

                return twin

        except httpx.TimeoutException:
            # Timeout - graceful degradation
            print(f"Warning: Timeout loading twin for {student_id}")
            return None

        except httpx.HTTPError as e:
            # HTTP error - graceful degradation
            print(f"Warning: HTTP error loading twin: {e}")
            return None

        except Exception as e:
            # Unexpected error - graceful degradation
            print(f"Warning: Failed to load twin from KB: {e}")
            return None

    async def sync_mastery_scores(
        self,
        student_id: str,
        tenant_id: str,
    ) -> Dict:
        """
        Sync mastery scores between twin and KnowledgeBase.

        Args:
            student_id: Student ID
            tenant_id: Tenant/organization ID

        Returns:
            Sync result
        """
        twin = self.twins.get(student_id)
        if twin is None:
            return {"error": "Twin not found"}

        headers = {"X-Tenant-ID": tenant_id}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Get topics from twin
        mastery_data = []
        for topic_id, mastery in twin.mastery_scores.items():
            assessment = twin.assess_mastery(topic_id)
            mastery_data.append({
                "topic_id": topic_id,
                "mastery_score": mastery,
                "confidence": assessment["confidence"],
                "needs_review": assessment["needs_review"],
                "last_practiced": twin.last_practice.get(topic_id).isoformat()
                if topic_id in twin.last_practice else None,
            })

        # Sync with KnowledgeBase
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/student/{student_id}/neural-mastery",
                    json={
                        "mastery_scores": mastery_data,
                        "cognitive_state": twin.get_cognitive_state(),
                    },
                    headers=headers,
                    timeout=10.0,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                return {"error": str(e), "synced": False}

    async def update_from_interaction(
        self,
        student_id: str,
        topic_id: str,
        interaction_type: str,
        content: str,
        difficulty: float = 0.5,
        tenant_id: str = "default",
    ) -> Dict:
        """
        Update twin based on student interaction.

        Args:
            student_id: Student ID
            topic_id: Topic being learned
            interaction_type: Type (study, quiz, review)
            content: Learning content
            difficulty: Task difficulty
            tenant_id: Tenant ID

        Returns:
            Update result
        """
        twin = self.get_or_create_twin(student_id)

        # Register topic if new
        if topic_id not in twin.topic_neurons:
            twin.register_topic(topic_id, topic_id)

        # Determine learning steps based on interaction type
        steps_map = {
            "study": 100,
            "quiz": 50,
            "review": 30,
        }
        steps = steps_map.get(interaction_type, 50)

        # Learn
        result = twin.learn_topic(
            topic_id=topic_id,
            learning_data=content,
            steps=steps,
            difficulty=difficulty,
        )

        # Sync with KnowledgeBase
        await self.sync_mastery_scores(student_id, tenant_id)

        return {
            "student_id": student_id,
            "topic_id": topic_id,
            "mastery_change": result["mastery_change"],
            "new_mastery": result["new_mastery"],
            "synced": True,
        }

    async def get_learning_recommendations(
        self,
        student_id: str,
        available_topics: List[str],
        tenant_id: str = "default",
    ) -> List[Dict]:
        """
        Get personalized learning recommendations.

        Args:
            student_id: Student ID
            available_topics: Topics available for learning
            tenant_id: Tenant ID

        Returns:
            Recommended topics with rationale
        """
        twin = self.twins.get(student_id)
        if twin is None:
            return []

        recommendations = []

        for topic_id in available_topics:
            if topic_id in twin.topic_neurons:
                # Existing topic
                mastery_info = twin.assess_mastery(topic_id)
                prediction = twin.predict_performance(topic_id)

                recommendations.append({
                    "topic_id": topic_id,
                    "priority": self._calculate_priority(mastery_info),
                    "mastery": mastery_info["mastery"],
                    "needs_review": mastery_info["needs_review"],
                    "success_probability": prediction["success_probability"],
                    "recommendation": prediction["recommendation"],
                })
            else:
                # New topic
                recommendations.append({
                    "topic_id": topic_id,
                    "priority": 0.5,  # Medium priority
                    "mastery": 0.0,
                    "needs_review": False,
                    "success_probability": 0.5,
                    "recommendation": "new_topic",
                })

        # Sort by priority (descending)
        recommendations.sort(key=lambda x: x["priority"], reverse=True)

        return recommendations

    def _calculate_priority(self, mastery_info: Dict) -> float:
        """
        Calculate learning priority for a topic.

        Higher priority = more important to practice now.
        """
        mastery = mastery_info["mastery"]
        days_since_practice = mastery_info["days_since_practice"]

        # Priority increases with:
        # - Low mastery (need to learn)
        # - Long time since practice (risk of forgetting)
        # - Medium mastery (in learning zone)

        if mastery < 0.3:
            # Struggling - high priority
            priority = 0.9
        elif mastery < 0.6:
            # Learning zone - highest priority
            priority = 1.0
        elif mastery < 0.8:
            # Good but not mastered
            priority = 0.6
        else:
            # Mastered - lower priority unless not practiced
            priority = 0.3

        # Increase priority if not practiced recently
        if days_since_practice > 7:
            priority += 0.2
        elif days_since_practice > 14:
            priority += 0.4

        return min(1.0, priority)

    def save_twin(self, student_id: str, filepath: str):
        """Save twin state to file."""
        twin = self.twins.get(student_id)
        if twin:
            twin.save_state(filepath)

    def load_twin(self, filepath: str) -> str:
        """
        Load twin from file.

        Returns:
            Student ID
        """
        twin = NeuralDigitalTwin.load_state(filepath)
        self.twins[twin.student_id] = twin
        return twin.student_id
