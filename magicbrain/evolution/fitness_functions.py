"""
Fitness functions for genome evaluation.
"""
from __future__ import annotations
from typing import Dict, Callable
import numpy as np
from ..brain import TextBrain


class FitnessEvaluator:
    """Evaluates genome fitness on tasks."""

    @staticmethod
    def loss_fitness(brain: TextBrain, text: str, stoi: Dict, steps: int = 100) -> float:
        """
        Evaluate based on final loss after training.

        Args:
            brain: Brain instance
            text: Training text
            stoi: Character to index mapping
            steps: Number of training steps

        Returns:
            Fitness score (higher is better, -loss)
        """
        from ..tasks.text_task import train_loop

        # Train for specified steps
        final_loss = train_loop(brain, text, stoi, steps=steps, verbose=False)

        # Return negative loss (higher is better)
        return -final_loss

    @staticmethod
    def convergence_speed_fitness(brain: TextBrain, text: str, stoi: Dict, steps: int = 100) -> float:
        """
        Evaluate based on how quickly loss decreases.

        Args:
            brain: Brain instance
            text: Training text
            stoi: Character to index mapping
            steps: Number of training steps

        Returns:
            Fitness score (higher convergence speed is better)
        """
        from ..tasks.text_task import train_loop_with_history

        # Train and get loss history
        loss_history = train_loop_with_history(brain, text, stoi, steps=steps, verbose=False)

        if len(loss_history) < 10:
            return 0.0

        # Measure convergence: difference between early and late loss
        early_loss = np.mean(loss_history[:len(loss_history)//4])
        late_loss = np.mean(loss_history[-len(loss_history)//4:])

        convergence = early_loss - late_loss

        return float(convergence)

    @staticmethod
    def robustness_fitness(brain: TextBrain, text: str, stoi: Dict,
                          damage_fraction: float = 0.2, repair_steps: int = 50) -> float:
        """
        Evaluate robustness to damage.

        Args:
            brain: Brain instance
            text: Training text
            stoi: Character to index mapping
            damage_fraction: Fraction of weights to damage
            repair_steps: Steps for self-repair

        Returns:
            Fitness score (higher recovery is better)
        """
        from ..tasks.text_task import train_loop

        # Train initially
        initial_loss = train_loop(brain, text, stoi, steps=100, verbose=False)

        # Damage the network
        brain.damage_edges(damage_fraction)

        # Measure performance after damage
        damaged_loss = train_loop(brain, text, stoi, steps=1, verbose=False)

        # Allow self-repair
        repaired_loss = train_loop(brain, text, stoi, steps=repair_steps, verbose=False)

        # Recovery score: how much loss was recovered
        damage_amount = damaged_loss - initial_loss
        recovery = damaged_loss - repaired_loss

        if damage_amount < 1e-6:
            return 0.0

        recovery_ratio = recovery / damage_amount

        return float(recovery_ratio)

    @staticmethod
    def stability_fitness(brain: TextBrain, text: str, stoi: Dict, steps: int = 200) -> float:
        """
        Evaluate training stability (low variance in loss).

        Args:
            brain: Brain instance
            text: Training text
            stoi: Character to index mapping
            steps: Number of training steps

        Returns:
            Fitness score (higher stability is better)
        """
        from ..tasks.text_task import train_loop_with_history

        loss_history = train_loop_with_history(brain, text, stoi, steps=steps, verbose=False)

        if len(loss_history) < 10:
            return 0.0

        # Use second half (after initial settling)
        stable_region = loss_history[len(loss_history)//2:]

        # Lower variance is better
        variance = np.var(stable_region)
        stability = 1.0 / (1.0 + variance)

        return float(stability)

    @staticmethod
    def multi_objective_fitness(brain: TextBrain, text: str, stoi: Dict,
                                weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Compute multiple fitness objectives.

        Args:
            brain: Brain instance
            text: Training text
            stoi: Character to index mapping
            weights: Optional weights for objectives

        Returns:
            Dictionary of fitness scores
        """
        if weights is None:
            weights = {
                "loss": 1.0,
                "convergence": 0.5,
                "stability": 0.3,
            }

        scores = {}

        if "loss" in weights:
            scores["loss"] = FitnessEvaluator.loss_fitness(brain, text, stoi, steps=100)

        if "convergence" in weights:
            scores["convergence"] = FitnessEvaluator.convergence_speed_fitness(brain, text, stoi, steps=100)

        if "stability" in weights:
            scores["stability"] = FitnessEvaluator.stability_fitness(brain, text, stoi, steps=200)

        if "robustness" in weights:
            scores["robustness"] = FitnessEvaluator.robustness_fitness(brain, text, stoi)

        # Weighted sum
        total = sum(scores[k] * weights.get(k, 1.0) for k in scores)
        scores["total"] = total

        return scores
