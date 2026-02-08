"""
TextBrain variant with STDP learning instead of dopamine-modulated learning.
"""
from __future__ import annotations
import numpy as np
from ..brain import TextBrain
from .stdp import STDPRule, TripletSTDP, create_stdp_rule


class STDPBrain(TextBrain):
    """
    TextBrain with STDP learning rule.

    Replaces dopamine-modulated Hebbian learning with biologically
    plausible spike-timing dependent plasticity.
    """

    def __init__(
        self,
        genome: str,
        vocab_size: int,
        stdp_type: str = "standard",
        seed_override: int | None = None,
        **stdp_kwargs
    ):
        """
        Args:
            genome: Genome string
            vocab_size: Size of vocabulary
            stdp_type: Type of STDP ('standard', 'triplet', 'multiplicative')
            seed_override: Random seed override
            **stdp_kwargs: Additional STDP parameters
        """
        super().__init__(genome, vocab_size, seed_override)

        # Create STDP rule
        self.stdp_rule = create_stdp_rule(stdp_type, **stdp_kwargs)

        # Track spike times for STDP
        self.last_spike_time = np.full(self.N, -np.inf, dtype=np.float32)
        self.current_time = 0.0

        # For triplet STDP
        if isinstance(self.stdp_rule, TripletSTDP):
            self.stdp_rule.initialize_traces(len(self.src))

    def learn(self, target_id: int, probs: np.ndarray) -> float:
        """
        Learning step with STDP rule.

        Args:
            target_id: Target token ID
            probs: Predicted probabilities

        Returns:
            Loss value
        """
        # Compute loss (same as before)
        loss = -np.log(np.clip(probs[target_id], 1e-8, 1.0))
        loss = float(loss)

        # Update loss EMA
        self.loss_ema = 0.99 * self.loss_ema + 0.01 * loss

        # === Output layer learning (supervised) ===
        lr_out = float(self.p["lr"]) * self.lr_out_mul

        # Gradient for output layer
        grad = probs.copy()
        grad[target_id] -= 1.0

        # State for gradient computation
        state = self.compute_state()

        # Update output weights
        R_grad = np.outer(state, grad).astype(np.float32)
        self.R -= lr_out * np.clip(R_grad, -self.max_R_update, self.max_R_update)

        b_grad = grad.astype(np.float32)
        self.b -= lr_out * np.clip(b_grad, -self.max_b_update, self.max_b_update)

        # === Recurrent learning with STDP ===
        self._stdp_recurrent_update()

        # Weight consolidation
        self._consolidate_weights()

        # Homeostasis
        self._update_homeostasis()

        # Structural plasticity
        if self.step > 0 and self.step % int(self.p["prune_every"]) == 0:
            self._prune_and_rewire()

        # Increment step
        self.step += 1
        self.current_time += 1.0

        return loss

    def _stdp_recurrent_update(self):
        """Update recurrent weights using STDP."""
        # Update spike times for active neurons
        active_neurons = self.a > 0
        self.last_spike_time[active_neurons] = self.current_time

        # Compute spike time differences for each synapse
        dt = np.zeros(len(self.src), dtype=np.float32)

        for i, (src_idx, dst_idx) in enumerate(zip(self.src, self.dst)):
            t_pre = self.last_spike_time[src_idx]
            t_post = self.last_spike_time[dst_idx]

            # Only compute if both have spiked
            if t_pre > -np.inf and t_post > -np.inf:
                dt[i] = t_post - t_pre

        # Compute weight changes using STDP rule
        if isinstance(self.stdp_rule, TripletSTDP):
            # For triplet STDP, need to track traces
            pre_spikes = self.a[self.src]
            post_spikes = self.a[self.dst]

            self.stdp_rule.update_traces(pre_spikes, post_spikes, dt=1.0)
            dw = self.stdp_rule.compute_weight_change(pre_spikes, post_spikes)
        else:
            # Standard STDP based on spike timing
            current_w = self.w_slow + self.w_fast
            dw = self.stdp_rule.compute_weight_change(dt, current_w)

        # Scale learning rate
        lr_rec = float(self.p["lr"]) * self.lr_rec_mul

        # Apply to fast weights
        self.w_fast += lr_rec * dw

        # Enforce E/I signs
        self._enforce_ei_signs()

    def _consolidate_weights(self):
        """Transfer fast weights to slow weights."""
        cons_eps = float(self.p["cons_eps"])
        self.w_slow += cons_eps * self.w_fast
        self.w_fast *= float(self.p["w_fast_decay"])

        # Enforce E/I signs after consolidation
        self._enforce_ei_signs()

    def _update_homeostasis(self):
        """Update homeostatic thresholds."""
        homeo = float(self.p["homeo"])
        current_rate = float(np.mean(self.a))
        self.theta += homeo * (current_rate - self.target_rate)

    def reset_spike_times(self):
        """Reset spike time tracking (useful between sequences)."""
        self.last_spike_time.fill(-np.inf)
        self.current_time = 0.0

        if isinstance(self.stdp_rule, TripletSTDP):
            self.stdp_rule.initialize_traces(len(self.src))


class ComparisonBrain:
    """
    Helper class for comparing STDP vs dopamine learning.
    """

    @staticmethod
    def train_comparison(
        genome: str,
        text: str,
        stoi: dict,
        steps: int = 1000,
    ) -> dict:
        """
        Train both STDP and dopamine brains for comparison.

        Args:
            genome: Genome string
            text: Training text
            stoi: Character to index mapping
            steps: Training steps

        Returns:
            Dictionary with results
        """
        from ..tasks.text_task import train_loop_with_history

        vocab_size = len(stoi)

        # Create both brains
        dopamine_brain = TextBrain(genome, vocab_size)
        stdp_brain = STDPBrain(genome, vocab_size, stdp_type="standard")
        triplet_brain = STDPBrain(genome, vocab_size, stdp_type="triplet")

        # Train
        print("Training dopamine brain...")
        dopamine_losses = train_loop_with_history(dopamine_brain, text, stoi, steps, verbose=False)

        print("Training STDP brain...")
        stdp_losses = train_loop_with_history(stdp_brain, text, stoi, steps, verbose=False)

        print("Training triplet STDP brain...")
        triplet_losses = train_loop_with_history(triplet_brain, text, stoi, steps, verbose=False)

        return {
            "dopamine": {
                "losses": dopamine_losses,
                "final_loss": dopamine_losses[-1],
                "brain": dopamine_brain,
            },
            "stdp": {
                "losses": stdp_losses,
                "final_loss": stdp_losses[-1],
                "brain": stdp_brain,
            },
            "triplet_stdp": {
                "losses": triplet_losses,
                "final_loss": triplet_losses[-1],
                "brain": triplet_brain,
            },
        }
