"""Core spiking neural network for next-character prediction.

Implements a biologically-inspired SNN with the following key features:

- **Binary spikes** with sparse top-k winner-take-all activation
- **Dual weight system**: slow (long-term consolidation) and fast (adaptive plasticity)
- **Dual trace system**: fast and slow eligibility traces for temporal credit assignment
- **Axonal delays** (1-5 timesteps) modeled via delay buffers
- **Dopamine-modulated Hebbian learning**: dW = lr * dopamine * advantage * pre * post
- **Homeostatic threshold adaptation** to maintain target firing rate
- **Structural plasticity**: periodic pruning of weak synapses and rewiring
- **Excitatory/inhibitory balance**: weight sign constraints enforced after every update

The network architecture is fully determined by a compact genome string
(see :mod:`magicbrain.genome`).  The forward pass propagates binary spikes
through a spatially-embedded graph with distance-dependent delays.  Learning
combines gradient descent on the readout layer with reward-modulated Hebbian
updates on recurrent connections.

References:
    - Maass, W. (1997). Networks of spiking neurons: the third generation
      of neural network models. Neural Networks, 10(9), 1659-1671.
    - Izhikevich, E.M. (2007). Solving the distal reward problem through
      linkage of STDP and dopamine signaling. Cerebral Cortex, 17(10).
"""

from __future__ import annotations
import numpy as np
from .genome import decode_genome
from .graph import build_graph
from .utils import softmax, sparsify_topm, sigmoid, clamp

class TextBrain:
    """Spiking neural network for character-level language modeling.

    A sparse, biologically-plausible SNN where neurons communicate via binary
    spikes propagated through a 3D spatial graph with axonal delays.  The
    network learns via dopamine-modulated Hebbian plasticity on recurrent
    connections and gradient descent on a linear readout layer.

    The architecture (neuron count, connectivity, learning rates, etc.) is
    entirely specified by a base-4 genome string decoded via
    :func:`~magicbrain.genome.decode_genome`.

    Attributes:
        N: Number of neurons.
        K: Local connectivity (edges per neuron).
        vocab_size: Output vocabulary size.
        a: Binary spike vector (N,) -- current activation.
        w_slow: Long-term synaptic weights (E,) consolidated from w_fast.
        w_fast: Short-term adaptive weights (E,) updated by Hebbian rule.
        theta: Homeostatic firing thresholds (N,).
        trace_fast: Fast eligibility trace (N,), decays with trace_fast_decay.
        trace_slow: Slow eligibility trace (N,), decays with trace_slow_decay.
        R: Readout weight matrix (N, vocab_size).
        b: Readout bias vector (vocab_size,).
        dopamine: Current dopamine level in [0, 1], modulates learning.
        step: Training step counter.
    """

    def __init__(self, genome: str, vocab_size: int, seed_override: int | None = None,
                 use_act: bool = False):
        """Initialize the SNN from a genome string.

        Args:
            genome: Base-4 encoded architecture string (min 24 chars).
            vocab_size: Number of output tokens (characters).
            seed_override: If set, overrides the genome-encoded random seed.
            use_act: If True, use ACT-compensated arithmetic from Balansis
                for numerically stable weight updates and softmax.
        """
        self.genome_str = genome
        self.p = decode_genome(genome)

        self._act = None
        if use_act:
            from .integration.act_backend import ACTBackend
            self._act = ACTBackend()
        self.N = int(self.p["N"])
        self.K = int(self.p["K"])
        self.vocab_size = int(vocab_size)
        
        seed = int(self.p["seed"]) if seed_override is None else seed_override
        self.rng = np.random.default_rng(seed)
        
        self.pos, self.src, self.dst, self.delay, self.idx_by_delay = build_graph(
            self.N, self.K, float(self.p["p_long"]), seed
        )

        self.is_inhib = (self.rng.random(self.N) < float(self.p["p_inhib"])).astype(np.bool_)

        self.w_slow = self.rng.normal(0, 0.03, size=self.src.shape[0]).astype(np.float32)
        self.w_fast = np.zeros_like(self.w_slow)

        self._enforce_ei_signs()

        self.a = np.zeros(self.N, dtype=np.float32)
        self.theta = np.zeros(self.N, dtype=np.float32)
        self.trace_fast = np.zeros(self.N, dtype=np.float32)
        self.trace_slow = np.zeros(self.N, dtype=np.float32)
        self.buffers = [np.zeros(self.N, dtype=np.float32) for _ in range(6)]

        self.R = self.rng.normal(0, 0.12, size=(self.N, vocab_size)).astype(np.float32)
        self.b = np.zeros(vocab_size, dtype=np.float32)

        self.loss_ema = float(np.log(max(2, vocab_size)))

        self.recur_scale = float(1.0 / np.sqrt(float(self.K)))
        self.target_rate = float(self.p["k_active"]) / float(self.N)

        self.sens_fanout = int(max(10, min(16, int(self.p["k_active"]) // 4)))
        perm = self.rng.permutation(self.N).astype(np.int32)
        needed = vocab_size * self.sens_fanout
        sens_flat = perm[:needed] if needed <= self.N else np.resize(perm, needed)
        self.sens_idx = sens_flat.reshape(vocab_size, self.sens_fanout)

        self.noise_std = 0.01
        self.inp_gain = 1.0

        self.alpha = float(self.p["alpha"])
        self.beta = float(self.p["beta"])

        self.fast_clip = 1.5
        self.slow_clip = 2.0

        self.m_fast = min(self.N, 4 * int(self.p["k_active"]))
        self.m_slow = min(self.N, 8 * int(self.p["k_active"]))

        self.lr_out_mul = 0.6
        self.lr_rec_mul = 1.0
        self.max_R_update = 0.02
        self.max_b_update = 0.02

        self.state_sum_target = float(self.p["k_active"])

        self.dopamine = 0.0
        self.ach = 0.0
        self.serotonin = 0.0

        self.step = 0

    def _enforce_ei_signs(self) -> None:
        """Enforce excitatory/inhibitory sign constraints on synaptic weights.

        Inhibitory neurons (flagged by ``is_inhib``) must have non-positive
        outgoing weights.  This is applied to both w_slow and w_fast after
        every weight update to maintain Dale's principle.
        """
        inhib_src = self.is_inhib[self.src]
        if np.any(inhib_src):
            self.w_slow[inhib_src] = -np.abs(self.w_slow[inhib_src])
            self.w_fast[inhib_src] = -np.abs(self.w_fast[inhib_src])

    def _effective_w(self) -> np.ndarray:
        """Return the effective synaptic weight vector: w_slow + w_fast."""
        return (self.w_slow + self.w_fast).astype(np.float32)

    def reset_state(self) -> None:
        """Reset all dynamic state (spikes, traces, delay buffers) to zero."""
        self.a.fill(0)
        self.trace_fast.fill(0)
        self.trace_slow.fill(0)
        for b in self.buffers:
            b.fill(0)

    def compute_state(self) -> np.ndarray:
        """Compute the composite neural state vector for readout.

        Combines the current binary spike vector with fast and slow
        eligibility traces::

            state = a + alpha * clip(trace_fast) + beta * clip(trace_slow)

        The result is L1-normalized to ``k_active`` to maintain a stable
        input magnitude to the readout layer regardless of activation
        sparsity.

        Returns:
            State vector of shape (N,), non-negative, L1-sum ~ k_active.
        """
        tf = np.clip(self.trace_fast, 0.0, self.fast_clip)
        ts = np.clip(self.trace_slow, 0.0, self.slow_clip)

        tf = sparsify_topm(tf, self.m_fast)
        ts = sparsify_topm(ts, self.m_slow)

        state = (self.a + self.alpha * tf + self.beta * ts).astype(np.float32)

        s = float(np.sum(state))
        if s > 1e-6:
            state *= (self.state_sum_target / s)
        return state

    def _update_modulators(self, loss: float) -> float:
        """Update neuromodulatory signals based on prediction error.

        Computes the advantage signal (loss_ema - loss) and maps it through
        a sigmoid to produce a dopamine level in [0, 1].  Positive advantage
        (loss decreased) yields high dopamine (reward), strengthening active
        synapses.  The loss EMA is updated with a slow exponential average
        (tau = 0.005).

        Args:
            loss: Current cross-entropy loss.

        Returns:
            The raw advantage signal (loss_ema - loss), used for scaling
            the Hebbian weight update.
        """
        adv = float(self.loss_ema - loss)
        self.loss_ema = 0.995 * self.loss_ema + 0.005 * loss

        gain = float(self.p["dopamine_gain"])
        bias = float(self.p["dopamine_bias"])
        self.dopamine = sigmoid(gain * adv + bias)

        return adv

    def forward(self, token_id: int) -> np.ndarray:
        """Perform one forward pass: input token -> output probability distribution.

        The forward pass proceeds in five stages:

        1. **Delay buffer shift**: Advance all axonal delay buffers by one
           timestep and apply buffer decay.
        2. **Input injection**: Set sensory neurons for ``token_id`` to 1.
           Select remaining active neurons via top-k on the net input
           (delayed signal minus threshold plus noise).
        3. **Trace update**: Update fast and slow eligibility traces with
           exponential decay plus current spikes.
        4. **Spike propagation**: For each delay group d in [1,5], scatter
           spike contributions (a[src] * w_eff * recur_scale) into
           buffers[d] at destination neurons.
        5. **Readout**: Compute state vector, project through linear readout
           (R, b), apply softmax.

        Args:
            token_id: Input token index in [0, vocab_size).

        Returns:
            Probability distribution over vocabulary, shape (vocab_size,).
        """
        self.step += 1

        delayed_now = self.buffers[1].copy()
        for d in range(1, 5):
            self.buffers[d] = self.buffers[d + 1]
        self.buffers[5].fill(0)

        bd = float(self.p["buf_decay"])
        for d in range(1, 6):
            self.buffers[d] *= bd

        x = delayed_now - self.theta
        if self.noise_std > 0:
            x += self.rng.normal(0, self.noise_std, size=self.N).astype(np.float32)

        a = np.zeros(self.N, dtype=np.float32)
        sidx = self.sens_idx[token_id]
        a[sidx] = 1.0

        remaining = int(self.p["k_active"] - self.sens_fanout)
        if remaining > 0:
            x2 = x.copy()
            x2[sidx] = -1e9
            x[sidx] += self.inp_gain

            idx = np.argpartition(x2, -remaining)[-remaining:]
            pos_idx = idx[x2[idx] > 0]
            if pos_idx.size >= remaining:
                idx = pos_idx[:remaining]
            a[idx] = 1.0

        self.a = a

        fd = float(self.p["trace_fast_decay"])
        sd = float(self.p["trace_slow_decay"])

        self.trace_fast = fd * self.trace_fast + self.a
        np.clip(self.trace_fast, 0.0, self.fast_clip, out=self.trace_fast)

        self.trace_slow = sd * self.trace_slow + 0.15 * self.trace_fast
        np.clip(self.trace_slow, 0.0, self.slow_clip, out=self.trace_slow)

        rs = self.recur_scale
        w_eff = self._effective_w()
        for d in range(1, 6):
            idxe = self.idx_by_delay[d]
            if idxe.size == 0:
                continue
            contrib = (self.a[self.src[idxe]] * w_eff[idxe] * rs).astype(np.float32)
            np.add.at(self.buffers[d], self.dst[idxe], contrib)
            np.clip(self.buffers[d], -5.0, 5.0, out=self.buffers[d])

        state = self.compute_state()
        logits = (state @ self.R + self.b).astype(np.float32)
        if self._act is not None and self._act.available:
            return self._act.softmax(logits)
        return softmax(logits)

    def _consolidate(self) -> None:
        """Consolidate fast weights into slow weights (memory consolidation).

        Implements a slow exponential moving average:
        ``w_slow <- (1 - eps) * w_slow + eps * w_fast``,
        analogous to synaptic tagging and capture in neuroscience.
        """
        eps = float(self.p["cons_eps"])
        if eps <= 0:
            return
        self.w_slow = ((1.0 - eps) * self.w_slow + eps * self.w_fast).astype(np.float32)

    def _decay_fast(self) -> None:
        """Exponentially decay fast weights toward zero."""
        d = float(self.p["w_fast_decay"])
        self.w_fast *= d

    def _prune_and_rewire(self) -> None:
        """Structural plasticity: prune weak synapses and optionally rewire.

        Removes the weakest ``prune_frac`` fraction of edges by absolute
        effective weight.  A ``rewire_frac`` subset of pruned edges is
        reassigned to random source/destination pairs with distance-based
        delays and small random initial weights.  Remaining pruned edges
        are zeroed out.  Delay index is rebuilt after rewiring.
        """
        prune_frac = float(self.p["prune_frac"])
        if prune_frac <= 0:
            return

        E = int(self.src.shape[0])
        n_prune = int(E * prune_frac)
        if n_prune <= 0:
            return

        w_eff = self._effective_w()
        absw = np.abs(w_eff)
        prune_idx = np.argpartition(absw, n_prune)[:n_prune]

        n_rewire = int(n_prune * float(self.p["rewire_frac"]))
        if n_rewire > 0:
            rewire_sel = prune_idx[:n_rewire]
            self.src[rewire_sel] = self.rng.integers(0, self.N, size=n_rewire, dtype=np.int32)
            self.dst[rewire_sel] = self.rng.integers(0, self.N, size=n_rewire, dtype=np.int32)
            dist = np.linalg.norm(self.pos[self.src[rewire_sel]] - self.pos[self.dst[rewire_sel]], axis=1).astype(np.float32)
            self.delay[rewire_sel] = np.clip((dist * 6).astype(np.int32) + 1, 1, 5)

            self.w_slow[rewire_sel] = self.rng.normal(0, 0.01, size=n_rewire).astype(np.float32)
            self.w_fast[rewire_sel] = 0.0

        zero_sel = prune_idx[n_rewire:]
        if zero_sel.size > 0:
            self.w_slow[zero_sel] = 0.0
            self.w_fast[zero_sel] = 0.0

        self._enforce_ei_signs()

        idx_by_delay = [np.array([], dtype=np.int32)]
        for d in range(1, 6):
            idx_by_delay.append(np.where(self.delay == d)[0].astype(np.int32))
        self.idx_by_delay = idx_by_delay

    def damage_edges(self, frac: float = 0.2) -> None:
        """Zero out a random fraction of synaptic weights (lesion experiment).

        Args:
            frac: Fraction of edges to damage, in [0, 1].
        """
        frac = clamp(float(frac), 0.0, 1.0)
        E = int(self.src.shape[0])
        n = int(E * frac)
        if n <= 0:
            return
        idx = self.rng.choice(E, size=n, replace=False)
        self.w_slow[idx] = 0.0
        self.w_fast[idx] = 0.0

    def learn(self, target_id: int, probs: np.ndarray) -> float:
        """Perform one learning step given the target token and predicted probabilities.

        The learning algorithm combines two mechanisms:

        1. **Readout gradient descent**: Standard cross-entropy gradient on the
           linear readout layer (R, b), with per-element clipping.

        2. **Dopamine-modulated Hebbian update** on recurrent weights::

               dW = lr * dopamine * advantage * pre * post

           where ``pre`` = presynaptic trace, ``post`` = postsynaptic spike,
           ``dopamine`` = sigmoid(gain * advantage + bias), and
           ``advantage`` = loss_ema - loss.

        After weight updates: E/I sign constraints are enforced, fast weights
        are decayed and consolidated into slow weights, homeostatic thresholds
        are adapted, and periodic pruning/rewiring is triggered.

        Args:
            target_id: Correct next-token index.
            probs: Predicted probability distribution from :meth:`forward`.

        Returns:
            Cross-entropy loss: -log(probs[target_id]).
        """
        p = float(probs[target_id])
        loss = float(-np.log(p + 1e-9))

        adv = self._update_modulators(loss)

        lr = float(self.p["lr"])
        lr_out = lr * self.lr_out_mul
        lr_rec = lr * self.lr_rec_mul

        grad = probs.copy()
        grad[target_id] -= 1.0

        state = self.compute_state()

        if self._act is not None and self._act.available:
            dR = (lr_out * self._act.outer_product(state, grad)).astype(np.float32)
        else:
            dR = (lr_out * (state[:, None] * grad[None, :])).astype(np.float32)
        np.clip(dR, -self.max_R_update, self.max_R_update, out=dR)
        self.R -= dR

        db = (lr_out * grad).astype(np.float32)
        np.clip(db, -self.max_b_update, self.max_b_update, out=db)
        self.b -= db

        np.clip(self.R, -1.0, 1.0, out=self.R)

        pre = self.trace_fast[self.src]
        post = self.a[self.dst]

        dW = (lr_rec * self.dopamine * adv * pre * post).astype(np.float32)
        np.clip(dW, -0.02, 0.02, out=dW)
        if self._act is not None and self._act.available:
            self.w_fast = self._act.weight_update(self.w_fast, dW, 1.0)
        else:
            self.w_fast += dW

        self._enforce_ei_signs()

        np.clip(self.w_fast, -0.5, 0.5, out=self.w_fast)
        np.clip(self.w_slow, -0.5, 0.5, out=self.w_slow)

        self._decay_fast()
        self._consolidate()

        self.theta += (float(self.p["homeo"]) * (self.a - self.target_rate)).astype(np.float32)

        prune_every = int(self.p["prune_every"])
        if prune_every > 0 and (self.step % prune_every) == 0:
            self._prune_and_rewire()

        return loss

    def avg_theta(self) -> float:
        """Return the mean homeostatic threshold across all neurons."""
        return float(np.mean(self.theta))

    def mean_abs_w(self) -> float:
        """Return the mean absolute effective synaptic weight."""
        return float(np.mean(np.abs(self._effective_w())))

    def firing_rate(self) -> float:
        """Return the current fraction of active (spiking) neurons."""
        return float(np.mean(self.a))
