from __future__ import annotations
import numpy as np
from .genome import decode_genome
from .graph import build_graph
from .utils import softmax, sparsify_topm, sigmoid, clamp

class TextBrain:
    def __init__(self, genome: str, vocab_size: int, seed_override: int | None = None,
                 use_act: bool = False):
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

    def _enforce_ei_signs(self):
        inhib_src = self.is_inhib[self.src]
        if np.any(inhib_src):
            self.w_slow[inhib_src] = -np.abs(self.w_slow[inhib_src])
            self.w_fast[inhib_src] = -np.abs(self.w_fast[inhib_src])

    def _effective_w(self) -> np.ndarray:
        return (self.w_slow + self.w_fast).astype(np.float32)

    def reset_state(self):
        self.a.fill(0)
        self.trace_fast.fill(0)
        self.trace_slow.fill(0)
        for b in self.buffers:
            b.fill(0)

    def compute_state(self) -> np.ndarray:
        tf = np.clip(self.trace_fast, 0.0, self.fast_clip)
        ts = np.clip(self.trace_slow, 0.0, self.slow_clip)

        tf = sparsify_topm(tf, self.m_fast)
        ts = sparsify_topm(ts, self.m_slow)

        state = (self.a + self.alpha * tf + self.beta * ts).astype(np.float32)

        s = float(np.sum(state))
        if s > 1e-6:
            state *= (self.state_sum_target / s)
        return state

    def _update_modulators(self, loss: float):
        adv = float(self.loss_ema - loss)
        self.loss_ema = 0.995 * self.loss_ema + 0.005 * loss

        gain = float(self.p["dopamine_gain"])
        bias = float(self.p["dopamine_bias"])
        self.dopamine = sigmoid(gain * adv + bias)

        return adv

    def forward(self, token_id: int) -> np.ndarray:
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

    def _consolidate(self):
        eps = float(self.p["cons_eps"])
        if eps <= 0:
            return
        self.w_slow = ((1.0 - eps) * self.w_slow + eps * self.w_fast).astype(np.float32)

    def _decay_fast(self):
        d = float(self.p["w_fast_decay"])
        self.w_fast *= d

    def _prune_and_rewire(self):
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

    def damage_edges(self, frac: float = 0.2):
        frac = clamp(float(frac), 0.0, 1.0)
        E = int(self.src.shape[0])
        n = int(E * frac)
        if n <= 0:
            return
        idx = self.rng.choice(E, size=n, replace=False)
        self.w_slow[idx] = 0.0
        self.w_fast[idx] = 0.0

    def learn(self, target_id: int, probs: np.ndarray) -> float:
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
        return float(np.mean(self.theta))

    def mean_abs_w(self) -> float:
        return float(np.mean(np.abs(self._effective_w())))

    def firing_rate(self) -> float:
        return float(np.mean(self.a))
