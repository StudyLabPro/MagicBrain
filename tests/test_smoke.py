import os
import numpy as np
import pytest
from magicbrain.brain import TextBrain
from magicbrain.genome import decode_genome
from magicbrain.io import save_model, load_model
from magicbrain.tasks.text_task import build_vocab

def test_genome_decode():
    g = "30121033102301230112332100123"
    p = decode_genome(g)
    assert p["N"] >= 256
    assert p["K"] >= 8
    assert p["lr"] > 0

def test_brain_init():
    g = "30121033102301230112332100123"
    brain = TextBrain(g, vocab_size=10)
    assert brain.N == int(brain.p["N"])
    assert brain.a.shape == (brain.N,)
    assert brain.w_slow.shape[0] == brain.N * brain.K + int(brain.N * brain.K * brain.p["p_long"])

def test_brain_learn_step():
    g = "30121033102301230112332100123"
    brain = TextBrain(g, vocab_size=5)
    
    # Forward
    probs = brain.forward(0)
    assert probs.shape == (5,)
    assert np.isclose(np.sum(probs), 1.0)
    
    # Learn
    loss = brain.learn(1, probs)
    assert loss > 0

def test_save_load(tmp_path):
    g = "30121033102301230112332100123"
    vocab = {"a": 0, "b": 1}
    itos = {0: "a", 1: "b"}
    
    brain = TextBrain(g, vocab_size=2)
    brain.step = 123
    
    # Change some weights
    brain.w_slow[0] = 0.5
    brain.R[0, 0] = 0.5
    
    path = tmp_path / "model.npz"
    save_model(brain, vocab, itos, str(path))
    
    loaded_brain, l_stoi, l_itos, _ = load_model(str(path))
    
    assert loaded_brain.step == 123
    assert loaded_brain.genome_str == g
    assert np.allclose(loaded_brain.w_slow, brain.w_slow)
    assert np.allclose(loaded_brain.R, brain.R)
    assert l_stoi == vocab

def test_ei_sign_invariant():
    g = "30121033102301230112332100123"
    brain = TextBrain(g, vocab_size=5)
    
    # Check if inhibitory weights are non-positive
    # self.src maps edge to source neuron
    # self.is_inhib[src] tells if source is inhibitory
    
    inhib_mask = brain.is_inhib[brain.src]
    
    # w_slow and w_fast should be <= 0 for inhib sources
    assert np.all(brain.w_slow[inhib_mask] <= 1e-9)
    assert np.all(brain.w_fast[inhib_mask] <= 1e-9)
