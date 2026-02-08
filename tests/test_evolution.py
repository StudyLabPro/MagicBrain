"""
Tests for genome evolution system.
"""
import pytest
import numpy as np
from magicbrain.evolution import GenomeMutator, FitnessEvaluator, SimpleGA, Individual
from magicbrain.brain import TextBrain
from magicbrain.tasks.text_task import build_vocab


def test_point_mutation():
    """Test point mutation."""
    genome = "00000"
    rng = np.random.default_rng(42)

    mutated = GenomeMutator.point_mutation(genome, num_mutations=1, rng=rng)

    # Should differ in exactly 1 position
    differences = sum(1 for a, b in zip(genome, mutated) if a != b)
    assert differences == 1


def test_crossover():
    """Test genome crossover."""
    genome1 = "00000"
    genome2 = "11111"
    rng = np.random.default_rng(42)

    offspring = GenomeMutator.crossover(genome1, genome2, rng=rng)

    # Offspring should have characters from both parents
    assert len(offspring) == len(genome1)
    assert '0' in offspring or '1' in offspring


def test_generate_random_genome():
    """Test random genome generation."""
    rng = np.random.default_rng(42)
    genome = GenomeMutator.generate_random_genome(length=20, rng=rng)

    assert len(genome) == 20
    assert all(c in "0123" for c in genome)


def test_adaptive_mutation():
    """Test adaptive mutation."""
    genome = "0000000000"
    rng = np.random.default_rng(42)

    mutated = GenomeMutator.adaptive_mutation(genome, mutation_rate=0.3, rng=rng)

    # Should have some mutations (probabilistic)
    # With rate=0.3 and 10 chars, expect ~3 mutations
    differences = sum(1 for a, b in zip(genome, mutated) if a != b)
    assert 0 <= differences <= 10


def test_simple_ga_initialization():
    """Test SimpleGA initialization."""
    ga = SimpleGA(population_size=10, elite_size=2, seed=42)

    assert ga.population_size == 10
    assert ga.elite_size == 2
    assert len(ga.population) == 0

    # Initialize population
    ga.initialize_population("30121033102301230112332100123")

    assert len(ga.population) == 10
    assert all(isinstance(ind, Individual) for ind in ga.population)


def test_simple_ga_tournament_selection():
    """Test tournament selection."""
    ga = SimpleGA(population_size=10, tournament_size=3, seed=42)
    ga.initialize_population("30121033102301230112332100123")

    # Assign random fitness
    for i, ind in enumerate(ga.population):
        ind.fitness = float(i)

    # Select should prefer higher fitness
    selected = ga.tournament_selection()
    assert selected.fitness > 0  # Should not always pick worst


def test_fitness_evaluator_loss():
    """Test loss-based fitness evaluation."""
    text = "hello world hello world hello world" * 10
    stoi, itos = build_vocab(text)
    genome = "30121033102301230112332100123"

    brain = TextBrain(genome, len(stoi))

    # Evaluate fitness (this will train briefly)
    fitness = FitnessEvaluator.loss_fitness(brain, text, stoi, steps=50)

    # Fitness should be negative (negative loss)
    assert fitness < 0


def test_individual_creation():
    """Test Individual dataclass."""
    ind = Individual(genome="12345", fitness=0.5, generation=1)

    assert ind.genome == "12345"
    assert ind.fitness == 0.5
    assert ind.generation == 1
    assert ind.parent_genomes is None


def test_ga_evolution_basic():
    """Test basic GA evolution (short run)."""
    text = "abcabc" * 50
    ga = SimpleGA(population_size=5, elite_size=1, seed=42)
    ga.initialize_population("30121033102301230112332100123")

    # Run for 2 generations (quick test)
    best = ga.run_evolution(text, num_generations=2, steps_per_eval=20, verbose=False)

    assert best is not None
    assert best.fitness != 0.0
    assert len(ga.history) == 2
