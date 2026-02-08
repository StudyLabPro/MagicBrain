"""
MagicBrain Quickstart Examples
Demonstrates new features from v0.2.0
"""
from magicbrain import TextBrain
from magicbrain.tasks.text_task import build_vocab, train_loop
from magicbrain.diagnostics import LiveMonitor, SynapticAnalyzer
from magicbrain.evolution import SimpleGA, GenomeMutator
from magicbrain.io import save_model


# Example 1: Basic Training with Monitoring
def example_monitored_training():
    """Train a brain with live monitoring."""
    print("=" * 60)
    print("Example 1: Training with Live Monitoring")
    print("=" * 60)

    # Sample text
    text = "hello world " * 100

    # Create brain
    genome = "30121033102301230112332100123"
    stoi, itos = build_vocab(text)
    brain = TextBrain(genome, len(stoi))

    # Create monitor
    monitor = LiveMonitor(log_every=50)
    analyzer = SynapticAnalyzer()

    # Training loop
    import numpy as np
    ids = np.array([stoi[c] for c in text], dtype=np.int32)
    n = len(ids) - 1

    print("Training for 500 steps with monitoring...")
    for step in range(500):
        idx = step % n
        x, y = int(ids[idx]), int(ids[idx + 1])

        probs = brain.forward(x)
        loss = brain.learn(y, probs)

        # Record metrics
        if monitor.should_log(step):
            monitor.record(brain, loss, step)
            monitor.print_status(step, loss, brain)

        # Analyze weights periodically
        if step % 200 == 0:
            analyzer.record(brain)

    # Get summary
    summary = monitor.get_summary()
    print("\nTraining Summary:")
    print(f"  Final loss: {summary['recent_avg_loss']:.4f}")
    print(f"  Final dopamine: {summary['recent_avg_dopamine']:.3f}")

    # Weight evolution
    if len(analyzer.weight_history) > 1:
        print("\nWeight Evolution:")
        ei_ratios = analyzer.get_weight_evolution('ei_ratio')
        print(f"  E/I ratio: {ei_ratios[0]:.2f} → {ei_ratios[-1]:.2f}")


# Example 2: Genome Evolution
def example_genome_evolution():
    """Evolve genomes using genetic algorithm."""
    print("\n" + "=" * 60)
    print("Example 2: Genome Evolution")
    print("=" * 60)

    text = "abcdefg " * 50

    # Create GA
    ga = SimpleGA(
        population_size=8,
        elite_size=2,
        tournament_size=3,
        mutation_rate=0.15,
        seed=42
    )

    print("Initializing population...")
    ga.initialize_population("30121033102301230112332100123")

    print("Evolving for 3 generations...")
    best = ga.run_evolution(
        text=text,
        num_generations=3,
        fitness_fn="loss",
        steps_per_eval=50,
        verbose=True
    )

    print(f"\nBest genome: {best.genome}")
    print(f"Fitness: {best.fitness:.4f}")

    # Hall of fame
    print("\nTop 3 genomes:")
    for i, ind in enumerate(ga.get_hall_of_fame(3)):
        print(f"  {i+1}. Fitness={ind.fitness:.4f} {ind.genome}")


# Example 3: Genome Mutations
def example_mutations():
    """Demonstrate genome mutation operations."""
    print("\n" + "=" * 60)
    print("Example 3: Genome Mutations")
    print("=" * 60)

    import numpy as np
    rng = np.random.default_rng(42)

    genome = "30121033102301230112332100123"
    print(f"Original: {genome}")

    # Point mutation
    mutated = GenomeMutator.point_mutation(genome, num_mutations=2, rng=rng)
    print(f"Point mutation (2 points): {mutated}")

    # Adaptive mutation
    mutated = GenomeMutator.adaptive_mutation(genome, mutation_rate=0.1, rng=rng)
    print(f"Adaptive mutation (rate=0.1): {mutated}")

    # Crossover
    genome2 = "11111111111111111111111111111"
    offspring = GenomeMutator.crossover(genome, genome2, rng=rng)
    print(f"Crossover offspring: {offspring}")

    # Random genome
    random_genome = GenomeMutator.generate_random_genome(length=25, rng=rng)
    print(f"Random genome: {random_genome}")


# Example 4: Backend Selection
def example_backend_selection():
    """Demonstrate backend selection."""
    print("\n" + "=" * 60)
    print("Example 4: Backend Selection")
    print("=" * 60)

    from magicbrain.backends import get_backend, auto_select_backend

    # Get specific backend
    numpy_backend = get_backend("numpy")
    print(f"NumPy backend: {numpy_backend.name}")

    # Auto-select best available
    auto_backend = auto_select_backend()
    print(f"Auto-selected: {auto_backend.name}")

    # Try JAX if available
    try:
        jax_backend = get_backend("jax")
        print(f"JAX backend: {jax_backend.name}")
        print(f"  Has GPU: {jax_backend.has_gpu()}")
    except ImportError:
        print("JAX backend not available (install with: pip install jax)")


if __name__ == "__main__":
    # Run all examples
    example_monitored_training()
    example_genome_evolution()
    example_mutations()
    example_backend_selection()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
