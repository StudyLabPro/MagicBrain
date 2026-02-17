import argparse
import sys
import os
from .brain import TextBrain
from .io import save_model, load_model
from .tasks.text_task import build_vocab, train_loop
from .tasks.self_repair import benchmark_self_repair
from .sampling import sample
from .diagnostics import LiveMonitor
from .evolution import SimpleGA, GenomeMutator

DEFAULT_TEXT = """
From fairest creatures we desire increase,
That thereby beauty's rose might never die,
But as the riper should by time decease,
His tender heir might bear his memory:
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.
""" * 50

def main():
    parser = argparse.ArgumentParser(description="MagicBrain CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # TRAIN
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--genome", type=str, default="30121033102301230112332100123")
    train_parser.add_argument("--text", type=str, help="Path to text file. If not provided, uses default Shakespeare snippet.")
    train_parser.add_argument("--steps", type=int, default=10000)
    train_parser.add_argument("--out", type=str, default="model.npz", help="Output model path")
    train_parser.add_argument("--load", type=str, help="Resume from existing model path")

    # SAMPLE
    sample_parser = subparsers.add_parser("sample", help="Sample text from model")
    sample_parser.add_argument("--model", type=str, required=True)
    sample_parser.add_argument("--seed", type=str, default="To be")
    sample_parser.add_argument("--n", type=int, default=500)
    sample_parser.add_argument("--temp", type=float, default=0.75)

    # REPAIR
    repair_parser = subparsers.add_parser("repair", help="Run self-repair benchmark")
    repair_parser.add_argument("--genome", type=str, default="30121033102301230112332100123")
    repair_parser.add_argument("--text", type=str, help="Path to text file")
    repair_parser.add_argument("--damage", type=float, default=0.2)

    # EVOLVE
    evolve_parser = subparsers.add_parser("evolve", help="Evolve genomes using genetic algorithm")
    evolve_parser.add_argument("--genome", type=str, default="30121033102301230112332100123",
                              help="Initial genome (optional)")
    evolve_parser.add_argument("--text", type=str, help="Path to text file for training")
    evolve_parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    evolve_parser.add_argument("--population", type=int, default=20, help="Population size")
    evolve_parser.add_argument("--steps", type=int, default=100, help="Training steps per evaluation")
    evolve_parser.add_argument("--fitness", type=str, default="loss",
                              choices=["loss", "convergence", "stability"],
                              help="Fitness function to use")
    evolve_parser.add_argument("--out", type=str, default="best_genome.txt",
                              help="Output file for best genome")

    # MONITOR
    monitor_parser = subparsers.add_parser("monitor", help="Train with live monitoring")
    monitor_parser.add_argument("--genome", type=str, default="30121033102301230112332100123")
    monitor_parser.add_argument("--text", type=str, help="Path to text file")
    monitor_parser.add_argument("--steps", type=int, default=10000)
    monitor_parser.add_argument("--out", type=str, default="model.npz")
    monitor_parser.add_argument("--metrics", type=str, default="metrics.json",
                               help="Output file for metrics")

    # TRAIN-DISTRIBUTED
    dist_parser = subparsers.add_parser("train-distributed", help="Distributed training with FedAvg")
    dist_parser.add_argument("--genome", type=str, default="30121033102301230112332100123")
    dist_parser.add_argument("--data", type=str, help="Path to text file")
    dist_parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    dist_parser.add_argument("--sync-every", type=int, default=100, help="Steps between sync rounds")
    dist_parser.add_argument("--steps", type=int, default=10000, help="Total training steps per worker")
    dist_parser.add_argument("--partition", type=str, default="overlapping",
                             choices=["overlapping", "round_robin", "shuffle"],
                             help="Data partitioning strategy")
    dist_parser.add_argument("--checkpoint-dir", type=str, default=None,
                             help="Directory for training checkpoints")
    dist_parser.add_argument("--out", type=str, default="model.npz", help="Output model path")

    # NEUROGENESIS
    ng_parser = subparsers.add_parser("neurogenesis", help="NeuroGenesis Engine: dataset-driven genome compilation")
    ng_sub = ng_parser.add_subparsers(dest="ng_command", required=True)

    # neurogenesis compile
    ng_compile = ng_sub.add_parser("compile", help="Compile dataset into genome")
    ng_compile.add_argument("--text", type=str, help="Path to text file")
    ng_compile.add_argument("--strategy", type=str, default="statistical",
                            choices=["hash", "statistical", "hybrid"],
                            help="Compilation strategy")
    ng_compile.add_argument("--length", type=int, default=28, help="Genome length")

    # neurogenesis run
    ng_run = ng_sub.add_parser("run", help="Full pipeline: compile -> develop -> train -> reconstruct")
    ng_run.add_argument("--text", type=str, help="Path to text file")
    ng_run.add_argument("--strategy", type=str, default="statistical",
                        choices=["hash", "statistical", "hybrid"])
    ng_run.add_argument("--steps", type=int, default=10000)
    ng_run.add_argument("--cppn", action="store_true", help="Use CPPN for weight generation")
    ng_run.add_argument("--out", type=str, default="model.npz", help="Output model path")

    # neurogenesis benchmark
    ng_bench = ng_sub.add_parser("benchmark", help="Benchmark strategies against baselines")
    ng_bench.add_argument("--text", type=str, help="Path to text file")
    ng_bench.add_argument("--steps", type=int, default=5000)
    ng_bench.add_argument("--trials", type=int, default=3)

    # neurogenesis attractors
    ng_attr = ng_sub.add_parser("attractors", help="Discover attractors in a trained model")
    ng_attr.add_argument("--model", type=str, required=True, help="Path to trained model")
    ng_attr.add_argument("--probes", type=int, default=500, help="Number of random probes")

    args = parser.parse_args()

    if args.command == "train":
        text = DEFAULT_TEXT
        if args.text:
            with open(args.text, "r", encoding="utf-8") as f:
                text = f.read()
        
        # Normalize
        text = " ".join(text.split())

        if args.load:
            print(f"Loading model from {args.load}...")
            brain, stoi, itos, _ = load_model(args.load)
        else:
            print(f"Creating new brain with genome: {args.genome}")
            stoi, itos = build_vocab(text)
            brain = TextBrain(args.genome, len(stoi))

        print("Starting training...")
        train_loop(brain, text, stoi, steps=args.steps)
        
        print(f"Saving model to {args.out}...")
        save_model(brain, stoi, itos, args.out)

    elif args.command == "sample":
        print(f"Loading model from {args.model}...")
        brain, stoi, itos, _ = load_model(args.model)
        
        print(f"Sampling with seed='{args.seed}'...")
        out = sample(brain, stoi, itos, args.seed, n=args.n, temperature=args.temp)
        print("-" * 60)
        print(out)
        print("-" * 60)

    elif args.command == "repair":
        text = DEFAULT_TEXT
        if args.text:
            with open(args.text, "r", encoding="utf-8") as f:
                text = f.read()
        text = " ".join(text.split())

        stoi, itos = build_vocab(text)
        brain = TextBrain(args.genome, len(stoi))

        # Pre-train
        print("Pre-training...")
        train_loop(brain, text, stoi, steps=10000, print_every=2000)

        benchmark_self_repair(brain, text, stoi, itos, damage_frac=args.damage)

    elif args.command == "evolve":
        text = DEFAULT_TEXT
        if args.text:
            with open(args.text, "r", encoding="utf-8") as f:
                text = f.read()
        text = " ".join(text.split())

        print(f"Starting genome evolution with population={args.population}, generations={args.generations}")
        print(f"Fitness function: {args.fitness}")
        print(f"Initial genome: {args.genome}")

        # Create GA
        ga = SimpleGA(
            population_size=args.population,
            elite_size=max(2, args.population // 10),
            tournament_size=3,
            mutation_rate=0.15,
            crossover_rate=0.7,
        )

        # Initialize population
        ga.initialize_population(args.genome)

        # Run evolution
        best = ga.run_evolution(
            text=text,
            num_generations=args.generations,
            fitness_fn=args.fitness,
            steps_per_eval=args.steps,
            verbose=True
        )

        print("\n" + "=" * 60)
        print(f"Best genome found: {best.genome}")
        print(f"Fitness: {best.fitness:.4f}")
        print(f"Generation: {best.generation}")
        print("=" * 60)

        # Save best genome
        with open(args.out, 'w') as f:
            f.write(f"# Best genome from evolution\n")
            f.write(f"# Fitness: {best.fitness:.4f}\n")
            f.write(f"# Generation: {best.generation}\n")
            f.write(f"{best.genome}\n")

        print(f"\nBest genome saved to: {args.out}")

        # Show hall of fame
        print("\nHall of Fame (Top 5):")
        for i, ind in enumerate(ga.get_hall_of_fame(5)):
            print(f"  {i+1}. Fitness={ind.fitness:.4f} Gen={ind.generation} Genome={ind.genome}")

    elif args.command == "monitor":
        text = DEFAULT_TEXT
        if args.text:
            with open(args.text, "r", encoding="utf-8") as f:
                text = f.read()
        text = " ".join(text.split())

        print(f"Creating brain with genome: {args.genome}")
        stoi, itos = build_vocab(text)
        brain = TextBrain(args.genome, len(stoi))

        # Create monitor
        monitor = LiveMonitor(log_every=100)

        print("Starting training with live monitoring...")

        # Training loop with monitoring
        import numpy as np
        ids = np.array([stoi[c] for c in text], dtype=np.int32)
        n = len(ids) - 1

        for step in range(args.steps):
            idx = step % n
            x = int(ids[idx])
            y = int(ids[idx + 1])

            probs = brain.forward(x)
            loss = brain.learn(y, probs)

            # Record metrics
            if monitor.should_log(step):
                metrics = monitor.record(brain, loss, step)
                monitor.print_status(step, loss, brain)

        # Save model and metrics
        print(f"\nSaving model to {args.out}...")
        save_model(brain, stoi, itos, args.out)

        print(f"Saving metrics to {args.metrics}...")
        monitor.save(args.metrics)

        # Print summary
        summary = monitor.get_summary()
        print("\nTraining Summary:")
        print(f"  Total steps: {summary['total_steps']}")
        print(f"  Recent avg loss: {summary['recent_avg_loss']:.4f}")
        print(f"  Recent avg dopamine: {summary['recent_avg_dopamine']:.3f}")
        print(f"  Recent avg firing rate: {summary['recent_avg_firing_rate']:.3f}")

    elif args.command == "train-distributed":
        from .training.coordinator import TrainingCoordinator

        text = DEFAULT_TEXT
        if args.data:
            with open(args.data, "r", encoding="utf-8") as f:
                text = f.read()
        text = " ".join(text.split())

        print(f"Distributed training: genome={args.genome}, workers={args.workers}, "
              f"sync_every={args.sync_every}, steps={args.steps}")
        print(f"Partition mode: {args.partition}")

        coordinator = TrainingCoordinator(
            genome_str=args.genome,
            data=text,
            n_workers=args.workers,
            sync_every=args.sync_every,
            partition_mode=args.partition,
            checkpoint_dir=args.checkpoint_dir,
        )

        def on_round(rnd, loss):
            if (rnd + 1) % 10 == 0 or rnd == 0:
                print(f"  Round {rnd + 1}: avg_loss={loss:.4f}")

        result = coordinator.train(total_steps=args.steps, callback=on_round)

        print(f"\nTraining complete:")
        print(f"  Final loss: {result.final_loss:.4f}")
        print(f"  Total steps: {result.total_steps}")
        print(f"  Wall time: {result.wall_time:.1f}s")
        print(f"  Speedup vs single: {result.speedup_vs_single:.2f}x")
        print(f"  Per-worker losses: {[f'{l:.4f}' for l in result.per_worker_losses]}")

        # Save final model
        if hasattr(coordinator, '_current_weights'):
            brain = TextBrain(args.genome, len(coordinator.vocab))
            weights = coordinator._current_weights
            brain.w_slow = weights["w_slow"]
            brain.w_fast = weights["w_fast"]
            brain.R = weights["R"]
            brain.b = weights["b"]
            brain.theta = weights["theta"]
        else:
            brain = coordinator._ref_brain

        print(f"Saving model to {args.out}...")
        save_model(brain, coordinator.vocab, coordinator.itos, args.out)

    elif args.command == "neurogenesis":
        from .neurogenesis.compiler import GenomeCompiler, analyze_dataset
        from .neurogenesis.reconstruction import ReconstructionOperator
        from .neurogenesis.attractor_dynamics import AttractorDynamics

        if args.ng_command == "compile":
            text = DEFAULT_TEXT
            if args.text:
                with open(args.text, "r", encoding="utf-8") as f:
                    text = f.read()
            text = " ".join(text.split())

            compiler = GenomeCompiler()
            result = compiler.compile_with_metadata(
                text, strategy=args.strategy, genome_length=args.length,
            )

            print(f"Strategy: {result['strategy']}")
            print(f"Data: {result['data_size']} bytes, "
                  f"vocab={result['stats']['vocab_size']}, "
                  f"entropy={result['stats']['entropy']:.2f}")
            print(f"Hash: {result['data_hash'][:16]}...")
            print(f"Genome: {result['genome']}")

        elif args.ng_command == "run":
            from .tasks.neurogenesis_task import run_neurogenesis_pipeline

            text = DEFAULT_TEXT
            if args.text:
                with open(args.text, "r", encoding="utf-8") as f:
                    text = f.read()
            text = " ".join(text.split())

            result = run_neurogenesis_pipeline(
                text,
                strategy=args.strategy,
                steps=args.steps,
                use_cppn=args.cppn,
                verbose=True,
            )

            # Save model
            stoi, itos = build_vocab(text)
            print(f"\nSaving model to {args.out}...")
            save_model(result.brain, stoi, itos, args.out)

            print(f"\nGenome: {result.genome}")
            print(f"Final loss: {result.final_loss:.4f}")

        elif args.ng_command == "benchmark":
            from .tasks.neurogenesis_task import run_benchmark

            text = DEFAULT_TEXT
            if args.text:
                with open(args.text, "r", encoding="utf-8") as f:
                    text = f.read()
            text = " ".join(text.split())

            run_benchmark(
                text,
                steps=args.steps,
                trials=args.trials,
                verbose=True,
            )

        elif args.ng_command == "attractors":
            brain, stoi, itos, meta = load_model(args.model)
            print(f"Model loaded: N={brain.N}, K={brain.K}, step={brain.step}")

            recon = ReconstructionOperator()
            print(f"\nSearching for attractors ({args.probes} probes)...")

            decodings = recon.reconstruct_from_attractors(
                brain, itos, n_probes=args.probes,
            )

            print(f"\nFound {len(decodings)} unique attractors:")
            for dec in decodings[:20]:
                print(f"  #{dec.attractor_index}: token='{dec.token}' "
                      f"conf={dec.confidence:.3f} energy={dec.energy:.4f}")

if __name__ == "__main__":
    main()
