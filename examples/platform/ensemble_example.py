"""
Ensemble Example

Demonstrates creating an ensemble of SNN models and aggregating their outputs.
"""
import numpy as np
from magicbrain.platform import (
    ModelOrchestrator,
    ExecutionStrategy,
)
from magicbrain.models.snn import SNNTextModel


def main():
    print("=" * 60)
    print("MagicBrain Platform - Ensemble Example")
    print("=" * 60)

    # Create ensemble of 3 SNN models with different genomes
    print("\n1. Creating ensemble of 3 SNN models...")

    genomes = [
        "30121033102301230112332100123",
        "30121033102301230112332100124",  # Slightly different
        "30121033102301230112332100125",  # Slightly different
    ]

    models = []
    for i, genome in enumerate(genomes):
        model = SNNTextModel(
            genome=genome,
            vocab_size=50,
            model_id=f"snn_ensemble_{i}",
            version="1.0.0",
            description=f"Ensemble member {i}"
        )
        models.append(model)
        print(f"  ✓ Created model {i} with {model.brain.N} neurons")

    # Set up orchestrator for parallel execution
    print("\n2. Setting up parallel orchestrator...")

    orch = ModelOrchestrator()
    for model in models:
        orch.add_model(model)

    print(f"  ✓ Registered {len(models)} models")

    # Execute all models in parallel
    print("\n3. Executing ensemble (parallel)...")

    input_token = 0
    result = orch.execute(
        input_data=input_token,
        strategy=ExecutionStrategy.PARALLEL
    )

    print(f"  ✓ Execution time: {result.execution_time_ms:.2f}ms")
    print(f"  ✓ All models executed: {result.success}")

    # Aggregate outputs (voting/averaging)
    print("\n4. Aggregating ensemble outputs...")

    outputs = [
        result.get_output(f"snn_ensemble_{i}")
        for i in range(len(models))
    ]

    # Softmax each output
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    probs = [softmax(output) for output in outputs]

    # Average probabilities
    ensemble_probs = np.mean(probs, axis=0)

    # Get top predictions
    top_k = 5
    top_indices = np.argsort(ensemble_probs)[-top_k:][::-1]

    print(f"\n  Top {top_k} predictions:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"    {rank}. Token {idx}: {ensemble_probs[idx]:.4f}")

    # Compare with individual model predictions
    print("\n5. Individual model top predictions:")

    for i, prob in enumerate(probs):
        top_idx = np.argmax(prob)
        print(f"  Model {i}: Token {top_idx} ({prob[top_idx]:.4f})")

    print("\n6. Ensemble diversity metrics...")

    # Calculate agreement between models
    top_predictions = [np.argmax(p) for p in probs]
    agreement = len(set(top_predictions)) / len(models)

    print(f"  Diversity score: {agreement:.2f}")
    print(f"  (1.0 = all different, 0.33 = all same)")

    print("\n" + "=" * 60)
    print("Ensemble example completed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
