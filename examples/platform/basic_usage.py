"""
Basic Platform Usage Example

Demonstrates core platform features:
- Creating and registering models
- Simple orchestration
- Execution strategies
"""
from magicbrain.platform import (
    ModelRegistry,
    ModelOrchestrator,
    ExecutionStrategy,
)
from magicbrain.models.snn import SNNTextModel


def main():
    print("=" * 60)
    print("MagicBrain Platform - Basic Usage Example")
    print("=" * 60)

    # Step 1: Create models
    print("\n1. Creating SNN models...")

    model1 = SNNTextModel(
        genome="30121033102301230112332100123",
        vocab_size=50,
        model_id="snn_model_1",
        version="1.0.0",
        description="First SNN model"
    )

    model2 = SNNTextModel(
        genome="30121033102301230112332100123",
        vocab_size=50,
        model_id="snn_model_2",
        version="1.0.0",
        description="Second SNN model"
    )

    print(f"  ✓ Created {model1.metadata.model_id}")
    print(f"  ✓ Created {model2.metadata.model_id}")

    # Step 2: Register models
    print("\n2. Registering models in registry...")

    registry = ModelRegistry()
    registry.register(model1, tags=["snn", "text"])
    registry.register(model2, tags=["snn", "text"])

    print(f"  ✓ Registry contains {len(registry.list_models())} models")

    # Step 3: Create orchestrator
    print("\n3. Setting up orchestrator...")

    orch = ModelOrchestrator(registry=registry)
    orch.add_model(model1)
    orch.add_model(model2)
    orch.connect("snn_model_1", "snn_model_2")

    print(f"  ✓ Orchestrator configured")
    print(f"  ✓ Graph: {list(orch.get_graph().keys())}")

    # Step 4: Execute sequential
    print("\n4. Executing sequential pipeline...")

    input_token = 0  # Start token
    result = orch.execute(
        input_data=input_token,
        strategy=ExecutionStrategy.SEQUENTIAL,
        entry_model="snn_model_1"
    )

    print(f"  ✓ Execution successful: {result.success}")
    print(f"  ✓ Execution time: {result.execution_time_ms:.2f}ms")
    print(f"  ✓ Models executed: {result.models_executed}")
    print(f"  ✓ Final output shape: {result.get_final_output().shape}")

    # Step 5: Model statistics
    print("\n5. Model statistics...")

    for model_id in result.models_executed:
        model = orch.get_model(model_id)
        if hasattr(model, 'get_brain_stats'):
            stats = model.get_brain_stats()
            print(f"\n  {model_id}:")
            print(f"    - Neurons: {stats['N']}")
            print(f"    - Firing rate: {stats['firing_rate']:.4f}")
            print(f"    - Dopamine: {stats['dopamine']:.4f}")

    # Step 6: Search and list
    print("\n6. Registry operations...")

    all_models = registry.list_models()
    print(f"  ✓ Total models: {len(all_models)}")

    snn_models = registry.list_models(tags=["snn"])
    print(f"  ✓ SNN models: {len(snn_models)}")

    search_results = registry.search("snn")
    print(f"  ✓ Search 'snn': {len(search_results)} results")

    print("\n" + "=" * 60)
    print("Example completed successfully! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
