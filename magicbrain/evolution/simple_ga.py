"""
Simple Genetic Algorithm for genome evolution.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from ..brain import TextBrain
from ..tasks.text_task import build_vocab
from .genome_mutator import GenomeMutator
from .fitness_functions import FitnessEvaluator


@dataclass
class Individual:
    """Represents one genome in the population."""
    genome: str
    fitness: float = 0.0
    generation: int = 0
    parent_genomes: Optional[List[str]] = None


class SimpleGA:
    """
    Simple Genetic Algorithm for evolving genomes.
    Uses tournament selection, crossover, and mutation.
    """

    def __init__(
        self,
        population_size: int = 20,
        elite_size: int = 2,
        tournament_size: int = 3,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        seed: int = 42,
    ):
        """
        Args:
            population_size: Number of individuals in population
            elite_size: Number of best individuals to preserve
            tournament_size: Size of tournament for selection
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            seed: Random seed
        """
        self.population_size = population_size
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = np.random.default_rng(seed)

        self.population: List[Individual] = []
        self.generation = 0
        self.best_ever: Optional[Individual] = None
        self.history: List[Dict] = []

    def initialize_population(self, initial_genome: str = None):
        """
        Initialize population with random genomes.

        Args:
            initial_genome: Optional starting genome (others are mutations of it)
        """
        self.population = []

        if initial_genome is None:
            initial_genome = GenomeMutator.generate_random_genome(25, self.rng)

        # Add initial genome
        self.population.append(Individual(genome=initial_genome, generation=0))

        # Create variants
        for i in range(1, self.population_size):
            # Mutate initial genome with varying intensity
            num_mutations = self.rng.integers(1, 6)
            mutated = GenomeMutator.point_mutation(initial_genome, num_mutations, self.rng)
            self.population.append(Individual(genome=mutated, generation=0))

    def evaluate_population(self, text: str, vocab_size: int, stoi: Dict,
                           fitness_fn: str = "loss", steps: int = 100):
        """
        Evaluate fitness for all individuals in population.

        Args:
            text: Training text
            vocab_size: Vocabulary size
            stoi: Character to index mapping
            fitness_fn: Fitness function to use
            steps: Training steps for evaluation
        """
        for individual in self.population:
            if individual.fitness > 0:  # Already evaluated
                continue

            # Create brain with this genome
            brain = TextBrain(individual.genome, vocab_size)

            # Evaluate fitness
            if fitness_fn == "loss":
                fitness = FitnessEvaluator.loss_fitness(brain, text, stoi, steps)
            elif fitness_fn == "convergence":
                fitness = FitnessEvaluator.convergence_speed_fitness(brain, text, stoi, steps)
            elif fitness_fn == "stability":
                fitness = FitnessEvaluator.stability_fitness(brain, text, stoi, steps)
            else:
                fitness = FitnessEvaluator.loss_fitness(brain, text, stoi, steps)

            individual.fitness = fitness

        # Update best ever
        current_best = max(self.population, key=lambda x: x.fitness)
        if self.best_ever is None or current_best.fitness > self.best_ever.fitness:
            self.best_ever = Individual(
                genome=current_best.genome,
                fitness=current_best.fitness,
                generation=self.generation
            )

    def tournament_selection(self) -> Individual:
        """Select individual via tournament selection."""
        candidates = self.rng.choice(self.population, size=self.tournament_size, replace=False)
        return max(candidates, key=lambda x: x.fitness)

    def evolve_generation(self) -> List[Individual]:
        """
        Create next generation via selection, crossover, mutation.

        Returns:
            New population
        """
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)

        # Elitism: keep best individuals
        new_population = sorted_pop[:self.elite_size]

        # Create offspring
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            # Crossover
            if self.rng.random() < self.crossover_rate:
                offspring_genome = GenomeMutator.crossover(parent1.genome, parent2.genome, self.rng)
            else:
                offspring_genome = parent1.genome

            # Mutation
            if self.rng.random() < self.mutation_rate:
                mutation_strength = self.rng.integers(1, 4)
                offspring_genome = GenomeMutator.point_mutation(offspring_genome, mutation_strength, self.rng)

            offspring = Individual(
                genome=offspring_genome,
                generation=self.generation + 1,
                parent_genomes=[parent1.genome, parent2.genome]
            )

            new_population.append(offspring)

        return new_population

    def run_evolution(
        self,
        text: str,
        num_generations: int = 10,
        fitness_fn: str = "loss",
        steps_per_eval: int = 100,
        verbose: bool = True
    ) -> Individual:
        """
        Run evolutionary algorithm.

        Args:
            text: Training text
            num_generations: Number of generations to evolve
            fitness_fn: Fitness function name
            steps_per_eval: Training steps per fitness evaluation
            verbose: Print progress

        Returns:
            Best individual found
        """
        # Build vocabulary
        stoi, itos = build_vocab(text)
        vocab_size = len(stoi)

        for gen in range(num_generations):
            self.generation = gen

            # Evaluate fitness
            self.evaluate_population(text, vocab_size, stoi, fitness_fn, steps_per_eval)

            # Get statistics
            fitnesses = [ind.fitness for ind in self.population]
            best_fitness = max(fitnesses)
            avg_fitness = np.mean(fitnesses)
            std_fitness = np.std(fitnesses)

            if verbose:
                print(f"Gen {gen:3d} | Best: {best_fitness:8.4f} | "
                      f"Avg: {avg_fitness:8.4f} ± {std_fitness:6.4f}")

            # Record history
            self.history.append({
                "generation": gen,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "std_fitness": std_fitness,
                "best_genome": max(self.population, key=lambda x: x.fitness).genome,
            })

            # Evolve to next generation
            if gen < num_generations - 1:
                self.population = self.evolve_generation()

        return self.best_ever

    def get_hall_of_fame(self, n: int = 5) -> List[Individual]:
        """
        Get top N best individuals ever seen.

        Args:
            n: Number of individuals to return

        Returns:
            List of best individuals
        """
        all_individuals = []

        for record in self.history:
            all_individuals.append(Individual(
                genome=record["best_genome"],
                fitness=record["best_fitness"],
                generation=record["generation"]
            ))

        # Sort and return top N
        sorted_inds = sorted(all_individuals, key=lambda x: x.fitness, reverse=True)
        return sorted_inds[:n]
