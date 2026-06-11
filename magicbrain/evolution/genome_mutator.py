"""
Genome mutation operations for evolutionary search.
"""
from __future__ import annotations
from typing import List
import numpy as np


class GenomeMutator:
    """Performs controlled mutations on genome strings."""

    VALID_CHARS = "0123"

    @staticmethod
    def point_mutation(genome: str, num_mutations: int = 1, rng=None) -> str:
        """
        Apply point mutations (change random characters).

        Args:
            genome: Input genome string
            num_mutations: Number of characters to mutate
            rng: Random number generator

        Returns:
            Mutated genome string
        """
        if rng is None:
            rng = np.random.default_rng()

        genome_list = list(genome)
        positions = rng.choice(len(genome_list), size=min(num_mutations, len(genome_list)), replace=False)

        for pos in positions:
            # Choose different character
            current = genome_list[pos]
            choices = [c for c in GenomeMutator.VALID_CHARS if c != current]
            genome_list[pos] = rng.choice(choices)

        return ''.join(genome_list)

    @staticmethod
    def insertion_mutation(genome: str, num_insertions: int = 1, rng=None) -> str:
        """
        Insert random characters at random positions.

        Args:
            genome: Input genome string
            num_insertions: Number of characters to insert
            rng: Random number generator

        Returns:
            Mutated genome string
        """
        if rng is None:
            rng = np.random.default_rng()

        genome_list = list(genome)

        for _ in range(num_insertions):
            pos = rng.integers(0, len(genome_list) + 1)
            char = rng.choice(list(GenomeMutator.VALID_CHARS))
            genome_list.insert(pos, char)

        return ''.join(genome_list)

    @staticmethod
    def deletion_mutation(genome: str, num_deletions: int = 1, rng=None) -> str:
        """
        Delete random characters.

        Args:
            genome: Input genome string
            num_deletions: Number of characters to delete
            rng: Random number generator

        Returns:
            Mutated genome string
        """
        if rng is None:
            rng = np.random.default_rng()

        if len(genome) <= num_deletions:
            return genome  # Don't delete everything

        genome_list = list(genome)
        positions = rng.choice(len(genome_list), size=min(num_deletions, len(genome_list) - 1), replace=False)
        positions = sorted(positions, reverse=True)

        for pos in positions:
            del genome_list[pos]

        return ''.join(genome_list)

    @staticmethod
    def adaptive_mutation(genome: str, mutation_rate: float = 0.05, rng=None) -> str:
        """
        Apply adaptive mutation with probability-based changes.

        Args:
            genome: Input genome string
            mutation_rate: Probability of mutating each character
            rng: Random number generator

        Returns:
            Mutated genome string
        """
        if rng is None:
            rng = np.random.default_rng()

        genome_list = list(genome)

        for i in range(len(genome_list)):
            if rng.random() < mutation_rate:
                current = genome_list[i]
                choices = [c for c in GenomeMutator.VALID_CHARS if c != current]
                genome_list[i] = rng.choice(choices)

        return ''.join(genome_list)

    @staticmethod
    def crossover(genome1: str, genome2: str, rng=None) -> str:
        """
        Single-point crossover between two genomes.

        Args:
            genome1: First parent genome
            genome2: Second parent genome
            rng: Random number generator

        Returns:
            Offspring genome
        """
        if rng is None:
            rng = np.random.default_rng()

        # Ensure same length for crossover
        min_len = min(len(genome1), len(genome2))
        max_len = max(len(genome1), len(genome2))

        # Crossover point
        point = rng.integers(1, min_len)

        # Create offspring
        if len(genome1) > len(genome2):
            offspring = genome1[:point] + genome2[point:] + genome1[min_len:]
        else:
            offspring = genome1[:point] + genome2[point:]

        return offspring

    @staticmethod
    def uniform_crossover(genome1: str, genome2: str, mix_rate: float = 0.5, rng=None) -> str:
        """
        Uniform crossover - each position independently inherited.

        Args:
            genome1: First parent genome
            genome2: Second parent genome
            mix_rate: Probability of taking from genome1 (vs genome2)
            rng: Random number generator

        Returns:
            Offspring genome
        """
        if rng is None:
            rng = np.random.default_rng()

        min_len = min(len(genome1), len(genome2))
        offspring = []

        for i in range(min_len):
            if rng.random() < mix_rate:
                offspring.append(genome1[i])
            else:
                offspring.append(genome2[i])

        # Add remainder from longer parent
        if len(genome1) > len(genome2):
            offspring.extend(genome1[min_len:])
        elif len(genome2) > len(genome1):
            offspring.extend(genome2[min_len:])

        return ''.join(offspring)

    @staticmethod
    def generate_random_genome(length: int = 25, rng=None) -> str:
        """
        Generate completely random genome.

        Args:
            length: Genome length
            rng: Random number generator

        Returns:
            Random genome string
        """
        if rng is None:
            rng = np.random.default_rng()

        chars = rng.choice(list(GenomeMutator.VALID_CHARS), size=length)
        return ''.join(chars)
