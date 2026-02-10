"""NeuroGenesis Engine — neurogenomic memory system.

Data is not stored directly. A generative program (genome) is stored
that can reproduce the data through dynamic neural structure activation.

Pipeline: Dataset -> Compile -> Genome -> Develop -> Train -> Reconstruct
"""

from .compiler import GenomeCompiler
from .energy import EnergyFunction
from .attractor_dynamics import AttractorDynamics
from .cppn import CPPN
from .development import DevelopmentOperator
from .pattern_memory import PatternMemory
from .reconstruction import ReconstructionOperator
from .genome_v2 import GenomeV2, decode_genome_v2

__all__ = [
    "GenomeCompiler",
    "EnergyFunction",
    "AttractorDynamics",
    "CPPN",
    "DevelopmentOperator",
    "PatternMemory",
    "ReconstructionOperator",
    "GenomeV2",
    "decode_genome_v2",
]
