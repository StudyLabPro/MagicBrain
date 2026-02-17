"""NeuroGenesis Engine — neurogenomic memory system.

Data is not stored directly. A generative program (genome) is stored
that can reproduce the data through dynamic neural structure activation.

Pipeline: Dataset -> Compile -> Genome -> Develop -> Train -> Reconstruct
"""

from .compiler import GenomeCompiler, CompilationMetrics
from .energy import EnergyFunction
from .attractor_dynamics import AttractorDynamics, AttractorMetrics
from .cppn import CPPN
from .development import DevelopmentOperator, DevelopmentMetrics
from .pattern_memory import PatternMemory, PatternQualityMetrics
from .reconstruction import ReconstructionOperator
from .genome_v2 import GenomeV2, decode_genome_v2
from .pipeline import NeurogenesisPipeline, PipelineConfig, PipelineResult

__all__ = [
    "GenomeCompiler",
    "CompilationMetrics",
    "EnergyFunction",
    "AttractorDynamics",
    "AttractorMetrics",
    "CPPN",
    "DevelopmentOperator",
    "DevelopmentMetrics",
    "PatternMemory",
    "PatternQualityMetrics",
    "ReconstructionOperator",
    "GenomeV2",
    "decode_genome_v2",
    "NeurogenesisPipeline",
    "PipelineConfig",
    "PipelineResult",
]
