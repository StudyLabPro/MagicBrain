"""Biologically-plausible learning rules."""
from .stdp import STDPRule, TripletSTDP, AdditiveSTDP, MultiplicativeSTDP, create_stdp_rule
from .stdp_brain import STDPBrain, ComparisonBrain

__all__ = [
    "STDPRule",
    "TripletSTDP",
    "AdditiveSTDP",
    "MultiplicativeSTDP",
    "create_stdp_rule",
    "STDPBrain",
    "ComparisonBrain",
]
