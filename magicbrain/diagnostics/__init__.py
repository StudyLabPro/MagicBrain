"""Diagnostics and monitoring system for MagicBrain."""
from .live_monitor import LiveMonitor, TrainingMetrics, add_diagnostics_methods_to_brain
from .neuronal_dynamics import SpikeRaster, ActivityTracker
from .synaptic_metrics import SynapticAnalyzer, ConnectivityAnalyzer
from .plasticity_tracker import PlasticityTracker, StructuralMonitor
from .act_metrics import ACTMetricsTracker, ACTSnapshot

__all__ = [
    "LiveMonitor",
    "TrainingMetrics",
    "SpikeRaster",
    "ActivityTracker",
    "SynapticAnalyzer",
    "ConnectivityAnalyzer",
    "PlasticityTracker",
    "StructuralMonitor",
    "ACTMetricsTracker",
    "ACTSnapshot",
]
