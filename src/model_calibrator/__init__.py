"""
Model Calibrator — Runtime capability calibration for LLMs in agentic coding harnesses.

Library, not CLI. Harnesses embed and consume automatically.
"""

from model_calibrator.schema import (
    CapabilityManifest,
    EditFormatCapability,
    ToolCallingCapability,
    StructuredOutputCapability,
    ContextCapability,
    InstructionAdherenceCapability,
    MultiTurnCapability,
    BehavioralFlags,
    Quirks,
    ProbeResult,
    MeasuredRate,
    PositionBias,
    HeuristicSignals,
    AggregatorData,
)
from model_calibrator.heuristics import HeuristicInference
from model_calibrator.registry import Registry
from model_calibrator.client import CalibrationClient
from model_calibrator.calibrator import Calibrator

__version__ = "0.2.0"
__all__ = [
    # Main classes
    "HeuristicInference",
    "Registry",
    "CalibrationClient",
    # Schema
    "CapabilityManifest",
    "EditFormatCapability",
    "ToolCallingCapability",
    "StructuredOutputCapability",
    "ContextCapability",
    "InstructionAdherenceCapability",
    "MultiTurnCapability",
    "BehavioralFlags",
    "Quirks",
    "ProbeResult",
    "MeasuredRate",
    "PositionBias",
    "HeuristicSignals",
    "AggregatorData",
]


def get_manifest(
    model_id: str,
    base_url: str | None = None,
    api_key: str | None = None,
    min_trust: str = "community",
    calibrate_if_missing: bool = False,
    calibration_tier: str = "quick",
) -> CapabilityManifest | None:
    """
    Convenience function: fetch manifest from registry or calibrate.
    
    Args:
        model_id: Canonical model identifier (e.g., "ollama/llama3.2:8b-q4_K_M")
        base_url: API base URL (required if calibrate_if_missing=True)
        api_key: API key (required if calibrate_if_missing=True)
        min_trust: Minimum trust level ("verified", "community", "unverified")
        calibrate_if_missing: Run calibration if not in registry
        calibration_tier: "quick" ($0.50), "standard" ($2-3), "thorough" ($5-10)
    
    Returns:
        CapabilityManifest or None if not found and calibration disabled
    """
    # Try registry first
    manifest = Registry.get(model_id, min_trust=min_trust)
    if manifest is not None:
        return manifest
    
    # Try heuristics if we have metadata
    metadata = HeuristicInference.query_aggregators(model_id)
    if metadata:
        tier, confidence = HeuristicInference.infer(metadata)
        if confidence >= 0.7:
            # High confidence heuristic — use without calibration
            return HeuristicInference.to_manifest(model_id, metadata, tier, confidence)
    
    # Calibrate if requested
    if calibrate_if_missing and base_url and api_key:
        from model_calibrator.calibrator import Calibrator
        calibrator = Calibrator(base_url, api_key)
        return calibrator.calibrate(model_id, tier=calibration_tier)
    
    return None
