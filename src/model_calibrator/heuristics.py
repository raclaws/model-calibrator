"""
Heuristic inference for tier assignment.

Queries aggregators first (OpenRouter, BenchLM), then uses signals
(params, price, provider, family) to infer capability tier.
Uses sigmoid curves to avoid cliff effects at thresholds.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import httpx

from model_calibrator.schema import (
    AggregatorData,
    BehavioralFlags,
    Capabilities,
    CapabilityManifest,
    ContextCapability,
    EditFormat,
    EditFormatCapability,
    Heuristics,
    HeuristicSignals,
    InstructionAdherenceCapability,
    Metadata,
    MultiTurnCapability,
    ProbeResult,
    StructuredOutputCapability,
    ToolCallingCapability,
    ToolCallingFormat,
)


def sigmoid(x: float) -> float:
    """Sigmoid function for smooth tier transitions."""
    return 1 / (1 + math.exp(-x))


class HeuristicInference:
    """
    Heuristic-based tier inference from model metadata.
    
    Queries aggregators first, then uses available signals to estimate
    capability tier without running calibration probes.
    """
    
    # Provider reputation scores for tool calling reliability
    PROVIDER_SCORES = {
        "anthropic": 0.95,
        "openai": 0.90,
        "google": 0.80,
        "mistral": 0.65,
        "meta": 0.45,
        "qwen": 0.70,
        "deepseek": 0.75,
        "ollama": 0.40,  # Unknown, depends on model
        "vllm": 0.40,
        "local": 0.30,
    }
    
    # Model families known for coding
    CODE_FAMILIES = {
        "qwen-coder", "qwen2.5-coder", "deepseek-coder", "codellama",
        "starcoder", "starcoder2", "wizardcoder", "phind", "magicoder",
    }
    
    @classmethod
    def query_aggregators(cls, model_id: str) -> dict[str, Any] | None:
        """
        Query aggregators (OpenRouter, BenchLM) for model metadata.
        
        Returns combined metadata dict or None if model unknown.
        """
        metadata: dict[str, Any] = {}
        
        # Try OpenRouter first (400+ models, good coverage)
        openrouter_data = cls._query_openrouter(model_id)
        if openrouter_data:
            metadata.update(openrouter_data)
            metadata["_aggregator_source"] = "openrouter"
        
        # Could add more aggregators here:
        # - BenchLM API
        # - Artificial Analysis API
        
        return metadata if metadata else None
    
    @classmethod
    def _query_openrouter(cls, model_id: str) -> dict[str, Any] | None:
        """Query OpenRouter API for model metadata."""
        try:
            # Normalize model_id to OpenRouter format
            # e.g., "ollama/llama3.2:8b" -> search for "llama-3.2-8b"
            search_term = cls._normalize_for_search(model_id)
            
            with httpx.Client(timeout=10) as client:
                resp = client.get("https://openrouter.ai/api/v1/models")
                if resp.status_code != 200:
                    return None
                
                models = resp.json().get("data", [])
                
                # Find best match
                for model in models:
                    model_name = model.get("id", "").lower()
                    if search_term in model_name or model_name in search_term:
                        return cls._extract_openrouter_metadata(model)
                
                return None
        except Exception:
            return None
    
    @classmethod
    def _normalize_for_search(cls, model_id: str) -> str:
        """Normalize model ID for search matching."""
        # Remove provider prefix
        if "/" in model_id:
            model_id = model_id.split("/", 1)[1]
        
        # Remove quantization suffix
        if "-q" in model_id.lower():
            model_id = model_id.rsplit("-q", 1)[0]
        if ":q" in model_id.lower():
            model_id = model_id.rsplit(":q", 1)[0]
        
        # Normalize separators
        return model_id.lower().replace(":", "-").replace("_", "-")
    
    @classmethod
    def _extract_openrouter_metadata(cls, model: dict) -> dict[str, Any]:
        """Extract relevant metadata from OpenRouter model entry."""
        pricing = model.get("pricing", {})
        
        # Extract output price (per token -> per 1M tokens)
        output_price = 0.0
        if pricing.get("completion"):
            try:
                output_price = float(pricing["completion"]) * 1_000_000
            except (ValueError, TypeError):
                pass
        
        # Extract input price
        input_price = 0.0
        if pricing.get("prompt"):
            try:
                input_price = float(pricing["prompt"]) * 1_000_000
            except (ValueError, TypeError):
                pass
        
        return {
            "openrouter_id": model.get("id"),
            "output_price_per_m": output_price,
            "input_price_per_m": input_price,
            "context_length": model.get("context_length", 0),
            "params_b": cls._estimate_params(model.get("id", "")),
            "provider": cls._extract_provider(model.get("id", "")),
            "model_family": cls._extract_family(model.get("id", "")),
            "supports_vision": "vision" in model.get("id", "").lower(),
            "supports_tools": model.get("tool_calling", False),
        }
    
    @classmethod
    def _estimate_params(cls, model_id: str) -> float | None:
        """Estimate parameter count from model name."""
        model_id = model_id.lower()
        
        # Common patterns: "70b", "8b", "7b", "32b"
        import re
        match = re.search(r"(\d+(?:\.\d+)?)\s*b(?:illion)?", model_id)
        if match:
            return float(match.group(1))
        
        # Check for specific model sizes
        size_hints = {
            "small": 7, "medium": 13, "large": 70,
            "mini": 3, "tiny": 1, "nano": 0.5,
        }
        for hint, size in size_hints.items():
            if hint in model_id:
                return size
        
        return None
    
    @classmethod
    def _extract_provider(cls, model_id: str) -> str:
        """Extract provider from model ID."""
        if "/" in model_id:
            provider = model_id.split("/")[0].lower()
            # Map common provider names
            provider_map = {
                "meta-llama": "meta",
                "mistralai": "mistral",
                "google": "google",
                "anthropic": "anthropic",
                "openai": "openai",
                "qwen": "qwen",
                "deepseek": "deepseek",
            }
            return provider_map.get(provider, provider)
        return "unknown"
    
    @classmethod
    def _extract_family(cls, model_id: str) -> str:
        """Extract model family from ID."""
        model_id = model_id.lower()
        
        families = [
            "llama", "qwen", "mistral", "gemma", "phi",
            "deepseek", "yi", "internlm", "claude", "gpt",
            "codellama", "starcoder", "wizardcoder",
        ]
        
        for family in families:
            if family in model_id:
                # Check for coder variant
                if "coder" in model_id:
                    return f"{family}-coder"
                return family
        
        return "unknown"
    
    @classmethod
    def infer(cls, metadata: dict[str, Any]) -> tuple[int, float]:
        """
        Infer capability tier from metadata.
        
        Returns (tier, confidence) where:
        - tier: 1 (basic), 2 (balanced), 3 (maximum)
        - confidence: 0.0-1.0 indicating certainty
        
        Uses sigmoid curves for smooth transitions.
        """
        score = 0.0
        weights = 0.0
        
        # Parameter count (strongest signal) — sigmoid centered at 40B
        if params := metadata.get("params_b"):
            weights += 1.5
            # Sigmoid: <10B -> ~0, 40B -> 0.5, >70B -> ~1
            param_score = sigmoid((params - 40) * 0.08)
            score += 3 * param_score * 1.5
        
        # Output price per 1M tokens — sigmoid centered at $5
        if price := metadata.get("output_price_per_m"):
            weights += 1.0
            # Sigmoid: <$1 -> ~0, $5 -> 0.5, >$10 -> ~1
            price_score = sigmoid((price - 5) * 0.4)
            score += 3 * price_score
        
        # BFCL score (direct measurement, high weight)
        if bfcl := metadata.get("bfcl_score"):
            weights += 2.0
            # Normalize to 0-1 (assuming 0-100 scale)
            score += 3 * (bfcl / 100) * 2.0
        
        # IFEval score
        if ifeval := metadata.get("ifeval_score"):
            weights += 1.0
            score += 3 * (ifeval / 100)
        
        # Provider reputation
        if provider := metadata.get("provider"):
            weights += 0.5
            provider_score = cls.PROVIDER_SCORES.get(provider.lower(), 0.3)
            score += 3 * provider_score * 0.5
        
        # Model family boost for code specialists
        if family := metadata.get("model_family"):
            if family.lower() in cls.CODE_FAMILIES or "coder" in family.lower():
                weights += 0.3
                score += 3 * 0.8 * 0.3  # Boost for code specialists
        
        # Tool calling support
        if metadata.get("supports_tools"):
            weights += 0.3
            score += 3 * 0.9 * 0.3
        
        # Calculate final tier and confidence
        if weights == 0:
            return (2, 0.2)  # Unknown model, default to T2 with low confidence
        
        avg_score = score / weights
        
        # Map to tier: <1.5 -> T1, 1.5-2.2 -> T2, >2.2 -> T3
        if avg_score >= 2.2:
            tier = 3
        elif avg_score >= 1.3:
            tier = 2
        else:
            tier = 1
        
        # Confidence based on number of signals
        confidence = min(0.95, 0.25 + (weights * 0.12))
        
        return (tier, round(confidence, 3))
    
    @classmethod
    def to_manifest(
        cls,
        model_id: str,
        metadata: dict[str, Any],
        tier: int,
        confidence: float | None = None,
    ) -> CapabilityManifest:
        """
        Create a heuristic-only manifest (no calibration probes).
        
        Used when confidence is high enough to skip calibration.
        """
        if confidence is None:
            tier, confidence = cls.infer(metadata)
        
        # Build heuristic signals
        signals = HeuristicSignals(
            params_b=metadata.get("params_b"),
            output_price_per_m=metadata.get("output_price_per_m"),
            provider=metadata.get("provider"),
            model_family=metadata.get("model_family"),
            release_date=metadata.get("release_date"),
        )
        
        # Build aggregator data if present
        aggregator_data = None
        if metadata.get("_aggregator_source"):
            aggregator_data = AggregatorData(
                source=metadata["_aggregator_source"],
                bfcl_score=metadata.get("bfcl_score"),
                ifeval_score=metadata.get("ifeval_score"),
                ruler_score=metadata.get("ruler_score"),
                retrieved_at=datetime.utcnow(),
            )
        
        # Default edit format based on tier
        edit_format_map = {1: EditFormat.WHOLE, 2: EditFormat.DIFF, 3: EditFormat.DIFF}
        recommended_format = edit_format_map[tier]
        
        # Tool calling based on metadata
        tool_supported = metadata.get("supports_tools", False)
        tool_tier = 3 if tool_supported and tier >= 2 else (2 if tool_supported else 1)
        tool_format = ToolCallingFormat.OPENAI_FUNCTIONS if tool_supported else ToolCallingFormat.NONE
        
        # Context from metadata
        context_length = metadata.get("context_length", 4096)
        # Estimate effective retrieval as 70% of advertised for T3, 50% for T2, 30% for T1
        retrieval_factor = {1: 0.3, 2: 0.5, 3: 0.7}[tier]
        effective_retrieval = int(context_length * retrieval_factor)
        
        return CapabilityManifest(
            model_id=model_id,
            calibrated_at=datetime.utcnow(),
            heuristics=Heuristics(
                overall_tier=tier,
                confidence=confidence,
                aggregator_data=aggregator_data,
                signals=signals,
            ),
            capabilities=Capabilities(
                edit_format=EditFormatCapability(
                    tier=tier,
                    recommended=recommended_format,
                    formats={},  # No probe results for heuristic-only
                ),
                tool_calling=ToolCallingCapability(
                    tier=tool_tier,
                    supported=tool_supported,
                    native_format=tool_format,
                ),
                structured_output=StructuredOutputCapability(
                    tier=min(tier, 2),  # Conservative estimate
                    json_mode=tier >= 2,
                ),
                context=ContextCapability(
                    tier=tier,
                    advertised=context_length,
                    effective_retrieval=effective_retrieval,
                ),
                instruction_adherence=InstructionAdherenceCapability(tier=tier),
                multi_turn=MultiTurnCapability(tier=max(1, tier - 1)),  # Conservative
            ),
            behavioral_flags=BehavioralFlags(
                use_repo_map=tier >= 2,
                supports_vision=metadata.get("supports_vision", False),
            ),
            metadata=Metadata(
                cost_per_1m_input=metadata.get("input_price_per_m", 0),
                cost_per_1m_output=metadata.get("output_price_per_m", 0),
            ),
        )
