"""
Pydantic schema for Model Capability Manifest v2.

Implements RFC v0.2.0 — per-capability tiers, structured quirks,
confidence intervals, Aider behavioral flags.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field


# === Enums ===

class EditFormat(str, Enum):
    """Supported edit formats for coding agents."""
    WHOLE = "whole"
    DIFF = "diff"
    UDIFF = "udiff"
    SEARCH_REPLACE = "search_replace"
    ARCHITECT = "architect"


class ToolCallingFormat(str, Enum):
    """Native tool calling formats by provider."""
    OPENAI_FUNCTIONS = "openai_functions"
    ANTHROPIC_TOOL_USE = "anthropic_tool_use"
    RAW_JSON = "raw_json"
    NONE = "none"


class PositionBiasType(str, Enum):
    """Types of context position bias."""
    NONE = "none"
    PRIMACY = "primacy"
    RECENCY = "recency"
    MIDDLE = "middle"


class QuirkTag(str, Enum):
    """Structured quirk tags for programmatic handling."""
    TRUNCATION_RISK = "truncation_risk"
    FORMAT_DRIFT = "format_drift"
    REFACTOR_EAGER = "refactor_eager"
    CONSERVATIVE_EDITS = "conservative_edits"
    UDIFF_LINE_ERRORS = "udiff_line_errors"
    CONTEXT_CLIFF = "context_cliff"
    TOOL_HALLUCINATION = "tool_hallucination"
    VERBOSE_OUTPUT = "verbose_output"
    TERSE_OUTPUT = "terse_output"
    MARKDOWN_HEAVY = "markdown_heavy"
    CODE_FENCE_ISSUES = "code_fence_issues"
    STREAMING_PARTIAL_JSON = "streaming_partial_json"


# === Statistics Helpers ===

def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Calculate Wilson score confidence interval for binomial proportion.
    More accurate than normal approximation for small samples.
    """
    if total == 0:
        return (0.0, 1.0)
    
    # Z-score for confidence level (1.96 for 95%)
    z = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
    
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    
    return (round(lower, 4), round(upper, 4))


# === Base Types ===

class ProbeResult(BaseModel):
    """Result of a calibration probe with statistical rigor."""
    success_rate: float = Field(ge=0, le=1, description="Success rate (0-1)")
    confidence_interval: tuple[float, float] = Field(
        description="95% Wilson score confidence interval [lower, upper]"
    )
    samples: int = Field(ge=1, description="Number of samples (minimum 20 for valid results)")
    failure_modes: list[str] = Field(default_factory=list, description="Observed failure types")
    
    @classmethod
    def from_trials(cls, successes: int, total: int, failure_modes: list[str] | None = None) -> ProbeResult:
        """Create ProbeResult from trial counts."""
        return cls(
            success_rate=round(successes / total, 4) if total > 0 else 0.0,
            confidence_interval=wilson_score_interval(successes, total),
            samples=total,
            failure_modes=failure_modes or [],
        )


class MeasuredRate(BaseModel):
    """A measured rate with confidence interval."""
    value: float = Field(ge=0, le=1, description="Measured rate (0-1)")
    confidence_interval: tuple[float, float] = Field(
        description="95% confidence interval [lower, upper]"
    )
    samples: int = Field(ge=1, description="Number of samples")
    
    @classmethod
    def from_trials(cls, successes: int, total: int) -> MeasuredRate:
        """Create MeasuredRate from trial counts."""
        return cls(
            value=round(successes / total, 4) if total > 0 else 0.0,
            confidence_interval=wilson_score_interval(successes, total),
            samples=total,
        )


class PositionBias(BaseModel):
    """Context position bias measurement."""
    type: PositionBiasType = Field(default=PositionBiasType.NONE)
    strength: float = Field(ge=0, le=1, default=0.0, description="Bias strength (0=none, 1=severe)")
    measured_at_tokens: int = Field(default=0, description="Context size where bias was measured")


# === Capability Sections ===

class EditFormatCapability(BaseModel):
    """Edit format compliance capabilities."""
    tier: Literal[1, 2, 3] = Field(description="Capability tier for edit formats")
    recommended: EditFormat = Field(description="Recommended format for this model")
    formats: dict[str, ProbeResult] = Field(
        default_factory=dict,
        description="Per-format probe results (whole, diff, udiff, etc.)"
    )


class ToolCallingCapability(BaseModel):
    """Tool/function calling capabilities."""
    tier: Literal[1, 2, 3] = Field(default=1, description="Capability tier for tool calling")
    supported: bool = Field(default=False, description="Whether model supports tool calling")
    native_format: ToolCallingFormat = Field(
        default=ToolCallingFormat.NONE,
        description="Provider's native tool calling format"
    )
    parallel_calls: bool = Field(
        default=False,
        description="Whether model reliably handles multiple tool calls in single response"
    )
    schema_complexity_limit: int = Field(
        default=0,
        ge=0,
        description="Max tools in schema before degradation"
    )
    hallucination_rate: MeasuredRate | None = Field(
        default=None,
        description="Rate of phantom/hallucinated tool calls"
    )
    required_param_compliance: MeasuredRate | None = Field(
        default=None,
        description="Rate of correctly providing required parameters"
    )


class StructuredOutputCapability(BaseModel):
    """Structured output (JSON/XML) capabilities."""
    tier: Literal[1, 2, 3] = Field(default=1, description="Capability tier for structured output")
    json_mode: bool = Field(default=False, description="Supports JSON mode")
    strict_schema: bool = Field(default=False, description="Supports strict schema enforcement")
    schema_adherence: MeasuredRate | None = Field(
        default=None,
        description="Rate of schema-compliant outputs"
    )


class ContextCapability(BaseModel):
    """Context window capabilities."""
    tier: Literal[1, 2, 3] = Field(default=1, description="Capability tier for context handling")
    advertised: int = Field(default=0, description="Advertised context window in tokens")
    effective_retrieval: int = Field(
        default=0,
        description="Reliable retrieval depth (needle-in-haystack)"
    )
    position_bias: PositionBias = Field(
        default_factory=PositionBias,
        description="Position bias characteristics"
    )


class InstructionAdherenceCapability(BaseModel):
    """Instruction following capabilities."""
    tier: Literal[1, 2, 3] = Field(default=1, description="Capability tier for instruction adherence")
    format_compliance: MeasuredRate | None = Field(
        default=None,
        description="Rate of following format instructions"
    )
    refusal_rate: MeasuredRate | None = Field(
        default=None,
        description="Rate of refusing valid requests"
    )
    system_prompt_following: MeasuredRate | None = Field(
        default=None,
        description="Rate of following system prompt instructions"
    )


class MultiTurnCapability(BaseModel):
    """Multi-turn conversation capabilities — critical for agentic use."""
    tier: Literal[1, 2, 3] = Field(default=1, description="Capability tier for multi-turn")
    instruction_retention: MeasuredRate | None = Field(
        default=None,
        description="How well model follows instructions over 10+ turns"
    )
    context_pollution_resistance: MeasuredRate | None = Field(
        default=None,
        description="Resistance to being derailed by irrelevant context"
    )
    measured_turns: int = Field(default=0, description="Number of turns in test conversation")


class Capabilities(BaseModel):
    """All capability sections."""
    edit_format: EditFormatCapability
    tool_calling: ToolCallingCapability = Field(default_factory=ToolCallingCapability)
    structured_output: StructuredOutputCapability = Field(default_factory=StructuredOutputCapability)
    context: ContextCapability = Field(default_factory=ContextCapability)
    instruction_adherence: InstructionAdherenceCapability = Field(
        default_factory=InstructionAdherenceCapability
    )
    multi_turn: MultiTurnCapability = Field(default_factory=MultiTurnCapability)


# === Aider Behavioral Flags ===

class BehavioralFlags(BaseModel):
    """Aider-compatible behavioral flags for harness configuration."""
    use_repo_map: bool = Field(default=True, description="Use repository map for context")
    examples_as_sys_msg: bool = Field(default=True, description="Put examples in system message")
    overeager: bool = Field(default=False, description="Model tends to over-edit/refactor")
    lazy: bool = Field(default=False, description="Model tends to produce incomplete output")
    cache_control: bool = Field(default=False, description="Supports prompt caching")
    supports_vision: bool = Field(default=False, description="Supports image inputs")
    supports_streaming: bool = Field(default=True, description="Supports streaming responses")


# === Quirks ===

class Quirks(BaseModel):
    """Structured quirks with tags for programmatic handling + free-form notes."""
    tags: list[QuirkTag] = Field(default_factory=list, description="Structured quirk tags")
    notes: list[str] = Field(default_factory=list, description="Free-form behavioral observations")


# === Heuristic Data ===

class AggregatorData(BaseModel):
    """Data pulled from aggregators (OpenRouter, BenchLM, etc.)."""
    source: str = Field(description="Aggregator name (openrouter, benchlm, artificial_analysis)")
    bfcl_score: float | None = Field(default=None, description="Berkeley Function Calling score")
    ifeval_score: float | None = Field(default=None, description="IFEval instruction following score")
    ruler_score: float | None = Field(default=None, description="RULER context utilization score")
    retrieved_at: datetime | None = Field(default=None, description="When data was fetched")


class HeuristicSignals(BaseModel):
    """Signals used for heuristic tier inference."""
    params_b: float | None = Field(default=None, description="Parameter count in billions")
    output_price_per_m: float | None = Field(default=None, description="Output price per 1M tokens USD")
    provider: str | None = Field(default=None, description="Provider name")
    model_family: str | None = Field(default=None, description="Model family (llama, qwen, etc.)")
    release_date: str | None = Field(default=None, description="Model release date")


class Heuristics(BaseModel):
    """Heuristic inference results."""
    overall_tier: Literal[1, 2, 3] = Field(description="Heuristic tier estimate")
    confidence: float = Field(ge=0, le=1, description="Confidence in tier assignment")
    aggregator_data: AggregatorData | None = Field(default=None)
    signals: HeuristicSignals = Field(default_factory=HeuristicSignals)


# === Metadata ===

class SamplingParams(BaseModel):
    """Sampling parameters used during calibration for reproducibility."""
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    seed: int | None = Field(default=42, description="Random seed if supported")


class Metadata(BaseModel):
    """Model metadata for harness configuration."""
    max_output_tokens: int = Field(default=4096, description="Max output tokens")
    cost_per_1m_input: float = Field(default=0, ge=0, description="Cost per 1M input tokens")
    cost_per_1m_output: float = Field(default=0, ge=0, description="Cost per 1M output tokens")
    weak_model_name: str | None = Field(default=None, description="Cheaper model for auxiliary tasks")
    editor_model_name: str | None = Field(default=None, description="Model for architect mode execution")


# === Main Manifest ===

class CapabilityManifest(BaseModel):
    """
    Complete capability manifest for a model.
    
    This is the primary output of calibration and the primary input for harnesses.
    """
    schema_version: str = Field(default="2.0.0", description="Manifest schema version")
    model_id: str = Field(description="Canonical model ID (provider/model:variant)")
    calibrated_at: datetime = Field(default_factory=datetime.utcnow)
    calibrator_version: str = Field(default="0.2.0")
    
    sampling_params: SamplingParams = Field(default_factory=SamplingParams)
    heuristics: Heuristics | None = Field(default=None)
    capabilities: Capabilities
    behavioral_flags: BehavioralFlags = Field(default_factory=BehavioralFlags)
    quirks: Quirks = Field(default_factory=Quirks)
    metadata: Metadata = Field(default_factory=Metadata)
    
    # === Convenience Properties ===
    
    @computed_field
    @property
    def best_edit_format(self) -> str:
        """Recommended edit format for this model."""
        return self.capabilities.edit_format.recommended.value
    
    @computed_field
    @property
    def max_reliable_context(self) -> int:
        """Maximum reliable context window (effective retrieval)."""
        return self.capabilities.context.effective_retrieval
    
    @computed_field
    @property
    def supports_tools(self) -> bool:
        """Whether model supports tool calling."""
        return self.capabilities.tool_calling.supported
    
    @computed_field
    @property
    def overall_tier(self) -> int:
        """Overall capability tier (average of capability tiers)."""
        tiers = [
            self.capabilities.edit_format.tier,
            self.capabilities.tool_calling.tier,
            self.capabilities.structured_output.tier,
            self.capabilities.context.tier,
            self.capabilities.instruction_adherence.tier,
            self.capabilities.multi_turn.tier,
        ]
        return round(sum(tiers) / len(tiers))
    
    @computed_field
    @property
    def is_stale(self) -> bool:
        """Whether manifest is older than 60 days."""
        return self.days_old > 60
    
    @property
    def days_old(self) -> int:
        """Days since calibration."""
        delta = datetime.utcnow() - self.calibrated_at
        return delta.days
