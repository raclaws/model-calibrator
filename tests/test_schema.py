"""Tests for schema models."""

import pytest
from datetime import datetime, timedelta

from model_calibrator.schema import (
    CapabilityManifest,
    Capabilities,
    EditFormat,
    EditFormatCapability,
    ToolCallingCapability,
    StructuredOutputCapability,
    ContextCapability,
    InstructionAdherenceCapability,
    MultiTurnCapability,
    ProbeResult,
    MeasuredRate,
    wilson_score_interval,
)


class TestWilsonScoreInterval:
    """Test Wilson score confidence interval calculation."""
    
    def test_wilson_50_percent(self):
        """50% success rate with 20 samples."""
        lower, upper = wilson_score_interval(10, 20)
        assert 0.28 < lower < 0.32
        assert 0.68 < upper < 0.72
    
    def test_wilson_100_percent(self):
        """100% success rate."""
        lower, upper = wilson_score_interval(20, 20)
        assert lower > 0.80
        assert upper == 1.0
    
    def test_wilson_0_percent(self):
        """0% success rate."""
        lower, upper = wilson_score_interval(0, 20)
        assert lower == 0.0
        assert upper < 0.20
    
    def test_wilson_empty(self):
        """Empty sample."""
        lower, upper = wilson_score_interval(0, 0)
        assert lower == 0.0
        assert upper == 1.0


class TestProbeResult:
    """Test ProbeResult model."""
    
    def test_from_trials(self):
        """Create ProbeResult from trial counts."""
        result = ProbeResult.from_trials(15, 20, ["error_a", "error_b"])
        assert result.success_rate == 0.75
        assert result.samples == 20
        assert len(result.confidence_interval) == 2
        assert result.failure_modes == ["error_a", "error_b"]
    
    def test_from_trials_perfect(self):
        """Perfect success rate."""
        result = ProbeResult.from_trials(20, 20)
        assert result.success_rate == 1.0
        assert result.failure_modes == []


class TestMeasuredRate:
    """Test MeasuredRate model."""
    
    def test_from_trials(self):
        """Create MeasuredRate from trial counts."""
        rate = MeasuredRate.from_trials(18, 20)
        assert rate.value == 0.9
        assert rate.samples == 20
        assert rate.confidence_interval[0] < 0.9
        assert rate.confidence_interval[1] > 0.9


class TestCapabilityManifest:
    """Test CapabilityManifest model."""
    
    def test_minimal_manifest(self):
        """Create minimal valid manifest."""
        manifest = CapabilityManifest(
            model_id="test/model",
            capabilities=Capabilities(
                edit_format=EditFormatCapability(
                    tier=2,
                    recommended=EditFormat.DIFF,
                ),
            ),
        )
        assert manifest.model_id == "test/model"
        assert manifest.schema_version == "2.0.0"
    
    def test_convenience_properties(self):
        """Test computed convenience properties."""
        manifest = CapabilityManifest(
            model_id="test/model",
            capabilities=Capabilities(
                edit_format=EditFormatCapability(tier=3, recommended=EditFormat.DIFF),
                tool_calling=ToolCallingCapability(tier=2, supported=True),
                structured_output=StructuredOutputCapability(tier=2),
                context=ContextCapability(tier=2, effective_retrieval=32000),
                instruction_adherence=InstructionAdherenceCapability(tier=2),
                multi_turn=MultiTurnCapability(tier=2),
            ),
        )
        
        assert manifest.best_edit_format == "diff"
        assert manifest.max_reliable_context == 32000
        assert manifest.supports_tools is True
        assert manifest.overall_tier == 2  # Average of all tiers
    
    def test_staleness_check(self):
        """Test is_stale property."""
        # Fresh manifest
        fresh = CapabilityManifest(
            model_id="test/model",
            calibrated_at=datetime.utcnow(),
            capabilities=Capabilities(
                edit_format=EditFormatCapability(tier=2, recommended=EditFormat.DIFF),
            ),
        )
        assert fresh.is_stale is False
        assert fresh.days_old < 1
        
        # Stale manifest (70 days old)
        stale = CapabilityManifest(
            model_id="test/model",
            calibrated_at=datetime.utcnow() - timedelta(days=70),
            capabilities=Capabilities(
                edit_format=EditFormatCapability(tier=2, recommended=EditFormat.DIFF),
            ),
        )
        assert stale.is_stale is True
        assert stale.days_old >= 70
