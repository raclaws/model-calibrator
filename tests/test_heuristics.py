"""Tests for heuristic inference."""

import pytest
from model_calibrator.heuristics import HeuristicInference
from model_calibrator.schema import EditFormat


class TestHeuristicInference:
    """Test tier inference from metadata."""
    
    def test_infer_tier3_large_expensive_model(self):
        """Large, expensive model should be T3."""
        metadata = {
            "params_b": 70,
            "output_price_per_m": 15,
            "provider": "anthropic",
        }
        tier, confidence = HeuristicInference.infer(metadata)
        assert tier == 3
        assert confidence > 0.6
    
    def test_infer_tier2_medium_model(self):
        """Medium model should be T2."""
        metadata = {
            "params_b": 32,
            "output_price_per_m": 5,  # Higher price for T2
            "provider": "openai",     # Better provider reputation
        }
        tier, confidence = HeuristicInference.infer(metadata)
        assert tier == 2
        assert confidence > 0.5
    
    def test_infer_tier1_small_model(self):
        """Small model should be T1."""
        metadata = {
            "params_b": 7,
            "output_price_per_m": 0,
            "provider": "ollama",
        }
        tier, confidence = HeuristicInference.infer(metadata)
        assert tier == 1
        assert confidence > 0.4
    
    def test_infer_unknown_model(self):
        """Unknown model should default to T2 with low confidence."""
        metadata = {}
        tier, confidence = HeuristicInference.infer(metadata)
        assert tier == 2
        assert confidence < 0.3
    
    def test_infer_coder_family_boost(self):
        """Coder models get a boost."""
        base_metadata = {
            "params_b": 32,
            "output_price_per_m": 0,
            "provider": "ollama",
        }
        
        coder_metadata = {
            **base_metadata,
            "model_family": "qwen-coder",
        }
        
        base_tier, base_conf = HeuristicInference.infer(base_metadata)
        coder_tier, coder_conf = HeuristicInference.infer(coder_metadata)
        
        # Coder should have higher confidence at least
        assert coder_conf >= base_conf
    
    def test_to_manifest_creates_valid_manifest(self):
        """to_manifest should create a valid CapabilityManifest."""
        metadata = {
            "params_b": 70,
            "output_price_per_m": 15,
            "provider": "anthropic",
            "context_length": 200000,
            "supports_tools": True,
            "_aggregator_source": "openrouter",
        }
        
        manifest = HeuristicInference.to_manifest(
            "anthropic/claude-sonnet-4",
            metadata,
            tier=3,
            confidence=0.9,
        )
        
        assert manifest.model_id == "anthropic/claude-sonnet-4"
        assert manifest.capabilities.edit_format.tier == 3
        assert manifest.capabilities.edit_format.recommended == EditFormat.DIFF
        assert manifest.capabilities.tool_calling.supported is True
        assert manifest.capabilities.context.advertised == 200000
        assert manifest.heuristics.overall_tier == 3
        assert manifest.heuristics.confidence == 0.9


class TestAggregatorNormalization:
    """Test model ID normalization for aggregator queries."""
    
    def test_normalize_ollama_format(self):
        """Ollama format should normalize correctly."""
        normalized = HeuristicInference._normalize_for_search("ollama/llama3.2:8b")
        assert "llama3" in normalized
        assert "ollama" not in normalized
    
    def test_normalize_quantization_suffix(self):
        """Quantization suffix should be removed."""
        normalized = HeuristicInference._normalize_for_search("llama3.2:8b-q4_K_M")
        assert "q4" not in normalized.lower()
    
    def test_extract_params(self):
        """Parameter extraction from model name."""
        assert HeuristicInference._estimate_params("llama-3.2-70b") == 70
        assert HeuristicInference._estimate_params("qwen2.5-coder-32b") == 32
        assert HeuristicInference._estimate_params("phi-3-mini") == 3  # "mini" hint
    
    def test_extract_provider(self):
        """Provider extraction from model ID."""
        assert HeuristicInference._extract_provider("anthropic/claude-3") == "anthropic"
        assert HeuristicInference._extract_provider("meta-llama/llama-3") == "meta"
        assert HeuristicInference._extract_provider("mistralai/mistral-7b") == "mistral"
    
    def test_extract_family(self):
        """Family extraction from model ID."""
        assert "llama" in HeuristicInference._extract_family("meta-llama/llama-3.2-70b")
        assert "qwen" in HeuristicInference._extract_family("qwen/qwen2.5-coder-32b")
        assert "coder" in HeuristicInference._extract_family("deepseek/deepseek-coder-33b")
