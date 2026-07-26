"""
Calibrator — orchestrates probes to produce a complete CapabilityManifest.

Tiered calibration:
- quick (~$0.50): edit_format only
- standard (~$2-3): edit_format + tool_calling + structured_output  
- thorough (~$5-10): all probes including context depth
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from model_calibrator.client import CalibrationClient
from model_calibrator.heuristics import HeuristicInference
from model_calibrator.probes.edit_format import EditFormatProbe
from model_calibrator.probes.tool_calling import ToolCallingProbe
from model_calibrator.registry import Registry
from model_calibrator.schema import (
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
    Quirks,
    QuirkTag,
    SamplingParams,
    StructuredOutputCapability,
    ToolCallingCapability,
    ToolCallingFormat,
)


CalibrationTier = Literal["quick", "standard", "thorough"]


class Calibrator:
    """
    Orchestrates calibration probes to produce a CapabilityManifest.
    
    Usage:
        calibrator = Calibrator(base_url, api_key)
        manifest = calibrator.calibrate("llama3.2:8b", tier="standard")
    """
    
    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0):
        """
        Initialize calibrator.
        
        Args:
            base_url: OpenAI-compatible API base URL
            api_key: API key
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self._client: CalibrationClient | None = None
    
    @property
    def client(self) -> CalibrationClient:
        """Lazy-init client."""
        if self._client is None:
            self._client = CalibrationClient(
                self.base_url,
                self.api_key,
                timeout=self.timeout,
            )
        return self._client
    
    def calibrate(
        self,
        model: str,
        tier: CalibrationTier = "standard",
        save_to_registry: bool = False,
        heuristic_metadata: dict | None = None,
    ) -> CapabilityManifest:
        """
        Run calibration probes and produce a manifest.
        
        Args:
            model: Model identifier (as accepted by the API)
            tier: Calibration tier (quick/standard/thorough)
            save_to_registry: Save result to local registry
            heuristic_metadata: Pre-fetched metadata (skips aggregator query)
        
        Returns:
            Complete CapabilityManifest
        """
        # Get heuristic data first
        if heuristic_metadata is None:
            heuristic_metadata = HeuristicInference.query_aggregators(model) or {}
        
        heuristic_tier, heuristic_conf = HeuristicInference.infer(heuristic_metadata)
        
        # Run probes based on tier
        edit_results = self._probe_edit_format(model)
        
        tool_results = None
        if tier in ("standard", "thorough"):
            tool_results = self._probe_tool_calling(model)
        
        # Build manifest
        manifest = self._build_manifest(
            model=model,
            heuristic_metadata=heuristic_metadata,
            heuristic_tier=heuristic_tier,
            heuristic_conf=heuristic_conf,
            edit_results=edit_results,
            tool_results=tool_results,
        )
        
        # Save to registry if requested
        if save_to_registry:
            Registry.save(manifest, trust_level="unverified")
        
        return manifest
    
    def _probe_edit_format(self, model: str) -> dict[str, ProbeResult]:
        """Run edit format probes for all formats."""
        probe = EditFormatProbe(self.client)
        results = {}
        
        for fmt in [EditFormat.WHOLE, EditFormat.DIFF]:
            # Only test whole and diff for efficiency
            # udiff and search_replace can be added for thorough tier
            results[fmt.value] = probe.run(model, fmt)
        
        return results
    
    def _probe_tool_calling(self, model: str) -> dict:
        """Run tool calling probe."""
        probe = ToolCallingProbe(self.client)
        return probe.run(model)
    
    def _build_manifest(
        self,
        model: str,
        heuristic_metadata: dict,
        heuristic_tier: int,
        heuristic_conf: float,
        edit_results: dict[str, ProbeResult],
        tool_results: dict | None,
    ) -> CapabilityManifest:
        """Build manifest from probe results."""
        # Determine best edit format
        best_format = EditFormat.WHOLE
        best_rate = 0.0
        for fmt_name, result in edit_results.items():
            if result.success_rate > best_rate:
                best_rate = result.success_rate
                best_format = EditFormat(fmt_name)
        
        # Calculate edit format tier
        if best_rate >= 0.9:
            edit_tier = 3
        elif best_rate >= 0.7:
            edit_tier = 2
        else:
            edit_tier = 1
        
        # Tool calling capability
        if tool_results:
            tool_cap = ToolCallingCapability(
                tier=tool_results.get("tier", 1),
                supported=tool_results.get("supported", False),
                native_format=tool_results.get("native_format", ToolCallingFormat.NONE),
                parallel_calls=tool_results.get("parallel_calls", False),
                schema_complexity_limit=tool_results.get("schema_complexity_limit", 0),
                hallucination_rate=tool_results.get("hallucination_rate"),
                required_param_compliance=tool_results.get("required_param_compliance"),
            )
        else:
            tool_cap = ToolCallingCapability(tier=heuristic_tier)
        
        # Detect quirks from probe results
        quirk_tags = []
        quirk_notes = []
        
        for fmt_name, result in edit_results.items():
            if "line_number_hallucination" in result.failure_modes:
                quirk_tags.append(QuirkTag.UDIFF_LINE_ERRORS)
            if "truncation" in result.failure_modes:
                quirk_tags.append(QuirkTag.TRUNCATION_RISK)
            if "indentation_mismatch" in result.failure_modes:
                quirk_notes.append(f"Indentation issues in {fmt_name} format")
        
        # Build heuristics section
        heuristics = Heuristics(
            overall_tier=heuristic_tier,
            confidence=heuristic_conf,
            signals=HeuristicSignals(
                params_b=heuristic_metadata.get("params_b"),
                output_price_per_m=heuristic_metadata.get("output_price_per_m"),
                provider=heuristic_metadata.get("provider"),
                model_family=heuristic_metadata.get("model_family"),
            ),
        )
        
        return CapabilityManifest(
            model_id=model,
            calibrated_at=datetime.utcnow(),
            sampling_params=SamplingParams(),
            heuristics=heuristics,
            capabilities=Capabilities(
                edit_format=EditFormatCapability(
                    tier=edit_tier,
                    recommended=best_format,
                    formats=edit_results,
                ),
                tool_calling=tool_cap,
                structured_output=StructuredOutputCapability(tier=min(edit_tier, 2)),
                context=ContextCapability(
                    tier=heuristic_tier,
                    advertised=heuristic_metadata.get("context_length", 4096),
                    effective_retrieval=int(heuristic_metadata.get("context_length", 4096) * 0.5),
                ),
                instruction_adherence=InstructionAdherenceCapability(tier=edit_tier),
                multi_turn=MultiTurnCapability(tier=max(1, edit_tier - 1)),
            ),
            behavioral_flags=BehavioralFlags(
                use_repo_map=edit_tier >= 2,
                supports_vision=heuristic_metadata.get("supports_vision", False),
            ),
            quirks=Quirks(tags=list(set(quirk_tags)), notes=quirk_notes),
            metadata=Metadata(
                cost_per_1m_input=heuristic_metadata.get("input_price_per_m", 0),
                cost_per_1m_output=heuristic_metadata.get("output_price_per_m", 0),
            ),
        )
    
    def close(self):
        """Close the client."""
        if self._client:
            self._client.close()
            self._client = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
