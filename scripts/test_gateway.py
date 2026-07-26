"""Test calibrator against real gateway model."""

import os
from model_calibrator import Calibrator
import yaml

# Gateway config
GATEWAY_URL = "https://gateway.ai.cloudflare.com/v1/66bc302ceeffd5db7f4e1c191467acd8/default2/custom-deadcat/v1"
API_KEY = os.environ.get("HERMES_CUSTOM_GATEWAY_AI_CLOUDFLARE_COM_API_KEY")

# Test with a cheaper model
MODEL = "kr/deepseek-3.2"  # or "kr/claude-haiku-4.5"

def main():
    print(f"Testing calibration against: {MODEL}")
    print(f"Gateway: {GATEWAY_URL[:50]}...")
    print("-" * 60)
    
    with Calibrator(GATEWAY_URL, API_KEY, timeout=180) as calibrator:
        # Run quick calibration (edit_format only)
        manifest = calibrator.calibrate(
            model=MODEL,
            tier="quick",
            heuristic_metadata={
                "params_b": 685,  # DeepSeek-V3 is 685B MoE
                "provider": "deepseek",
                "model_family": "deepseek",
                "context_length": 128000,
            },
        )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"CALIBRATION RESULTS: {manifest.model_id}")
    print(f"{'='*60}")
    print(f"Overall tier: T{manifest.overall_tier}")
    print(f"Best edit format: {manifest.best_edit_format}")
    print(f"Supports tools: {manifest.supports_tools}")
    print(f"Max reliable context: {manifest.max_reliable_context:,}")
    print(f"\nEdit format results:")
    for fmt, result in manifest.capabilities.edit_format.formats.items():
        ci_low, ci_high = result.confidence_interval
        print(f"  {fmt}: {result.success_rate:.0%} ({ci_low:.0%}-{ci_high:.0%}, n={result.samples})")
        if result.failure_modes:
            print(f"    failures: {', '.join(result.failure_modes)}")
    
    if manifest.quirks.tags or manifest.quirks.notes:
        print(f"\nQuirks:")
        for tag in manifest.quirks.tags:
            print(f"  - {tag.value}")
        for note in manifest.quirks.notes:
            print(f"  - {note}")
    
    # Save manifest
    output_path = f"/root/model-calibrator/registry/profiles/{MODEL.replace('/', '-')}.yaml"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(manifest.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
    print(f"\nManifest saved to: {output_path}")


if __name__ == "__main__":
    main()
