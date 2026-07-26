# Model Calibrator

Runtime capability calibration for LLMs in agentic coding harnesses.

## Problem

Aider works because Paul maintains model-specific tuning — different edit formats, prompts, and temperatures per model family. This knowledge is embedded in code, not a portable spec. Other harnesses (OpenCode, Goose, Claude Code) either hardcode frontier model assumptions or fail silently on untested models.

**Example:** GLM-5 scores 100% on whole-file edits but **0% on SEARCH/REPLACE diffs** — without calibration, a harness would try diff mode and get garbage.

## Solution

A library (not CLI) that discovers what a model can reliably do. Harnesses embed this to auto-configure for any model.

```python
from model_calibrator import Registry, HeuristicInference, Calibrator

# 1. Check pre-calibrated registry first
manifest = Registry.get("google/gemma-4-26b-a4b-it:free")

# 2. Or infer from metadata (instant, free)
metadata = HeuristicInference.query_aggregators("meta-llama/llama-3.2-8b")
tier, confidence = HeuristicInference.infer(metadata)

# 3. Or run calibration probes (~$0.50-2.00)
with Calibrator(base_url, api_key) as cal:
    manifest = cal.calibrate("ollama/qwen2.5-coder:32b", tier="standard")

print(f"Recommended: {manifest.best_edit_format}")  # "diff" or "whole"
print(f"Tier: T{manifest.overall_tier}")            # 1, 2, or 3
```

## Pre-Calibrated Registry (13 models)

| Model | Source | Whole | Diff | Tier | Recommended |
|-------|--------|-------|------|------|-------------|
| kr/deepseek-3.2 | Gateway | 100% | 100% | T3 | diff |
| kr/claude-haiku-4.5 | Gateway | 100% | 100% | T3 | diff |
| kr/minimax-m2.5 | Gateway | 100% | 100% | T3 | diff |
| kr/glm-5 | Gateway | 100% | **0%** | T2 | whole |
| google/gemma-4-26b-a4b-it:free | OpenRouter | 100% | 100% | T3 | diff |
| nvidia/nemotron-3-super-120b-a12b:free | OpenRouter | 100% | 100% | T3 | diff |
| poolside/laguna-s-2.1:free | OpenRouter | 100% | 100% | T3 | diff |
| openai/gpt-oss-20b:free | OpenRouter | 100% | 100% | T3 | diff |
| cohere/north-mini-code:free | OpenRouter | 50% | 50% | T1 | whole |
| nvidia/nemotron-3-nano-30b-a3b:free | OpenRouter | 50% | 50% | T1 | whole |
| nim/poolside-laguna-xs-2.1 | NVIDIA NIM | 100% | 100% | T3 | diff |
| nim/meta-llama-3.2-3b-instruct | NVIDIA NIM | 100% | 100% | T3 | diff |
| nim/meta-llama-3.3-70b-instruct | NVIDIA NIM | 100% | timeout | T2 | whole |

## Key Findings

1. **GLM-5** — Cannot produce SEARCH/REPLACE blocks at all (0% diff success)
2. **Cohere North Mini, Nemotron Nano** — Fail on medium-complexity files (50% overall)
3. **Llama 3.3 70B** — Works but 169s latency causes diff timeouts on NIM
4. **Poolside Laguna XS** — Surprisingly fast (1.5s whole, 0.8s diff)

## Tier System

| Tier | Profile | Edit Format | Tool Calling |
|------|---------|-------------|--------------|
| T1 | Fast/cheap | whole only | Basic/none |
| T2 | Balanced | whole reliable, diff partial | Reliable |
| T3 | Maximum | diff, udiff, architect | Advanced |

Per-capability tiers: A model can be T3 for code but T1 for tool calling.

## Features

- **Heuristic inference**: Instant tier from metadata (params, price, provider, family)
- **Aggregator queries**: Pull benchmark data from OpenRouter API
- **Calibration probes**: Edit format, tool calling, structured output
- **Wilson score CIs**: Statistical validity (20 samples minimum, 95% confidence)
- **Per-capability tiers**: Not just one overall tier
- **Aider-compatible flags**: `use_repo_map`, `overeager`, `lazy`, `examples_as_sys_msg`

## Installation

```bash
pip install model-calibrator
```

Or from source:
```bash
git clone https://github.com/raclaws/model-calibrator
cd model-calibrator
pip install -e ".[dev]"
```

## Project Structure

```
model-calibrator/
├── src/model_calibrator/
│   ├── schema.py        # CapabilityManifest, per-capability tiers, Wilson CI
│   ├── heuristics.py    # Aggregator queries, sigmoid tier inference
│   ├── client.py        # OpenAI-compatible API client (temp=0, seed=42)
│   ├── calibrator.py    # Orchestrates probes → manifest
│   ├── registry.py      # Local cache + remote profiles
│   └── probes/
│       ├── edit_format.py   # 20 samples, 4 formats, validation
│       └── tool_calling.py  # Format detection, hallucination, parallel
├── registry/profiles/   # Pre-calibrated YAML manifests
├── scripts/             # Calibration scripts per provider
└── tests/               # 21 tests passing
```

## RFC

Full specification at [geniza/specs/model-capability-manifest-rfc.md](https://github.com/raclaws/geniza/blob/main/specs/model-capability-manifest-rfc.md)

## Roadmap

- [x] Phase 1: Schema + heuristics
- [x] Phase 2: Edit format + tool calling probes  
- [x] Phase 3: Calibrator orchestrator
- [x] Phase 4: Pre-populate registry (13 models)
- [ ] Phase 5: Harness adapters (Aider, OpenCode)
- [ ] Phase 6: Community registry + submission process

## License

MIT
