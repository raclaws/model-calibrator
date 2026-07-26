# Model Calibrator

Runtime capability calibration for LLMs in agentic coding harnesses.

## Overview

A library (not CLI) for discovering what a model can reliably do in an agentic coding context. Harnesses like Aider, Goose, and OpenCode embed this to auto-configure for any model.

## Installation

```bash
pip install model-calibrator
```

## Quick Start

```python
from model_calibrator import get_manifest, HeuristicInference, Registry

# Get manifest for a model (registry → heuristics → calibration)
manifest = get_manifest("ollama/llama3.2:8b-q4_K_M")

if manifest:
    print(f"Recommended edit format: {manifest.best_edit_format}")
    print(f"Supports tools: {manifest.supports_tools}")
    print(f"Max reliable context: {manifest.max_reliable_context}")
```

## Features

- **Heuristic inference**: Instant tier assignment from model metadata (params, price, provider)
- **Aggregator queries**: Pull existing benchmark data from OpenRouter, BenchLM
- **Calibration probes**: Edit format, tool calling, structured output, context depth, multi-turn
- **Community registry**: Pre-calibrated profiles for popular models
- **Per-capability tiers**: A model can be T3 for code but T1 for tool calling

## License

MIT
