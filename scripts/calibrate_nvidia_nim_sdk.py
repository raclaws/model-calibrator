"""Calibrate NVIDIA NIM models using OpenAI SDK with streaming."""

import time
import yaml
from datetime import datetime
from pathlib import Path
from openai import OpenAI

NVIDIA_KEY = "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"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_KEY
)

# Models that work via chat/completions
MODELS = [
    {"id": "poolside/laguna-xs-2.1", "provider": "poolside", "family": "laguna"},
    {"id": "meta/llama-3.2-3b-instruct", "provider": "meta", "family": "llama"},
    {"id": "meta/llama-3.3-70b-instruct", "provider": "meta", "family": "llama"},
    {"id": "qwen/qwen3-next-80b-a3b-instruct", "provider": "qwen", "family": "qwen"},
]

SIMPLE_CODE = '''def add(a: int, b: int) -> int:
    return a + b

def subtract(a: int, b: int) -> int:
    return a - b
'''

def stream_complete(model: str, messages: list, max_tokens: int = 2048, timeout: int = 120) -> str:
    """Stream completion and return full response."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            stream=True,
            timeout=timeout,
        )
        
        response = ""
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
        return response
    except Exception as e:
        raise e

def test_whole_format(model: str) -> dict:
    prompt = f"""Here is a Python file:

```python
{SIMPLE_CODE}
```

Add a function `multiply(a: int, b: int) -> int` that returns the product.
Return the COMPLETE updated file in a code block."""

    try:
        start = time.time()
        response = stream_complete(model, [{"role": "user", "content": prompt}])
        elapsed = time.time() - start
        success = "def multiply" in response and "```" in response
        return {"success": success, "time": round(elapsed, 1), "len": len(response)}
    except Exception as e:
        return {"success": False, "error": str(e)[:80]}

def test_diff_format(model: str) -> dict:
    system = """You are a code editor using SEARCH/REPLACE blocks.
For each change, output:
<<<<<<< SEARCH
exact lines to find
=======
replacement lines
>>>>>>> REPLACE"""

    prompt = f"""Here is a Python file:

```python
{SIMPLE_CODE}
```

Rename the function `subtract` to `minus`."""

    try:
        start = time.time()
        response = stream_complete(model, [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ])
        elapsed = time.time() - start
        success = "<<<<<<< SEARCH" in response and "minus" in response
        return {"success": success, "time": round(elapsed, 1), "len": len(response)}
    except Exception as e:
        return {"success": False, "error": str(e)[:80]}

def calibrate_model(model_info: dict) -> dict:
    model_id = model_info["id"]
    print(f"\n{'='*60}")
    print(f"Calibrating: {model_id}")
    print(f"{'='*60}")
    
    results = {
        "model_id": model_id,
        "source": "nvidia-nim",
        "calibrated_at": datetime.utcnow().isoformat(),
        "metadata": model_info,
        "tests": {},
    }
    
    print(f"  [1/2] Whole format...", end=" ", flush=True)
    r = test_whole_format(model_id)
    results["tests"]["whole"] = r
    if r["success"]:
        print(f"✓ ({r.get('time', '?')}s)")
    else:
        print(f"✗ {r.get('error', '')[:40]}")
    
    print(f"  [2/2] Diff format...", end=" ", flush=True)
    r = test_diff_format(model_id)
    results["tests"]["diff"] = r
    if r["success"]:
        print(f"✓ ({r.get('time', '?')}s)")
    else:
        print(f"✗ {r.get('error', '')[:40]}")
    
    tests = results["tests"]
    whole_ok = tests["whole"]["success"]
    diff_ok = tests["diff"]["success"]
    
    if diff_ok:
        results["tier"] = 3
        results["recommended_format"] = "diff"
    elif whole_ok:
        results["tier"] = 2
        results["recommended_format"] = "whole"
    else:
        results["tier"] = 1
        results["recommended_format"] = "whole"
    
    print(f"  Tier: T{results['tier']}, Recommended: {results['recommended_format']}")
    return results

def main():
    print(f"NVIDIA NIM Calibration (OpenAI SDK + streaming)")
    print(f"Models: {len(MODELS)}")
    
    all_results = []
    
    for model_info in MODELS:
        try:
            result = calibrate_model(model_info)
            all_results.append(result)
            
            output_dir = Path("/root/model-calibrator/registry/profiles")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = "nim-" + model_info["id"].replace("/", "-") + ".yaml"
            with open(output_dir / filename, "w") as f:
                yaml.dump(result, f, default_flow_style=False, sort_keys=False)
            print(f"  Saved: {filename}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({"model_id": model_info["id"], "error": str(e)})
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        if "error" in r:
            print(f"{r['model_id']}: ERROR")
        else:
            print(f"{r['model_id']}: T{r['tier']} → {r['recommended_format']}")

if __name__ == "__main__":
    main()
