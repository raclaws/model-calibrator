"""Calibrate OpenRouter free models and save to registry."""

import os
import httpx
import yaml
from datetime import datetime
from pathlib import Path

OPENROUTER_URL = "https://openrouter.ai/api/v1"
API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Free models to calibrate (coding-relevant ones)
MODELS = [
    {"id": "cohere/north-mini-code:free", "context": 256000, "provider": "cohere", "family": "north"},
    {"id": "google/gemma-4-31b-it:free", "context": 262144, "provider": "google", "family": "gemma"},
    {"id": "google/gemma-4-26b-a4b-it:free", "context": 262144, "provider": "google", "family": "gemma"},
    {"id": "nvidia/nemotron-3-super-120b-a12b:free", "context": 262144, "provider": "nvidia", "family": "nemotron"},
    {"id": "nvidia/nemotron-3-nano-30b-a3b:free", "context": 256000, "provider": "nvidia", "family": "nemotron"},
    {"id": "poolside/laguna-s-2.1:free", "context": 262144, "provider": "poolside", "family": "laguna"},
    {"id": "openai/gpt-oss-20b:free", "context": 131072, "provider": "openai", "family": "gpt-oss"},
]

SIMPLE_CODE = '''def add(a: int, b: int) -> int:
    return a + b

def subtract(a: int, b: int) -> int:
    return a - b
'''

MEDIUM_CODE = '''from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class User:
    id: int
    name: str
    email: str
    created_at: datetime
    is_active: bool = True

class UserService:
    def __init__(self):
        self._users: dict[int, User] = {}
        self._next_id = 1
    
    def create_user(self, name: str, email: str) -> User:
        user = User(
            id=self._next_id,
            name=name,
            email=email,
            created_at=datetime.utcnow(),
        )
        self._users[user.id] = user
        self._next_id += 1
        return user
    
    def get_user(self, user_id: int) -> Optional[User]:
        return self._users.get(user_id)
    
    def delete_user(self, user_id: int) -> bool:
        if user_id in self._users:
            del self._users[user_id]
            return True
        return False
'''

def openrouter_complete(model: str, prompt: str, system: str = None, timeout: int = 90) -> str:
    """Call OpenRouter API (OpenAI format)."""
    client = httpx.Client(timeout=timeout)
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": model,
        "max_tokens": 2048,
        "temperature": 0,
        "messages": messages,
    }
    
    response = client.post(
        f"{OPENROUTER_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/raclaws/model-calibrator",
        },
        json=payload,
    )
    response.raise_for_status()
    data = response.json()
    
    client.close()
    return data["choices"][0]["message"]["content"]

def test_whole_format(model: str) -> dict:
    """Test whole file edit."""
    prompt = f"""Here is a Python file:

```python
{SIMPLE_CODE}
```

Add a function `multiply(a: int, b: int) -> int` that returns the product.
Return the COMPLETE updated file in a code block."""

    try:
        response = openrouter_complete(model, prompt)
        success = "def multiply" in response and "```" in response
        return {"success": success, "response_len": len(response)}
    except Exception as e:
        return {"success": False, "error": str(e)[:100]}

def test_diff_format(model: str) -> dict:
    """Test SEARCH/REPLACE diff edit."""
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
        response = openrouter_complete(model, prompt, system=system)
        success = "<<<<<<< SEARCH" in response and "minus" in response
        return {"success": success, "response_len": len(response)}
    except Exception as e:
        return {"success": False, "error": str(e)[:100]}

def test_medium_whole(model: str) -> dict:
    """Test whole file edit on medium complexity file."""
    prompt = f"""Here is a Python file:

```python
{MEDIUM_CODE}
```

Add a method `update_user(user_id: int, **kwargs) -> Optional[User]` that updates user attributes.
Return the COMPLETE updated file in a code block."""

    try:
        response = openrouter_complete(model, prompt, timeout=120)
        success = "def update_user" in response and "```" in response
        return {"success": success, "response_len": len(response)}
    except Exception as e:
        return {"success": False, "error": str(e)[:100]}

def test_medium_diff(model: str) -> dict:
    """Test diff edit on medium complexity file."""
    system = """You are a code editor using SEARCH/REPLACE blocks.
For each change, output:
<<<<<<< SEARCH
exact lines to find
=======
replacement lines
>>>>>>> REPLACE"""

    prompt = f"""Here is a Python file:

```python
{MEDIUM_CODE}
```

Add email validation to create_user: raise ValueError if '@' not in email."""

    try:
        response = openrouter_complete(model, prompt, system=system, timeout=120)
        success = "<<<<<<< SEARCH" in response and "ValueError" in response
        return {"success": success, "response_len": len(response)}
    except Exception as e:
        return {"success": False, "error": str(e)[:100]}

def calibrate_model(model_info: dict) -> dict:
    """Run all tests for a model."""
    model_id = model_info["id"]
    print(f"\n{'='*60}")
    print(f"Calibrating: {model_id}")
    print(f"{'='*60}")
    
    results = {
        "model_id": model_id,
        "calibrated_at": datetime.utcnow().isoformat(),
        "metadata": model_info,
        "tests": {},
    }
    
    # Simple whole
    print(f"  [1/4] Simple whole format...", end=" ", flush=True)
    r = test_whole_format(model_id)
    results["tests"]["simple_whole"] = r
    print("✓" if r["success"] else f"✗ {r.get('error', '')[:40]}")
    
    # Simple diff
    print(f"  [2/4] Simple diff format...", end=" ", flush=True)
    r = test_diff_format(model_id)
    results["tests"]["simple_diff"] = r
    print("✓" if r["success"] else f"✗ {r.get('error', '')[:40]}")
    
    # Medium whole
    print(f"  [3/4] Medium whole format...", end=" ", flush=True)
    r = test_medium_whole(model_id)
    results["tests"]["medium_whole"] = r
    print("✓" if r["success"] else f"✗ {r.get('error', '')[:40]}")
    
    # Medium diff
    print(f"  [4/4] Medium diff format...", end=" ", flush=True)
    r = test_medium_diff(model_id)
    results["tests"]["medium_diff"] = r
    print("✓" if r["success"] else f"✗ {r.get('error', '')[:40]}")
    
    # Calculate scores
    tests = results["tests"]
    whole_score = sum(1 for k in ["simple_whole", "medium_whole"] if tests[k]["success"]) / 2
    diff_score = sum(1 for k in ["simple_diff", "medium_diff"] if tests[k]["success"]) / 2
    
    results["scores"] = {"whole": whole_score, "diff": diff_score}
    
    # Determine tier and recommended format
    if diff_score >= 0.75:
        results["tier"] = 3
        results["recommended_format"] = "diff"
    elif whole_score >= 0.75:
        results["tier"] = 2
        results["recommended_format"] = "whole"
    else:
        results["tier"] = 1
        results["recommended_format"] = "whole"
    
    print(f"\n  Results: whole={whole_score:.0%}, diff={diff_score:.0%}")
    print(f"  Tier: T{results['tier']}, Recommended: {results['recommended_format']}")
    
    return results

def main():
    print(f"OpenRouter Free Model Calibration")
    print(f"Models: {len(MODELS)}")
    
    all_results = []
    
    for model_info in MODELS:
        try:
            result = calibrate_model(model_info)
            all_results.append(result)
            
            # Save individual manifest
            output_dir = Path("/root/model-calibrator/registry/profiles")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = model_info["id"].replace("/", "-").replace(":", "-") + ".yaml"
            with open(output_dir / filename, "w") as f:
                yaml.dump(result, f, default_flow_style=False, sort_keys=False)
            print(f"  Saved: registry/profiles/{filename}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({"model_id": model_info["id"], "error": str(e)})
    
    # Print summary
    print(f"\n{'='*60}")
    print("CALIBRATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<45} {'Whole':>7} {'Diff':>7} {'Tier':>5} {'Format':>8}")
    print("-" * 75)
    
    for r in all_results:
        if "error" in r:
            print(f"{r['model_id']:<45} {'ERROR':>7}")
        else:
            scores = r["scores"]
            print(f"{r['model_id']:<45} {scores['whole']:>6.0%} {scores['diff']:>6.0%} "
                  f"T{r['tier']:>4} {r['recommended_format']:>8}")

if __name__ == "__main__":
    main()
