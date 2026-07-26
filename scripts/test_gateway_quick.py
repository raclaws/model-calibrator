"""Quick validation test — 3 samples only."""

import os
from datetime import datetime
from model_calibrator.client import CalibrationClient
from model_calibrator.schema import EditFormat

GATEWAY_URL = "https://gateway.ai.cloudflare.com/v1/66bc302ceeffd5db7f4e1c191467acd8/default2/custom-deadcat/v1"
API_KEY = os.environ.get("HERMES_CUSTOM_GATEWAY_AI_CLOUDFLARE_COM_API_KEY")
MODEL = "kr/deepseek-3.2"

# Simple test file
SIMPLE_CODE = '''def add(a: int, b: int) -> int:
    return a + b

def subtract(a: int, b: int) -> int:
    return a - b
'''

def test_whole_format():
    """Test whole file edit."""
    client = CalibrationClient(GATEWAY_URL, API_KEY, timeout=60)
    
    prompt = f"""Here is a Python file:

```python
{SIMPLE_CODE}
```

Add a function `multiply(a: int, b: int) -> int` that returns the product.
Return the COMPLETE updated file in a code block."""

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Testing whole format...")
    response = client.complete_text(MODEL, prompt)
    
    success = "def multiply" in response and "```" in response
    print(f"  Result: {'✓' if success else '✗'}")
    print(f"  Response length: {len(response)} chars")
    if not success:
        print(f"  Response preview: {response[:200]}...")
    
    client.close()
    return success

def test_diff_format():
    """Test diff edit."""
    client = CalibrationClient(GATEWAY_URL, API_KEY, timeout=60)
    
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

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Testing diff format...")
    response = client.complete_text(MODEL, prompt, system=system)
    
    success = "<<<<<<< SEARCH" in response and "def minus" in response
    print(f"  Result: {'✓' if success else '✗'}")
    print(f"  Response length: {len(response)} chars")
    if not success:
        print(f"  Response preview: {response[:300]}...")
    
    client.close()
    return success

def test_tool_calling():
    """Test tool calling."""
    client = CalibrationClient(GATEWAY_URL, API_KEY, timeout=60)
    
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }]
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Testing tool calling...")
    try:
        response = client.complete_with_tools(
            MODEL,
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=tools,
        )
        
        message = response.get("choices", [{}])[0].get("message", {})
        tool_calls = message.get("tool_calls", [])
        
        success = len(tool_calls) > 0
        print(f"  Result: {'✓' if success else '✗'}")
        if success:
            print(f"  Tool called: {tool_calls[0].get('function', {}).get('name')}")
        else:
            print(f"  Response: {message.get('content', '')[:200]}...")
    except Exception as e:
        print(f"  Error: {e}")
        success = False
    
    client.close()
    return success

if __name__ == "__main__":
    print(f"Model: {MODEL}")
    print(f"Gateway: {GATEWAY_URL[:50]}...")
    print("-" * 50)
    
    results = {
        "whole": test_whole_format(),
        "diff": test_diff_format(),
        "tools": test_tool_calling(),
    }
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for test, passed in results.items():
        print(f"  {test}: {'✓ PASS' if passed else '✗ FAIL'}")
