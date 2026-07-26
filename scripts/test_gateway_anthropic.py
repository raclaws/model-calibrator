"""Quick validation test — Anthropic Messages API format."""

import os
import httpx
from datetime import datetime

GATEWAY_URL = "https://gateway.ai.cloudflare.com/v1/66bc302ceeffd5db7f4e1c191467acd8/default2/custom-deadcat/v1"
API_KEY = os.environ.get("HERMES_CUSTOM_GATEWAY_AI_CLOUDFLARE_COM_API_KEY")
MODEL = "kr/deepseek-3.2"

# Simple test file
SIMPLE_CODE = '''def add(a: int, b: int) -> int:
    return a + b

def subtract(a: int, b: int) -> int:
    return a - b
'''

def anthropic_complete(prompt: str, system: str = None) -> str:
    """Call gateway API (returns OpenAI format despite /messages endpoint)."""
    client = httpx.Client(timeout=120)
    
    payload = {
        "model": MODEL,
        "max_tokens": 2048,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system
    
    response = client.post(
        f"{GATEWAY_URL}/messages",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        json=payload,
    )
    
    response.raise_for_status()
    data = response.json()
    
    # Gateway returns OpenAI format: choices[0].message.content
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    
    # Fallback: Anthropic format
    content = data.get("content", [])
    text = ""
    for block in content:
        if block.get("type") == "text":
            text += block.get("text", "")
    
    client.close()
    return text

def test_whole_format():
    """Test whole file edit."""
    prompt = f"""Here is a Python file:

```python
{SIMPLE_CODE}
```

Add a function `multiply(a: int, b: int) -> int` that returns the product.
Return the COMPLETE updated file in a code block."""

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Testing whole format...")
    response = anthropic_complete(prompt)
    
    success = "def multiply" in response and "```" in response
    print(f"  Result: {'✓' if success else '✗'}")
    print(f"  Response length: {len(response)} chars")
    if not success:
        print(f"  Response preview: {response[:300]}...")
    return success

def test_diff_format():
    """Test diff edit."""
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
    response = anthropic_complete(prompt, system=system)
    
    success = "<<<<<<< SEARCH" in response and "minus" in response
    print(f"  Result: {'✓' if success else '✗'}")
    print(f"  Response length: {len(response)} chars")
    if not success:
        print(f"  Response preview: {response[:300]}...")
    return success

def test_tool_calling():
    """Test tool calling via OpenAI format."""
    client = httpx.Client(timeout=120)
    
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
    
    payload = {
        "model": MODEL,
        "max_tokens": 1024,
        "stream": False,
        "tools": tools,
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
    }
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Testing tool calling...")
    try:
        response = client.post(
            f"{GATEWAY_URL}/messages",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        
        # OpenAI format: choices[0].message.tool_calls
        if "choices" in data:
            message = data["choices"][0].get("message", {})
            tool_calls = message.get("tool_calls", [])
            success = len(tool_calls) > 0
            print(f"  Result: {'✓' if success else '✗'}")
            if success:
                print(f"  Tool called: {tool_calls[0].get('function', {}).get('name')}")
            else:
                print(f"  Response: {message.get('content', '')[:200]}...")
        else:
            # Anthropic format fallback
            content = data.get("content", [])
            tool_uses = [b for b in content if b.get("type") == "tool_use"]
            success = len(tool_uses) > 0
            print(f"  Result: {'✓' if success else '✗'}")
            if success:
                print(f"  Tool called: {tool_uses[0].get('name')}")
            else:
                text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
                print(f"  Response: {text[:200]}...")
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
