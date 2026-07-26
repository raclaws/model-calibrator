"""
Tool calling probe — tests model's ability to use function/tool calling.

Tests:
- Native format detection (OpenAI functions, Anthropic tool_use, raw JSON)
- Schema complexity (5, 10, 20, 50 tools)
- Hallucination resistance (don't call non-existent tools)
- Parallel tool calls
- Required vs optional parameter handling
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from model_calibrator.client import CalibrationClient
from model_calibrator.schema import MeasuredRate, ToolCallingFormat


class ToolFailureMode(str, Enum):
    """Taxonomy of tool calling failures."""
    PHANTOM_CALL = "phantom_call"  # Called non-existent tool
    HALLUCINATED_PARAMS = "hallucinated_params"  # Made up parameter names
    MISSING_REQUIRED = "missing_required"  # Didn't provide required param
    WRONG_TYPE = "wrong_type"  # Wrong parameter type
    MALFORMED_JSON = "malformed_json"  # Invalid JSON in arguments
    NO_TOOL_CALL = "no_tool_call"  # Didn't call any tool when expected
    WRONG_TOOL = "wrong_tool"  # Called wrong tool for the task


# === Tool Definitions ===

SIMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a file with content",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"},
                },
                "required": ["expression"],
            },
        },
    },
]


COMPLEX_TOOL = {
    "type": "function",
    "function": {
        "name": "search_database",
        "description": "Search a database with filters",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "operator": {"type": "string", "enum": ["eq", "gt", "lt", "contains"]},
                            "value": {"type": "string"},
                        },
                        "required": ["field", "operator", "value"],
                    },
                },
                "sort_by": {"type": "string"},
                "sort_order": {"type": "string", "enum": ["asc", "desc"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "offset": {"type": "integer", "minimum": 0},
            },
            "required": ["query"],
        },
    },
}


# === Test Cases ===

@dataclass
class ToolTestCase:
    """A single tool calling test case."""
    name: str
    prompt: str
    expected_tool: str | None  # None means no tool should be called
    required_params: list[str]
    tools: list[dict]


TOOL_TEST_CASES = [
    # Simple tool calls
    ToolTestCase(
        name="simple_weather",
        prompt="What's the weather in Tokyo?",
        expected_tool="get_weather",
        required_params=["location"],
        tools=SIMPLE_TOOLS,
    ),
    ToolTestCase(
        name="simple_search",
        prompt="Search for recent news about AI",
        expected_tool="search_web",
        required_params=["query"],
        tools=SIMPLE_TOOLS,
    ),
    ToolTestCase(
        name="simple_email",
        prompt="Send an email to john@example.com with subject 'Meeting' and body 'Let's meet tomorrow'",
        expected_tool="send_email",
        required_params=["to", "subject", "body"],
        tools=SIMPLE_TOOLS,
    ),
    ToolTestCase(
        name="simple_file",
        prompt="Create a file at /tmp/test.txt with content 'Hello World'",
        expected_tool="create_file",
        required_params=["path", "content"],
        tools=SIMPLE_TOOLS,
    ),
    ToolTestCase(
        name="simple_calc",
        prompt="Calculate 15 * 7 + 23",
        expected_tool="calculate",
        required_params=["expression"],
        tools=SIMPLE_TOOLS,
    ),
    # Hallucination trap - no tool should be called
    ToolTestCase(
        name="hallucination_trap_1",
        prompt="What is the capital of France?",
        expected_tool=None,
        required_params=[],
        tools=SIMPLE_TOOLS,
    ),
    ToolTestCase(
        name="hallucination_trap_2",
        prompt="Explain quantum computing",
        expected_tool=None,
        required_params=[],
        tools=SIMPLE_TOOLS,
    ),
    # Complex tool
    ToolTestCase(
        name="complex_search",
        prompt="Search the database for users where age > 25, sorted by name ascending, limit 10",
        expected_tool="search_database",
        required_params=["query"],
        tools=[COMPLEX_TOOL],
    ),
]


class ToolCallingProbe:
    """
    Probe for testing tool/function calling capabilities.
    
    Tests:
    - Native format detection
    - Schema complexity limits
    - Hallucination resistance
    - Parallel tool calls
    - Required parameter compliance
    """
    
    def __init__(self, client: "CalibrationClient"):
        from model_calibrator.client import CalibrationClient
        self.client = client
    
    def run(self, model: str) -> dict[str, Any]:
        """
        Run complete tool calling probe suite.
        
        Returns dict with:
        - tier: 1-3
        - supported: bool
        - native_format: ToolCallingFormat
        - parallel_calls: bool
        - schema_complexity_limit: int
        - hallucination_rate: MeasuredRate
        - required_param_compliance: MeasuredRate
        """
        results = {
            "tier": 1,
            "supported": False,
            "native_format": ToolCallingFormat.NONE,
            "parallel_calls": False,
            "schema_complexity_limit": 0,
            "hallucination_rate": None,
            "required_param_compliance": None,
        }
        
        # Test if tool calling is supported at all
        format_detected = self._detect_native_format(model)
        if format_detected == ToolCallingFormat.NONE:
            return results
        
        results["supported"] = True
        results["native_format"] = format_detected
        
        # Run hallucination and compliance tests
        hallucination_results = self._test_hallucination_resistance(model)
        compliance_results = self._test_param_compliance(model)
        
        results["hallucination_rate"] = hallucination_results
        results["required_param_compliance"] = compliance_results
        
        # Test schema complexity
        results["schema_complexity_limit"] = self._test_schema_complexity(model)
        
        # Test parallel calls
        results["parallel_calls"] = self._test_parallel_calls(model)
        
        # Calculate tier
        results["tier"] = self._calculate_tier(results)
        
        return results
    
    def _detect_native_format(self, model: str) -> ToolCallingFormat:
        """Detect which tool calling format the model supports."""
        test_prompt = "What's the weather in Paris?"
        
        try:
            response = self.client.complete_with_tools(
                model=model,
                messages=[{"role": "user", "content": test_prompt}],
                tools=SIMPLE_TOOLS[:2],
                tool_choice="auto",
            )
            
            message = response.get("choices", [{}])[0].get("message", {})
            
            # Check for OpenAI-style tool_calls
            if message.get("tool_calls"):
                return ToolCallingFormat.OPENAI_FUNCTIONS
            
            # Check for Anthropic-style content blocks
            content = message.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        return ToolCallingFormat.ANTHROPIC_TOOL_USE
            
            # Check if model outputs raw JSON
            if isinstance(content, str) and "{" in content:
                try:
                    parsed = json.loads(content)
                    if "name" in parsed or "function" in parsed:
                        return ToolCallingFormat.RAW_JSON
                except json.JSONDecodeError:
                    pass
            
            return ToolCallingFormat.NONE
            
        except Exception:
            return ToolCallingFormat.NONE
    
    def _test_hallucination_resistance(self, model: str, n: int = 20) -> MeasuredRate:
        """Test how often model calls non-existent tools or hallucinates params."""
        hallucinations = 0
        
        # Mix of valid tool calls and hallucination traps
        test_cases = TOOL_TEST_CASES * 3  # Repeat to get enough samples
        test_cases = test_cases[:n]
        
        for case in test_cases:
            try:
                response = self.client.complete_with_tools(
                    model=model,
                    messages=[{"role": "user", "content": case.prompt}],
                    tools=case.tools,
                    tool_choice="auto",
                )
                
                message = response.get("choices", [{}])[0].get("message", {})
                tool_calls = message.get("tool_calls", [])
                
                # Hallucination trap: no tool should be called
                if case.expected_tool is None:
                    if tool_calls:
                        hallucinations += 1
                    continue
                
                # Valid case: check if correct tool called
                if not tool_calls:
                    continue  # Not a hallucination, just no call
                
                called_name = tool_calls[0].get("function", {}).get("name", "")
                valid_names = {t["function"]["name"] for t in case.tools}
                
                if called_name not in valid_names:
                    hallucinations += 1  # Called non-existent tool
                    
            except Exception:
                continue
        
        # Hallucination rate (lower is better, so we return rate of hallucinations)
        return MeasuredRate.from_trials(hallucinations, n)
    
    def _test_param_compliance(self, model: str, n: int = 20) -> MeasuredRate:
        """Test how often model provides all required parameters."""
        compliant = 0
        
        test_cases = [c for c in TOOL_TEST_CASES if c.expected_tool is not None]
        test_cases = (test_cases * 5)[:n]  # Repeat to get enough samples
        
        for case in test_cases:
            try:
                response = self.client.complete_with_tools(
                    model=model,
                    messages=[{"role": "user", "content": case.prompt}],
                    tools=case.tools,
                    tool_choice="auto",
                )
                
                message = response.get("choices", [{}])[0].get("message", {})
                tool_calls = message.get("tool_calls", [])
                
                if not tool_calls:
                    continue
                
                # Parse arguments
                args_str = tool_calls[0].get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    continue
                
                # Check if all required params present
                if all(param in args for param in case.required_params):
                    compliant += 1
                    
            except Exception:
                continue
        
        return MeasuredRate.from_trials(compliant, n)
    
    def _test_schema_complexity(self, model: str) -> int:
        """Test max number of tools before degradation."""
        max_working = 0
        
        for n_tools in [5, 10, 20, 50]:
            # Generate n_tools tool definitions
            tools = self._generate_n_tools(n_tools)
            
            # Test if model can correctly select from them
            success_count = 0
            for _ in range(3):  # 3 trials per complexity level
                try:
                    # Pick a random tool to test
                    target_tool = tools[n_tools // 2]
                    prompt = f"Use the {target_tool['function']['name']} tool"
                    
                    response = self.client.complete_with_tools(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        tools=tools,
                        tool_choice="auto",
                    )
                    
                    message = response.get("choices", [{}])[0].get("message", {})
                    tool_calls = message.get("tool_calls", [])
                    
                    if tool_calls:
                        called = tool_calls[0].get("function", {}).get("name", "")
                        if called == target_tool["function"]["name"]:
                            success_count += 1
                except Exception:
                    pass
            
            # Need at least 2/3 success to consider this complexity level working
            if success_count >= 2:
                max_working = n_tools
            else:
                break
        
        return max_working
    
    def _test_parallel_calls(self, model: str, n: int = 5) -> bool:
        """Test if model can make multiple tool calls in one response."""
        success_count = 0
        
        for _ in range(n):
            try:
                prompt = "Get the weather in Tokyo and search for 'latest AI news'"
                
                response = self.client.complete_with_tools(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    tools=SIMPLE_TOOLS,
                    tool_choice="auto",
                )
                
                message = response.get("choices", [{}])[0].get("message", {})
                tool_calls = message.get("tool_calls", [])
                
                if len(tool_calls) >= 2:
                    success_count += 1
                    
            except Exception:
                pass
        
        # Consider parallel calls supported if >50% success
        return success_count > n // 2
    
    def _generate_n_tools(self, n: int) -> list[dict]:
        """Generate n unique tool definitions."""
        tools = []
        for i in range(n):
            tools.append({
                "type": "function",
                "function": {
                    "name": f"tool_{i}",
                    "description": f"Tool number {i} for testing",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string", "description": "Input value"},
                        },
                        "required": ["input"],
                    },
                },
            })
        return tools
    
    def _calculate_tier(self, results: dict) -> int:
        """Calculate tool calling tier from results."""
        if not results["supported"]:
            return 1
        
        hallucination = results["hallucination_rate"]
        compliance = results["required_param_compliance"]
        complexity = results["schema_complexity_limit"]
        
        # T3: Low hallucination (<5%), high compliance (>95%), handles 20+ tools
        if (hallucination and hallucination.value < 0.05 and
            compliance and compliance.value > 0.95 and
            complexity >= 20):
            return 3
        
        # T2: Moderate hallucination (<15%), decent compliance (>80%), handles 10+ tools
        if (hallucination and hallucination.value < 0.15 and
            compliance and compliance.value > 0.80 and
            complexity >= 10):
            return 2
        
        return 1
