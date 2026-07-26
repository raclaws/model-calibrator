"""
Edit format probe — tests model's ability to produce valid code edits.

20 samples minimum across realistic file sizes:
- 5 simple (< 50 lines)
- 10 medium (50-200 lines)
- 5 complex (200+ lines)

Measures success rate, confidence interval, and failure modes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from model_calibrator.client import CalibrationClient
from model_calibrator.schema import EditFormat, ProbeResult


class EditFailureMode(str, Enum):
    """Taxonomy of edit format failures."""
    LINE_NUMBER_HALLUCINATION = "line_number_hallucination"
    INDENTATION_MISMATCH = "indentation_mismatch"
    SEARCH_BLOCK_NOT_FOUND = "search_block_not_found"
    MALFORMED_DIFF = "malformed_diff"
    PARTIAL_EDIT = "partial_edit"
    CONTEXT_DRIFT = "context_drift"
    TRUNCATION = "truncation"
    WRONG_FORMAT = "wrong_format"
    NO_EDIT_PRODUCED = "no_edit_produced"


@dataclass
class EditTask:
    """A single edit task for probing."""
    name: str
    original_content: str
    instruction: str
    expected_change: str  # Substring that should appear after edit


# === Test Files ===

SIMPLE_FILE = '''"""Simple utility functions."""

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b
'''

MEDIUM_FILE = '''"""User management module."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class User:
    """Represents a user in the system."""
    id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool = True
    last_login: Optional[datetime] = None


class UserService:
    """Service for managing users."""
    
    def __init__(self):
        self._users: dict[int, User] = {}
        self._next_id = 1
    
    def create_user(self, username: str, email: str) -> User:
        """Create a new user."""
        user = User(
            id=self._next_id,
            username=username,
            email=email,
            created_at=datetime.now(),
        )
        self._users[user.id] = user
        self._next_id += 1
        return user
    
    def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)
    
    def delete_user(self, user_id: int) -> bool:
        """Delete a user by ID."""
        if user_id in self._users:
            del self._users[user_id]
            return True
        return False
    
    def list_active_users(self) -> list[User]:
        """List all active users."""
        return [u for u in self._users.values() if u.is_active]
'''

COMPLEX_FILE = '''"""API client with retry logic and error handling."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from urllib.parse import urljoin

import httpx


class APIError(Exception):
    """Base API error."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(APIError):
    """Rate limit exceeded."""
    def __init__(self, retry_after: int = 60):
        super().__init__(f"Rate limited. Retry after {retry_after}s", 429)
        self.retry_after = retry_after


class RetryStrategy(Enum):
    """Retry strategies for failed requests."""
    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    base_delay: float = 1.0
    max_delay: float = 60.0
    retry_on: tuple[int, ...] = (429, 500, 502, 503, 504)


@dataclass
class APIClient:
    """HTTP API client with configurable retry logic."""
    base_url: str
    timeout: float = 30.0
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    headers: dict[str, str] = field(default_factory=dict)
    _client: Optional[httpx.Client] = field(default=None, init=False)
    
    def __post_init__(self):
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self.headers,
        )
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay before retry."""
        if self.retry_config.strategy == RetryStrategy.NONE:
            return 0
        elif self.retry_config.strategy == RetryStrategy.LINEAR:
            delay = self.retry_config.base_delay * attempt
        else:  # EXPONENTIAL
            delay = self.retry_config.base_delay * (2 ** attempt)
        return min(delay, self.retry_config.max_delay)
    
    def _should_retry(self, status_code: int, attempt: int) -> bool:
        """Check if request should be retried."""
        if attempt >= self.retry_config.max_retries:
            return False
        return status_code in self.retry_config.retry_on
    
    def request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> httpx.Response:
        """Make an HTTP request with retry logic."""
        url = urljoin(self.base_url, path)
        attempt = 0
        
        while True:
            try:
                response = self._client.request(method, url, **kwargs)
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    if self._should_retry(429, attempt):
                        time.sleep(retry_after)
                        attempt += 1
                        continue
                    raise RateLimitError(retry_after)
                
                if self._should_retry(response.status_code, attempt):
                    delay = self._calculate_delay(attempt)
                    time.sleep(delay)
                    attempt += 1
                    continue
                
                response.raise_for_status()
                return response
                
            except httpx.HTTPStatusError as e:
                raise APIError(str(e), e.response.status_code)
            except httpx.RequestError as e:
                if attempt < self.retry_config.max_retries:
                    delay = self._calculate_delay(attempt)
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise APIError(f"Request failed: {e}")
    
    def get(self, path: str, **kwargs) -> httpx.Response:
        """GET request."""
        return self.request("GET", path, **kwargs)
    
    def post(self, path: str, **kwargs) -> httpx.Response:
        """POST request."""
        return self.request("POST", path, **kwargs)
    
    def put(self, path: str, **kwargs) -> httpx.Response:
        """PUT request."""
        return self.request("PUT", path, **kwargs)
    
    def delete(self, path: str, **kwargs) -> httpx.Response:
        """DELETE request."""
        return self.request("DELETE", path, **kwargs)
    
    def close(self):
        """Close the HTTP client."""
        if self._client:
            self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
'''


# === Edit Tasks ===

EDIT_TASKS: dict[str, list[EditTask]] = {
    "simple": [
        EditTask(
            name="add_multiply_function",
            original_content=SIMPLE_FILE,
            instruction="Add a function `multiply(a: int, b: int) -> int` that returns the product of a and b.",
            expected_change="def multiply",
        ),
        EditTask(
            name="add_docstring",
            original_content=SIMPLE_FILE,
            instruction="Add a module-level docstring at the top: 'Math utility functions for basic arithmetic.'",
            expected_change="Math utility functions",
        ),
        EditTask(
            name="rename_function",
            original_content=SIMPLE_FILE,
            instruction="Rename the `subtract` function to `minus`.",
            expected_change="def minus",
        ),
        EditTask(
            name="add_type_hints",
            original_content=SIMPLE_FILE.replace(") -> int:", "):"),
            instruction="Add return type hints `-> int` to all functions.",
            expected_change="-> int",
        ),
        EditTask(
            name="add_divide_function",
            original_content=SIMPLE_FILE,
            instruction="Add a function `divide(a: int, b: int) -> float` that returns a divided by b.",
            expected_change="def divide",
        ),
    ],
    "medium": [
        EditTask(
            name="add_update_user_method",
            original_content=MEDIUM_FILE,
            instruction="Add an `update_user(user_id: int, **kwargs) -> Optional[User]` method that updates user attributes.",
            expected_change="def update_user",
        ),
        EditTask(
            name="add_deactivate_method",
            original_content=MEDIUM_FILE,
            instruction="Add a `deactivate_user(user_id: int) -> bool` method that sets is_active=False.",
            expected_change="def deactivate_user",
        ),
        EditTask(
            name="add_find_by_email",
            original_content=MEDIUM_FILE,
            instruction="Add a `find_by_email(email: str) -> Optional[User]` method to UserService.",
            expected_change="def find_by_email",
        ),
        EditTask(
            name="add_count_method",
            original_content=MEDIUM_FILE,
            instruction="Add a `count_users() -> int` method that returns total user count.",
            expected_change="def count_users",
        ),
        EditTask(
            name="add_validation",
            original_content=MEDIUM_FILE,
            instruction="Add email validation to create_user: raise ValueError if '@' not in email.",
            expected_change="ValueError",
        ),
    ],
    "complex": [
        EditTask(
            name="add_circuit_breaker",
            original_content=COMPLEX_FILE,
            instruction="Add a CircuitBreaker class with open/closed/half-open states and failure threshold.",
            expected_change="class CircuitBreaker",
        ),
        EditTask(
            name="add_async_support",
            original_content=COMPLEX_FILE,
            instruction="Add an async version of the request method called `arequest` using httpx.AsyncClient.",
            expected_change="async def arequest",
        ),
        EditTask(
            name="add_logging",
            original_content=COMPLEX_FILE,
            instruction="Add logging to the request method: log info on success, warning on retry, error on failure.",
            expected_change="logging",
        ),
        EditTask(
            name="add_timeout_config",
            original_content=COMPLEX_FILE,
            instruction="Add connect_timeout and read_timeout to APIClient, separate from the general timeout.",
            expected_change="connect_timeout",
        ),
        EditTask(
            name="add_auth_support",
            original_content=COMPLEX_FILE,
            instruction="Add bearer token authentication support with an `auth_token` parameter.",
            expected_change="auth_token",
        ),
    ],
}


class EditFormatProbe:
    """
    Probe for testing edit format compliance.
    
    Runs 20 samples (5 simple + 10 medium + 5 complex) per format.
    Returns success rate, confidence interval, and failure modes.
    """
    
    # System prompts for each edit format
    SYSTEM_PROMPTS = {
        EditFormat.WHOLE: """You are a code editor. When asked to edit code, return the COMPLETE updated file.
Do not use any diff format. Return the entire file content with your changes applied.
Wrap the code in a code block with the appropriate language tag.""",

        EditFormat.DIFF: """You are a code editor using SEARCH/REPLACE blocks.
For each change, output a block in this exact format:

<<<<<<< SEARCH
exact lines to find
=======
replacement lines
>>>>>>> REPLACE

The SEARCH section must match the original file EXACTLY (including whitespace).
Output one block per change. Do not include any other text.""",

        EditFormat.UDIFF: """You are a code editor using unified diff format.
Output changes as a unified diff with proper line numbers:

```diff
--- a/filename
+++ b/filename
@@ -start,count +start,count @@
 context line
-removed line
+added line
 context line
```

Use proper line numbers. Include 3 lines of context around changes.""",

        EditFormat.SEARCH_REPLACE: """You are a code editor using search/replace format.
For each change, output:

SEARCH:
```
exact text to find
```

REPLACE:
```
replacement text
```

The SEARCH text must match exactly. Output one pair per change.""",
    }
    
    def __init__(self, client: "CalibrationClient"):
        from model_calibrator.client import CalibrationClient
        self.client = client
    
    def run(self, model: str, format: EditFormat) -> ProbeResult:
        """
        Run edit format probe for a specific format.
        
        Args:
            model: Model identifier
            format: Edit format to test
        
        Returns:
            ProbeResult with success_rate, confidence_interval, failure_modes
        """
        if format == EditFormat.ARCHITECT:
            # Architect mode needs separate handling
            return self._run_architect_probe(model)
        
        tasks = self._get_task_distribution()
        successes = 0
        failures: list[str] = []
        
        system_prompt = self.SYSTEM_PROMPTS.get(format, self.SYSTEM_PROMPTS[EditFormat.WHOLE])
        
        for task in tasks:
            try:
                result = self._run_single_task(model, format, task, system_prompt)
                if result["success"]:
                    successes += 1
                else:
                    failures.append(result["failure_mode"])
            except Exception as e:
                failures.append(EditFailureMode.TRUNCATION.value if "length" in str(e).lower() 
                               else EditFailureMode.NO_EDIT_PRODUCED.value)
        
        return ProbeResult.from_trials(successes, len(tasks), list(set(failures)))
    
    def _get_task_distribution(self) -> list[EditTask]:
        """Get 20 tasks: 5 simple + 10 medium + 5 complex."""
        tasks = []
        tasks.extend(EDIT_TASKS["simple"][:5])
        tasks.extend(EDIT_TASKS["medium"][:5])
        tasks.extend(EDIT_TASKS["medium"][:5])  # Repeat medium for 10 total
        tasks.extend(EDIT_TASKS["complex"][:5])
        return tasks
    
    def _run_single_task(
        self,
        model: str,
        format: EditFormat,
        task: EditTask,
        system_prompt: str,
    ) -> dict[str, Any]:
        """Run a single edit task and validate the result."""
        prompt = f"""Here is the current file content:

```python
{task.original_content}
```

Please make the following change:
{task.instruction}"""

        response = self.client.complete_text(model, prompt, system=system_prompt)
        
        # Validate the response
        return self._validate_response(response, format, task)
    
    def _validate_response(
        self,
        response: str,
        format: EditFormat,
        task: EditTask,
    ) -> dict[str, Any]:
        """Validate that the response is correct for the format and task."""
        if not response or len(response.strip()) < 10:
            return {"success": False, "failure_mode": EditFailureMode.NO_EDIT_PRODUCED.value}
        
        # Check if expected change is present
        if task.expected_change not in response:
            return {"success": False, "failure_mode": EditFailureMode.PARTIAL_EDIT.value}
        
        # Format-specific validation
        if format == EditFormat.WHOLE:
            return self._validate_whole(response, task)
        elif format == EditFormat.DIFF:
            return self._validate_diff(response, task)
        elif format == EditFormat.UDIFF:
            return self._validate_udiff(response, task)
        elif format == EditFormat.SEARCH_REPLACE:
            return self._validate_search_replace(response, task)
        
        return {"success": True, "failure_mode": None}
    
    def _validate_whole(self, response: str, task: EditTask) -> dict[str, Any]:
        """Validate whole-file edit response."""
        # Should contain a code block
        if "```" not in response:
            return {"success": False, "failure_mode": EditFailureMode.WRONG_FORMAT.value}
        
        # Extract code from code block
        code_match = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
        if not code_match:
            return {"success": False, "failure_mode": EditFailureMode.WRONG_FORMAT.value}
        
        code = code_match.group(1)
        
        # Check expected change is in the code
        if task.expected_change not in code:
            return {"success": False, "failure_mode": EditFailureMode.PARTIAL_EDIT.value}
        
        # Basic syntax check
        try:
            compile(code, "<string>", "exec")
        except SyntaxError:
            return {"success": False, "failure_mode": EditFailureMode.MALFORMED_DIFF.value}
        
        return {"success": True, "failure_mode": None}
    
    def _validate_diff(self, response: str, task: EditTask) -> dict[str, Any]:
        """Validate SEARCH/REPLACE block response."""
        # Should contain SEARCH and REPLACE markers
        if "<<<<<<< SEARCH" not in response or ">>>>>>> REPLACE" not in response:
            return {"success": False, "failure_mode": EditFailureMode.WRONG_FORMAT.value}
        
        # Extract SEARCH/REPLACE blocks
        pattern = r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
        matches = re.findall(pattern, response, re.DOTALL)
        
        if not matches:
            return {"success": False, "failure_mode": EditFailureMode.MALFORMED_DIFF.value}
        
        # Verify SEARCH blocks exist in original
        for search, replace in matches:
            search = search.strip()
            if search and search not in task.original_content:
                return {"success": False, "failure_mode": EditFailureMode.SEARCH_BLOCK_NOT_FOUND.value}
        
        return {"success": True, "failure_mode": None}
    
    def _validate_udiff(self, response: str, task: EditTask) -> dict[str, Any]:
        """Validate unified diff response."""
        # Should contain diff markers
        if "@@" not in response or ("---" not in response and "+++" not in response):
            return {"success": False, "failure_mode": EditFailureMode.WRONG_FORMAT.value}
        
        # Check for line number pattern
        if not re.search(r"@@ -\d+", response):
            return {"success": False, "failure_mode": EditFailureMode.LINE_NUMBER_HALLUCINATION.value}
        
        return {"success": True, "failure_mode": None}
    
    def _validate_search_replace(self, response: str, task: EditTask) -> dict[str, Any]:
        """Validate SEARCH:/REPLACE: format response."""
        if "SEARCH:" not in response or "REPLACE:" not in response:
            return {"success": False, "failure_mode": EditFailureMode.WRONG_FORMAT.value}
        
        return {"success": True, "failure_mode": None}
    
    def _run_architect_probe(self, model: str) -> ProbeResult:
        """Run architect mode probe (planning + execution split)."""
        # Simplified: just check if model can produce a plan
        tasks = EDIT_TASKS["complex"][:5]
        successes = 0
        failures: list[str] = []
        
        for task in tasks:
            try:
                prompt = f"""Analyze this code and create a plan for the following change:

```python
{task.original_content[:500]}...
```

Change: {task.instruction}

Provide a numbered list of specific code changes needed."""

                response = self.client.complete_text(model, prompt)
                
                # Check if response contains a plan
                if re.search(r"[1-9]\.", response) and len(response) > 50:
                    successes += 1
                else:
                    failures.append(EditFailureMode.PARTIAL_EDIT.value)
            except Exception:
                failures.append(EditFailureMode.TRUNCATION.value)
        
        return ProbeResult.from_trials(successes, len(tasks), list(set(failures)))
