"""Input validation utilities for detecting potentially malicious content."""

import logging
import re

from openai.types import ModerationCreateResponse

from task_agent.config import settings

# Patterns for potentially malicious content
MALICIOUS_PATTERNS = {
    # Shell command injection patterns
    "shell_injection": [
        r";\s*(rm|mv|cp|dd|chmod|chown|cat|ls)\s+",  # Command chaining with dangerous commands
        r"\|\s*(rm|mv|cp|dd|chmod|chown|cat|ls|grep)\s+",  # Pipe to dangerous commands
        r"`.*\$.*`",  # Backtick command substitution with variables
        r"\$\(.*\)",  # Command substitution
        r'&&\s*(rm|mv|dd|chmod|cat|sudo)',  # AND operator with dangerous commands
        r'\|\|\s*(rm|mv|dd|cat|sudo)',  # OR operator with dangerous commands
        r'&&\s*sudo\s+',  # AND operator with sudo
        r'\|\|\s*sudo\s+',  # OR operator with sudo
    ],
    # Docker commands that could be abused
    "docker_abuse": [
        r"docker\s+(exec|run)\s+.*-v\s*/",  # Volume mounting with root
        r"docker\s+exec.*sudo",  # Docker exec with sudo
        r"docker\s+run.*--privileged",  # Privileged mode
        r"docker\s+run.*--pid=host",  # Host PID namespace
        r"docker\s+run.*--network=host",  # Host network
        r"docker\s+exec.*chmod",  # Docker exec with chmod
        r"docker\s+exec.*sh",  # Shell access in container
        r"docker\s+exec.*bash",  # Bash access in container
    ],
    # SQL injection patterns
    "sql_injection": [
        r"(\%27)|(\')|(\-\-)|(\%23)|(#)",  # SQL comment characters
        r"(\bor\b\s*\d+\s*=\s*\d+)",  # SQL bypass with OR
        r"(\band\b\s*\d+\s*=\s*\d+)",  # SQL bypass with AND
        r"union\s+select",  # UNION SELECT injection
        r"exec\s*\(",  # Command execution in SQL
    ],
    # Path traversal attempts
    "path_traversal": [
        r"\.\./",  # Parent directory traversal
        r"\.\.\\",  # Windows parent directory
        r"%2e%2e",  # URL encoded parent directory
        r"~root",  # Root home directory access
        r"/etc/passwd",  # Accessing password file
        r"/etc/shadow",  # Accessing shadow file
        r"C:\\Windows\\System32",  # Windows system directory
    ],
    # System commands and dangerous operations
    "system_commands": [
        r"\brm\s+-rf\s+/",  # Delete root filesystem
        r"\bdd\s+if=/dev/zero",  # Disk wipe
        r"\bmkfs\.",  # Format filesystem
        r"\bshutdown\b",  # Shutdown command
        r"\breboot\b",  # Reboot command
        r"\bsystemctl\s+(stop|disable|restart)",  # System service control
        r"\bservice\s+\w+\s+stop",  # Stop services
        r"\bkill\s+-9\s+\d+",  # Kill processes
        r"\bkillall\b",  # Kill all processes
        r"\binit\s+\d",  # Change runlevel
        r"\bnc\s+.*-e",  # Netcat with execute
        r"\bbash\s+-i",  # Interactive bash
        r"\bcurl\s+.*\|\s*bash",  # Curl pipe to bash
        r"\bwget\s+.*\|\s*sh",  # Wget pipe to shell
        r"\bchmod\s+777",  # Dangerous permission change
        r"\bchmod\s+-R",  # Recursive permission change
        r"\bsudo\s+(su|bash|sh)",  # Privilege escalation
    ],
    # Script/Code execution patterns
    "code_execution": [
        r"<script[^>]*>",  # Script tags
        r"javascript:",  # JavaScript protocol
        r"onload\s*=",  # Event handler injection
        r"onerror\s*=",  # Error handler injection
        r"eval\s*\(",  # Eval function
        r"exec\s*\(",  # Exec function
        r"system\s*\(",  # System function
        r"__import__\(",  # Python import injection
    ],
    # Known malicious keywords
    "malicious_keywords": [
        "malware", "virus", "trojan", "ransomware", "spyware",
        "keylogger", "backdoor", "rootkit", "botnet", "exploit",
        "payload", "shellcode", "injection", "bypass",
    ],
}

# Compile all regex patterns for better performance
COMPILED_PATTERNS = {}
for category, patterns in MALICIOUS_PATTERNS.items():
    # Skip malicious_keywords - it's handled separately with context checking
    if category == "malicious_keywords":
        continue
    if isinstance(patterns, list):
        COMPILED_PATTERNS[category] = [re.compile(p, re.IGNORECASE) for p in patterns]


async def scan_for_vulnerability(user_message: str) -> bool:
    """
    Scan user message for potentially malicious content.

    This function checks the input against various patterns that could indicate:
    - Shell command injection attempts
    - Docker container abuse
    - SQL injection patterns
    - Path traversal attempts
    - Dangerous system commands
    - Code execution attempts
    - Known malicious keywords

    Args:
        user_message: The user input to validate

    Returns:
        bool: True if the message appears safe (no vulnerabilities detected),
              False if potentially malicious content is detected

    Examples:
        >>> scan_for_vulnerability("Hello, how are you?")
        True
        >>> scan_for_vulnerability("rm -rf /")
        False
        >>> scan_for_vulnerability("docker run --privileged")
        False
    """
    logging.info(f"Scanning for vulnerabilities...")
    if not user_message or not isinstance(user_message, str):
        return True

    # Convert to lowercase for keyword matching
    message_lower = user_message.lower()

    # Check each category of malicious patterns
    for category, patterns in COMPILED_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(user_message):
                logging.warning(
                    f"Malicious content detected in category '{category}': "
                    f"pattern matched in message: {user_message[:100]}..."
                )
                return False

    # Check for malicious keywords
    keywords = MALICIOUS_PATTERNS.get("malicious_keywords", [])
    for keyword in keywords:
        if keyword in message_lower:
            # Only flag if keyword appears in a suspicious context
            # (not just mentioning the word in an educational context)
            suspicious_prefixes = ["create", "write", "build", "deploy", "install", "execute", "run", "download"]
            suspicious_suffixes = ["script", "code", "payload", "attack"]

            # Check if keyword appears near a suspicious prefix or suffix (within ~3 words)
            has_suspicious_context = False
            words = message_lower.split()

            for i, word in enumerate(words):
                if word == keyword or keyword in word:
                    # Check surrounding words
                    window_start = max(0, i - 3)
                    window_end = min(len(words), i + 4)
                    context_window = " ".join(words[window_start:window_end])

                    if any(prefix in context_window for prefix in suspicious_prefixes):
                        has_suspicious_context = True
                        break
                    if any(suffix in context_window for suffix in suspicious_suffixes):
                        has_suspicious_context = True
                        break

            if has_suspicious_context:
                logging.warning(
                    f"Malicious keyword '{keyword}' detected "
                    f"in suspicious context: {user_message[:100]}..."
                )
                return False

    if settings.MODERATION_API_CHECK_REQ:
        moderation_check_status: bool = await get_LLM_feedback_on_input(user_message)
        if not moderation_check_status:
            return False

    logging.info(f"Message passed vulnerability scan: {user_message[:50]}...")
    return True


async def get_vulnerability_details(user_message: str) -> dict:
    """
    Get detailed information about detected vulnerabilities.

    Args:
        user_message: The user input to analyze

    Returns:
        dict: A dictionary with:
            - is_safe (bool): Whether the message is safe
            - detected_issues (list): List of (category, pattern) tuples
            - risk_level (str): "none", "low", "medium", "high"

    Examples:
        >>> details = get_vulnerability_details("rm -rf /etc/passwd")
        >>> details["is_safe"]
        False
        >>> details["risk_level"]
        "high"
    """
    result = {
        "is_safe": True,
        "detected_issues": [],
        "risk_level": "none",
    }

    if not user_message or not isinstance(user_message, str):
        return result

    message_lower = user_message.lower()

    # Check each category
    for category, patterns in COMPILED_PATTERNS.items():
        if category == "malicious_keywords":
            continue  # Handle separately

        for pattern in patterns:
            if pattern.search(user_message):
                result["is_safe"] = False
                result["detected_issues"].append((category, pattern.pattern))

    # Check keywords
    keywords = MALICIOUS_PATTERNS.get("malicious_keywords", [])
    for keyword in keywords:
        if keyword in message_lower:
            suspicious_prefixes = ["create", "write", "build", "deploy", "install", "execute", "run", "download"]
            suspicious_suffixes = ["script", "code", "payload", "attack"]

            has_suspicious_context = False
            words = message_lower.split()

            for i, word in enumerate(words):
                if word == keyword or keyword in word:
                    window_start = max(0, i - 3)
                    window_end = min(len(words), i + 4)
                    context_window = " ".join(words[window_start:window_end])

                    if any(prefix in context_window for prefix in suspicious_prefixes):
                        has_suspicious_context = True
                        break
                    if any(suffix in context_window for suffix in suspicious_suffixes):
                        has_suspicious_context = True
                        break

            if has_suspicious_context:
                result["is_safe"] = False
                result["detected_issues"].append(("malicious_keyword", keyword))

    # Calculate risk level
    if not result["is_safe"]:
        issue_count = len(result["detected_issues"])
        high_risk_categories = {"system_commands", "docker_abuse", "shell_injection"}

        has_high_risk = any(
            cat in high_risk_categories for cat, _ in result["detected_issues"]
        )

        if has_high_risk or issue_count >= 3:
            result["risk_level"] = "high"
        elif issue_count >= 2:
            result["risk_level"] = "medium"
        else:
            result["risk_level"] = "low"

    return result


async def get_LLM_feedback_on_input(prompt: str) -> bool:
    """
    Check input using OpenAI's moderation API for harmful content.

    Uses OpenAI's moderation API to detect potentially harmful content including
    violence, self-harm, sexual content, hate speech, harassment, and more.

    Args:
        prompt: The user input to validate

    Returns:
        bool: True if the message appears safe (not flagged by moderation API),
              False if flagged as potentially harmful

    Examples:
        >>> import asyncio
        >>> result = asyncio.run(get_LLM_feedback_on_input("Hello"))
        True
        >>> result = asyncio.run(get_LLM_feedback_on_input("I want to hurt someone"))
        False
    """
    from openai import AsyncOpenAI

    logging.info("calling OpenAI's moderation API")
    try:
        client = AsyncOpenAI()

        response: ModerationCreateResponse = await client.moderations.create(
            model="omni-moderation-latest",
            input=prompt,
        )

        for res in response.results:
            if res.flagged:
                logging.warning(
                    f"Input flagged by moderation API. Categories: {res.categories}"
                )
                return False  # Unsafe - flagged

        logging.info("Input passed moderation API check")
        return True  # Safe - not flagged

    except Exception as e:
        logging.error(f"Moderation API error: {e}")
        # Fail closed for safety - treat API errors as unsafe
        return False
