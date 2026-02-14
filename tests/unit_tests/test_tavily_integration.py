"""
Test Tavily integration to verify search results and formatting
"""
import asyncio
import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from task_agent.utils.nodes import format_tavily_result


@pytest.mark.asyncio
async def test_tavily_direct_call():
    """Test direct Tavily API call to see actual response structure"""
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        pytest.skip("TAVILY_API_KEY not set")

    tavily = TavilyClient(api_key=api_key)

    result = tavily.search(query="why sky is blue", max_results=3)

    # Debug: print structure
    print(f"\n=== Tavily Response Structure ===")
    print(f"Type: {type(result)}")

    # Handle both dict and object response
    if isinstance(result, dict):
        print(f"Keys: {list(result.keys())}")
        print(f"\nFull response:")
        import json
        print(json.dumps(result, indent=2, default=str)[:2000])
    else:
        print(f"Attributes: {[a for a in dir(result) if not a.startswith('_')]}")

        # Check for answer
        if hasattr(result, 'answer'):
            print(f"\nAnswer attribute exists: {result.answer}")

        # Check for results
        if hasattr(result, 'results'):
            print(f"\nResults count: {len(result.results)}")
            for i, r in enumerate(result.results, 1):
                print(f"\n--- Result {i} ---")
                print(f"Type: {type(r)}")
                print(f"Attributes: {[a for a in dir(r) if not a.startswith('_')]}")
                if hasattr(r, 'title'):
                    print(f"Title: {r.title}")
                if hasattr(r, 'url'):
                    print(f"URL: {r.url}")
                if hasattr(r, 'content'):
                    print(f"Content: {r.content[:100]}...")


def test_format_tavily_result_with_mock():
    """Test the format_tavily_result helper with mocked response"""
    # Create mock response object
    mock_result = MagicMock()

    # Mock answer
    mock_result.answer = "The sky is blue due to Rayleigh scattering."

    # Mock results
    mock_result1 = MagicMock()
    mock_result1.title = "Why is the Sky Blue?"
    mock_result1.url = "https://example.com/sky-blue"
    mock_result1.content = "The sky appears blue because of Rayleigh scattering of sunlight."

    mock_result2 = MagicMock()
    mock_result2.title = "Atmospheric Scattering"
    mock_result2.url = "https://example.com/scattering"
    mock_result2.content = "Blue light is scattered more than other colors."

    mock_result.results = [mock_result1, mock_result2]

    # Format the result
    formatted = format_tavily_result(mock_result)

    print(f"\n=== Formatted Output ===")
    print(formatted)

    # Verify content
    assert "The sky is blue due to Rayleigh scattering" in formatted
    assert "Why is the Sky Blue?" in formatted
    assert "https://example.com/sky-blue" in formatted
    assert "Rayleigh scattering of sunlight" in formatted
    assert "Atmospheric Scattering" in formatted


def test_format_tavily_result_empty():
    """Test format_tavily_result with None or empty input"""
    result = format_tavily_result(None)
    assert result == "", "Should return empty string for None input"

    result = format_tavily_result(MagicMock(answer=None, results=[]))
    assert result == "", "Should return empty string for empty results"

    result = format_tavily_result({})
    assert result == "", "Should return empty string for empty dict"

    result = format_tavily_result({"answer": None, "results": []})
    assert result == "", "Should return empty string for dict with no answer or results"


def test_format_tavily_result_dict_not_blank():
    """Test format_tavily_result with dict response (real API format) - verify not blank"""
    # Test with results only (answer is None/null like real API)
    dict_response = {
        "query": "why sky is blue",
        "answer": None,  # Real API often returns null for answer
        "results": [
            {
                "title": "Why Is The Sky Blue?",
                "url": "https://example.com/sky-blue",
                "content": "The sky appears blue because of Rayleigh scattering of sunlight."
            },
            {
                "title": "Atmospheric Scattering",
                "url": "https://example.com/scattering",
                "content": "Blue light is scattered more than other colors."
            }
        ]
    }

    formatted = format_tavily_result(dict_response)

    print(f"\n=== Dict Format Test ===")
    print(f"Length: {len(formatted)}")
    print(f"Content preview: {formatted[:200]}...")

    # Verify not blank
    assert formatted != "", "format_tavily_result should not return blank string for valid dict input"
    assert len(formatted) > 0, "Formatted result should have content"
    assert "Why Is The Sky Blue?" in formatted, "Should contain title"
    assert "Rayleigh scattering" in formatted, "Should contain content"
    assert "https://example.com/sky-blue" in formatted, "Should contain URL"


def test_format_tavily_result_dict_with_answer():
    """Test format_tavily_result with dict response that has an answer"""
    dict_response = {
        "query": "capital of france",
        "answer": "The capital of France is Paris.",
        "results": [
            {
                "title": "Paris - Wikipedia",
                "url": "https://example.com/paris",
                "content": "Paris is the capital and most populous city of France."
            }
        ]
    }

    formatted = format_tavily_result(dict_response)

    print(f"\n=== Dict with Answer Test ===")
    print(f"Length: {len(formatted)}")
    print(f"Content: {formatted}")

    # Verify not blank and includes answer
    assert formatted != "", "Should not return blank for dict with answer"
    assert len(formatted) > 0, "Should have content"
    assert "The capital of France is Paris" in formatted, "Should include the answer"
    assert "Paris - Wikipedia" in formatted, "Should include result title"


def test_format_tavily_result_object_not_blank():
    """Test format_tavily_result with object response - verify not blank"""
    # Create mock object (backward compatibility)
    mock_result = MagicMock()
    mock_result.answer = "Python is a high-level programming language."

    mock_result1 = MagicMock()
    mock_result1.title = "What is Python?"
    mock_result1.url = "https://example.com/python"
    mock_result1.content = "Python is widely used for web development and data science."

    mock_result.results = [mock_result1]

    formatted = format_tavily_result(mock_result)

    print(f"\n=== Object Format Test ===")
    print(f"Length: {len(formatted)}")
    print(f"Content: {formatted}")

    # Verify not blank
    assert formatted != "", "format_tavily_result should not return blank string for valid object input"
    assert len(formatted) > 0, "Formatted result should have content"
    assert "Python is a high-level programming language" in formatted, "Should include answer"
    assert "What is Python?" in formatted, "Should include title"


def test_format_tavily_result_no_answer():
    """Test format_tavily_result when answer is missing but results exist"""
    dict_response = {
        "answer": None,
        "results": [
            {
                "title": "Test Result",
                "url": "https://example.com",
                "content": "This is test content."
            }
        ]
    }

    formatted = format_tavily_result(dict_response)

    print(f"\n=== No Answer Test ===")
    print(f"Length: {len(formatted)}")
    print(f"Content: {formatted}")

    # Verify not blank - should fall back to results
    assert formatted != "", "Should not be blank when answer is None but results exist"
    assert len(formatted) > 0, "Should have content from results"
    assert "Test Result" in formatted, "Should include result title"
    assert "This is test content" in formatted, "Should include result content"


@pytest.mark.asyncio
async def test_format_tavily_result_with_real_api():
    """Test format_tavily_result with real Tavily API call"""
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        pytest.skip("TAVILY_API_KEY not set")

    tavily = TavilyClient(api_key=api_key)
    search_result = tavily.search(query="what is python programming", max_results=2)

    formatted = format_tavily_result(search_result)

    print(f"\n=== Real API Formatted Output ===")
    print(formatted)

    # Verify we got some content
    assert len(formatted) > 0

    # Check if results exist (dict access)
    results = search_result.get('results', []) if isinstance(search_result, dict) else []
    if results:
        assert "python" in formatted.lower()


if __name__ == "__main__":
    # Run tests manually for debugging
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*60)
    print("Testing with mock data...")
    print("="*60)
    test_format_tavily_result_with_mock()

    print("\n" + "="*60)
    print("Testing empty results...")
    print("="*60)
    test_format_tavily_result_empty()

    print("\n" + "="*60)
    print("Testing with real Tavily API...")
    print("="*60)
    asyncio.run(test_tavily_direct_call())

    print("\n" + "="*60)
    print("Testing format function with real API...")
    print("="*60)
    asyncio.run(test_format_tavily_result_with_real_api())

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
