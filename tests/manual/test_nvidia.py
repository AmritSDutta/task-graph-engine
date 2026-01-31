"""
Simple NVIDIA AI Endpoints test - just print response.
"""
import asyncio

from langchain_nvidia_ai_endpoints import ChatNVIDIA


async def simple_test():
    """Simple test to print LLM response."""
    print("Creating client...", flush=True)
    client = ChatNVIDIA(
        model="mistralai/mixtral-8x7b-instruct-v0.1",  # mistralai/mistral-large-3-675b-instruct-2512
        temperature=0.7,
    )

    print("Sending request...", flush=True)
    result = await client.ainvoke("What is the capital of France? Keep it brief.")
    print(f"\nResponse:\n{result.content}", flush=True)


if __name__ == "__main__":
    asyncio.run(simple_test())
