"""Test script to verify the agent executes with external prompts."""
import asyncio
from langchain_core.messages import HumanMessage
from task_agent.graph import graph

async def test_agent():
    """Test the agent with a simple query."""
    print("=" * 60)
    print("Testing Agent with External Prompts")
    print("=" * 60)

    # Create a simple test input
    initial_state = {
        "messages": [HumanMessage(content="what is 2+2?")]
    }

    print(f"\nInput: {initial_state['messages'][0].content}")

    # Run the graph
    print("\n--- Starting execution ---\n")
    result = await graph.ainvoke(initial_state)

    print("\n--- Execution completed ---\n")
    print(f"Final message: {result['messages'][-1].content[:200]}...")

    # Check for final_report which indicates successful completion
    if "final_report" in result and result["final_report"]:
        print("\n" + "=" * 60)
        print("EXECUTION SUMMARY SUCCESS!")
        print("=" * 60)
        print(f"Final report generated: {len(result['final_report'])} chars")
        print("=" * 60)
    else:
        print("\nNo final_report in result")

    return result

if __name__ == "__main__":
    asyncio.run(test_agent())
