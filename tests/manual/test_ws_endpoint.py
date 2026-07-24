# test_websocket.py
import asyncio
import json
import logging
import websockets

# Configure logging to see output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_websocket():
    uri = "ws://localhost:2024/ws/custom"  # langgraph dev runs on 2024 by default

    async with websockets.connect(uri) as websocket:
        # Send a message to your graph
        message = {
            "content": "What is LangGraph?"
        }

        logger.info(f"Sending: {message}")
        await websocket.send(json.dumps(message))

        # Receive streaming responses
        message_count = 0
        final_report = None

        while True:
            try:
                response = await websocket.recv()
                data = json.loads(response)
                message_count += 1

                # Handle different event types
                event_type = data.get("type", "unknown")

                if event_type == "update":
                    event_data = data.get("data", {})
                    node_name = list(event_data.keys())[0] if event_data else "unknown"

                    logger.info(f"[{message_count}] UPDATE - Node: {node_name}")

                    # For 'end' node, extract and display the final report
                    if node_name == "end":
                        if "final_report" in event_data.get(node_name, {}):
                            final_report = event_data[node_name]["final_report"]
                            logger.info("=" * 80)
                            logger.info("FINAL REPORT:")
                            logger.info("=" * 80)
                            logger.info(final_report)
                            logger.info("=" * 80)
                        else:
                            # Show full data for end node if no final_report
                            logger.info(f"    Data: {json.dumps(event_data, indent=2)}")

                    # For other nodes, show keys to keep output clean
                    elif node_name in ["planner", "combiner"]:
                        logger.info(f"    Keys: {list(event_data[node_name].keys())}")

                    # For subtask nodes, show a brief summary
                    elif node_name == "subtask":
                        subtask_data = event_data.get(node_name, {})
                        todo_name = subtask_data.get("todo", {}).get("todo_name", "N/A")
                        todo_completed = subtask_data.get("todo", {}).get("todo_completed", False)
                        logger.info(f"    Task: {todo_name} | Completed: {todo_completed}")

                elif event_type == "message":
                    logger.info(f"[{message_count}] MESSAGE")
                    content = data.get("content", "")
                    # Truncate long content
                    if len(content) > 200:
                        content = content[:200] + "..."
                    logger.info(f"    Content: {content}")

                elif event_type == "done":
                    logger.info(f"[{message_count}] DONE - {data.get('message')}")
                    logger.info(f"Thread: {data.get('thread_id')}")
                    # Break after receiving done signal
                    break

                elif event_type == "error":
                    logger.error(f"[{message_count}] ERROR - {data.get('message')}")
                    break

                else:
                    logger.info(f"[{message_count}] UNKNOWN: {response}")

            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed by server")
                break
            except json.JSONDecodeError:
                logger.info(f"[{message_count}] Raw (non-JSON): {response}")

        logger.info(f"Total messages received: {message_count}")

        # Show final report at the end
        if final_report:
            print("\n" + "=" * 80)
            print("📋 FINAL RESPONSE:")
            print("=" * 80)
            print(final_report)
            print("=" * 80)
        else:
            print("\n⚠️  No final report received")


asyncio.run(test_websocket())
