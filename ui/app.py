import time
import json
import requests
import streamlit as st

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
DEPLOYMENT_URL = "http://localhost:2024"
ASSISTANT_ID = "agent"


# -------------------------------------------------------------------
# REST helpers
# -------------------------------------------------------------------
def create_thread() -> str:
    resp = requests.post(
        f"{DEPLOYMENT_URL}/threads",
        json={},  # required, otherwise 422
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    return data["thread_id"]


def submit_run(thread_id: str, input_data: dict) -> str:
    payload = {
        "assistant_id": ASSISTANT_ID,
        "input": input_data,
        "stream_mode": "updates",
    }
    resp = requests.post(
        f"{DEPLOYMENT_URL}/threads/{thread_id}/runs",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    return data["run_id"]


def get_run_state(thread_id: str, run_id: str) -> dict:
    resp = requests.get(
        f"{DEPLOYMENT_URL}/threads/{thread_id}/runs/{run_id}",
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()


def get_thread_state(thread_id: str) -> dict:
    """Fetch thread; this is where state/final_report lives in your setup."""
    resp = requests.get(
        f"{DEPLOYMENT_URL}/threads/{thread_id}",
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()


def get_models() -> dict:
    """Get available models with capabilities."""
    try:
        resp = requests.get(
            f"{DEPLOYMENT_URL}/api/models",
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code == 200:
            return resp.json()
        return {}
    except Exception:
        return {}


def _find_key_recursive(obj, key: str):
    """DFS search for key in nested dict/list structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                return v
            found = _find_key_recursive(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_key_recursive(item, key)
            if found is not None:
                return found
    return None


def extract_final_report_from_thread(thread_state: dict) -> str:
    """
    Try to extract `final_report` from the thread object.
    Your State has final_report: CombinedPlan, but it's persisted by the app,
    so we search the whole thread JSON.
    """
    final = _find_key_recursive(thread_state, "final_report")
    if final is None:
        # fallback: show entire thread state
        final = thread_state

    try:
        return "```json\n" + json.dumps(final, indent=2, default=str) + "\n```"
    except TypeError:
        return str(final)


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Task Graph Engine", layout="wide")

st.title("🤖 Task Graph Engine")
st.markdown(
    """
    AI task planning system with intelligent LLM selection.
    Supports various task types including coding, research, and analysis.
    """
)

# Sidebar with system info
with st.sidebar:
    st.header("ℹ️ System Info")

    # Fetch and display available models
    with st.spinner("Loading models..."):
        models_data = get_models()

    if models_data:
        st.success(f"✅ {models_data.get('count', 0)} models loaded")

        # Show coding models (most commonly used)
        coding_models = [
            name for name, details in models_data.get("models", {}).items()
            if "coding" in details.get("capabilities", set())
        ]

        if coding_models:
            st.subheader("💻 Coding Models")
            for model in coding_models[:5]:  # Show first 5
                caps = models_data["models"][model].get("capabilities", set())
                cost = models_data["models"][model].get("cost", 0)
                st.caption(f"• **{model}**")
                st.caption(f"  Cost: ${cost:.3f} | Caps: {', '.join(sorted(caps))}")
    else:
        st.warning("⚠️ Could not load model info")

    st.divider()
    if st.button("🔄 New Thread"):
        st.session_state.thread_id = None
        st.session_state.last_output = ""
        st.session_state.last_issue = ""
        st.rerun()

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "last_output" not in st.session_state:
    st.session_state.last_output = ""
if "last_issue" not in st.session_state:
    st.session_state.last_issue = ""

# Main input area
st.subheader("💬 Your Message")
issue_text = st.text_area(
    "Describe your task or ask a question",
    height=150,
    placeholder="Examples:\n• 'Write a Python function to sort a list'\n• 'Explain quantum computing in simple terms'\n• 'Why is the gold price surging? Generate a brief report'\n• 'Debug this code: [paste your code]'",
)

# Run button
run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

if run_button and issue_text.strip():
    # Create thread if needed
    if st.session_state.thread_id is None:
        with st.spinner("Creating thread..."):
            try:
                st.session_state.thread_id = create_thread()
                st.info(f"📝 Thread created: {st.session_state.thread_id}")
            except Exception as e:
                st.error(f"Error creating thread: {e}")
                st.stop()

    thread_id = st.session_state.thread_id

    # Build input payload (text-only)
    input_payload = {
        "messages": [
            {
                "role": "user",
                "content": issue_text
            }
        ],
    }

    # Submit run
    try:
        with st.spinner("Submitting task..."):
            run_id = submit_run(thread_id, input_payload)
            st.info(f"🔄 Run ID: {run_id}")
    except Exception as e:
        st.error(f"Error submitting run: {e}")
        st.stop()

    st.session_state.last_issue = issue_text

    # Poll for results
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    step = 0
    run_state = None

    with st.spinner("Processing your request..."):
        while True:
            try:
                run_state = get_run_state(thread_id, run_id)
            except Exception as e:
                st.error(f"Error polling run: {e}")
                break

            status = run_state.get("status", "unknown")
            step += 1

            # Update status display
            status_placeholder.markdown(f"**Status:** `{status}` · Poll #{step}")

            # Show progress bar for visual feedback
            progress_bar = progress_placeholder.progress(0)
            if status == "pending":
                progress_bar.progress(20)
            elif status == "running":
                progress_bar.progress(min(20 + step * 5, 90))
            elif status in ("success", "failed", "error", "cancelled"):
                progress_bar.progress(100 if status == "success" else 0)

            if status in ("success", "failed", "error", "cancelled"):
                break

            time.sleep(1)

    status_placeholder.empty()
    progress_placeholder.empty()

    # Display results
    if run_state is not None:
        status = run_state.get("status", "unknown")

        if status == "success":
            st.success("✅ Run completed successfully!")

            try:
                thread_state = get_thread_state(thread_id)
                final_report_md = extract_final_report_from_thread(thread_state)
            except Exception as e:
                st.error(f"Error fetching thread state: {e}")
                final_report_md = "No final_report found."

            st.session_state.last_output = final_report_md

            # Show execution details in expander
            with st.expander("🔍 Execution Details"):
                st.json(run_state)

        else:
            st.error(f"❌ Run finished with status: `{status}`")

            with st.expander("Error Details"):
                st.code(json.dumps(run_state, indent=2))


# Display final output
if st.session_state.last_output:
    st.divider()
    st.markdown("### 📊 Final Report")

    # Try to render as markdown if it's a simple string
    output = st.session_state.last_output
    if output.startswith("```json"):
        # It's JSON - display as code
        st.code(output, language="json")
    else:
        # It's markdown or plain text
        st.markdown(output)

    # Show what was submitted
    if st.session_state.last_issue:
        with st.expander("📝 Your Original Request"):
            st.write(st.session_state.last_issue)


# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
        Task Graph Engine v1.0 • Powered by LangGraph • <a href='http://127.0.0.1:2024/docs'>API Docs</a>
    </div>
    """,
    unsafe_allow_html=True
)
