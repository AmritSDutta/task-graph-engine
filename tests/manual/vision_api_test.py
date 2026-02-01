import base64

from langchain.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

"""
 Vision Models and Their Providers:

     1. Model: gpt-4o
        - Provider: OpenAI
        - Cost: $5.0/1M tokens

     2. Model: qwen3-vl:235b-instruct-cloud
        - Provider: Qwen
        - Cost: $0.01/1M tokens

     3. Model: gpt-5-mini
        - Provider: OpenAI
        - Cost: $1.5/1M tokens

     4. Model: gpt-5-nano
        - Provider: OpenAI
        - Cost: $0.015/1M tokens

     5. Model: gemini-2.5-flash
        - Provider: Google
        - Cost: $0.08/1M tokens

     6. Model: gemini-2.5-pro
        - Provider: Google
        - Cost: $0.5/1M tokens

     7. Model: gemini-3-flash-preview:cloud
        - Provider: Google
        - Cost: $0.011/1M tokens

     8. Model: gemma3:27b-cloud
        - Provider: Google
        - Cost: $0.011/1M tokens

     9. Model: GLM-4.6V-Flash
        - Provider: Zhipu (identified from model name pattern)
        - Cost: $0.015/1M tokens

     10. Model: llama-3.3-70b-versatile
         - Provider: Meta
         - Cost: $0.012/1M tokens
"""


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# In your test:
def ask_ollama(model_name: str, base64_image: str):
    model = ChatOllama(model=model_name)
    system_msg = SystemMessage("You are a helpful assistant.")
    human_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the content of this image."},
            {
                "type": "image",
                "base64": base64_image,
                "mime_type": "image/jpeg",
            },
        ]
    }

    # Use with chat models
    messages = [system_msg, human_msg]
    response = model.invoke(messages)
    print(f"Response[{model_name}]: {response}")


def ask_OpenAI(model_name: str, base64_image: str):
    model = ChatOpenAI(model=model_name)
    system_msg = SystemMessage("You are a helpful assistant.")
    human_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the content of this image."},
            {
                "type": "image",
                "base64": base64_image,
                "mime_type": "image/jpeg",
            },
        ]
    }

    # Use with chat models
    messages = [system_msg, human_msg]
    response = model.invoke(messages)
    print(f"Response[{model_name}]: {response}")


def ask_Gemini(model_name: str, base64_image: str):
    model = ChatGoogleGenerativeAI(model=model_name)
    system_msg = SystemMessage("You are a helpful assistant.")
    human_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the content of this image."},
            {
                "type": "image",
                "base64": base64_image,
                "mime_type": "image/jpeg",
            },
        ]
    }

    # Use with chat models
    messages = [system_msg, human_msg]
    response = model.invoke(messages)
    print(f"Response[{model_name}]: {response}")


def main():
    """
    gemini-2.5-flash-lite, qwen3-vl:235b-instruct-cloud, gpt-5-mini,
    gpt-5-nano, gemini-2.5-flash, gemini-3-flash-preview:cloud, gemma3:27b-cloud



    """
    base64_image: str = encode_image(r"C:\Users\amrit\Downloads\Amrit Shankar Dutta - Data Visualization.png")
    ask_ollama(model_name="qwen3-vl:235b-instruct-cloud", base64_image=base64_image)


if __name__ == "__main__":
    main()
