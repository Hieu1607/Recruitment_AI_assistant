# Placeholder for Groq / Ollama LLM provider wrapper
class LLMProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(self, prompt: str):
        return f"Generated response for: {prompt}"
