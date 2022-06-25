import os

from dotenv import load_dotenv

from language_models.models import GooseAILanguageModel

load_dotenv()


def test_goose_ai():
    api_key = os.environ.get("GOOSEAI_API_KEY")
    client = GooseAILanguageModel(api_key=api_key)
    prompt = "hello"
    max_tokens = 10
    completion = client.complete(prompt, max_tokens=max_tokens)
    assert completion
