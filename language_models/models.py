from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import requests


class LanguageModel(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError


@dataclass
class GooseAILanguageModelSettings:
    engine_id: str = "gpt-neo-20b"
    temperature: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class GooseAILanguageModel(LanguageModel):
    def __init__(
        self,
        api_key: str,
        **kwargs,
    ) -> None:
        self.settings = GooseAILanguageModelSettings(**kwargs)
        self.api_key = api_key
        super().__init__()

    def completion_route(self, engine) -> str:
        return f"https://api.goose.ai/v1/engines/{engine}/completions"

    def complete(
        self,
        prompt: str,
        max_tokens: int,
        n: int = 1,
        stop: list = None,
        **kwargs: any,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": kwargs.get("temperature") or self.settings.temperature,
            "topP": kwargs.get("top_p") or self.settings.top_p,
            "stopSequences": stop if stop else [],
        }
        engine = kwargs.get("engine_id") or self.settings.engine_id
        route = self.completion_route(engine)
        response = requests.post(route, json=payload, headers=headers)
        completion = response.json()
        print(completion)
        completion_text = completion["choices"][0]["text"]
        return completion_text