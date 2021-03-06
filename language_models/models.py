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
class AI21PenaltyData:
    scale: float = 0.0
    applyToWhitespaces: bool = True
    applyToPunctuations: bool = True
    applyToNumbers: bool = True
    applyToStopwords: bool = True
    applyToEmojis: bool = True


@dataclass
class AI21JurassicLanguageModelSettings:
    model_type: str = "j1-jumbo"
    temperature: float = 1.0
    top_p: float = 1.0
    presence_penalty: AI21PenaltyData = AI21PenaltyData()
    count_penalty: AI21PenaltyData = AI21PenaltyData()
    frequency_penalty: AI21PenaltyData = AI21PenaltyData()


class AI21JurassicLanguageModel(LanguageModel):
    def __init__(
        self,
        api_key: str,
        **kwargs,
    ) -> None:
        self.settings = AI21JurassicLanguageModelSettings(**kwargs)
        self.api_key = api_key
        super().__init__()

    def completion_route(self, model_type) -> str:
        return f"https://api.ai21.com/studio/v1/{model_type}/complete"

    def complete(
        self,
        prompt: str,
        max_tokens: int = 16,
        stop: list = None,
        logit_bias: dict[str, float] = None,
        **kwargs: any,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "maxTokens": max_tokens,
            "temperature": kwargs.get("temperature") or self.settings.temperature,
            "topP": kwargs.get("top_p") or self.settings.top_p,
            "stopSequences": stop if stop else [],
            "logitBias": logit_bias if logit_bias else {},
            "presencePenalty": (
                kwargs.get("presence_penalty") or self.settings.presence_penalty
            ).__dict__,
            "countPenalty": (
                kwargs.get("count_penalty") or self.settings.count_penalty
            ).__dict__,
            "frequencyPenalty": (
                kwargs.get("frequency_penalty") or self.settings.frequency_penalty
            ).__dict__,
        }
        model_type = kwargs.get("model_type") or self.settings.model_type
        route = self.completion_route(model_type)
        response = requests.post(route, json=payload, headers=headers)
        completion = response.json()
        completion_text = completion["completions"][0]["data"]["text"]
        return completion_text


@dataclass
class GooseAILanguageModelSettings:
    engine_id: str = "gpt-neo-20b"
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    tfs: float = 1.0
    top_a: float = 1.0
    typical_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    repetition_penalty_slope: float = 0.0
    repetition_penalty_range: int = 0


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
        max_tokens: int = 16,
        n: int = 1,
        min_tokens: int = 1,
        stop: list = None,
        logit_bias: dict[str, float] = None,
        **kwargs: any,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "n": n,
            "min_tokens": min_tokens,
            "stop": stop if stop else [],
            "logit_bias": logit_bias if logit_bias else {},
            "temperature": kwargs.get("temperature") or self.settings.temperature,
            "top_p": kwargs.get("top_p") or self.settings.top_p,
            "top_k": kwargs.get("top_k") or self.settings.top_k,
            "tfs": kwargs.get("tfs") or self.settings.tfs,
            "top_a": kwargs.get("top_a") or self.settings.top_a,
            "typical_p": kwargs.get("typical_p") or self.settings.typical_p,
            "frequency_penalty": kwargs.get("frequency_penalty")
            or self.settings.frequency_penalty,
            "presence_penalty": kwargs.get("presence_penalty")
            or self.settings.presence_penalty,
            "repetition_penalty": kwargs.get("repetition_penalty")
            or self.settings.repetition_penalty,
            "repetition_penalty_slope": kwargs.get("repetition_penalty_slope")
            or self.settings.repetition_penalty_slope,
            "repetition_penalty_range": kwargs.get("repetition_penalty_range")
            or self.settings.repetition_penalty_range,
        }
        engine = kwargs.get("engine_id") or self.settings.engine_id
        route = self.completion_route(engine)
        response = requests.post(route, json=payload, headers=headers)
        completion = response.json()
        completion_text = completion["choices"][0]["text"]
        return completion_text
