from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import httpx

from mtci.config import EndpointModelConfig, LocalModelConfig


class ModelError(Exception):
    pass


class BaseModelAdapter:
    def predict(self, xs: Sequence[str]) -> list[float]:
        raise NotImplementedError


@dataclass
class LocalModelAdapter(BaseModelAdapter):
    model: Any

    @classmethod
    def from_config(cls, config: LocalModelConfig) -> "LocalModelAdapter":
        model = load_entrypoint(config.entrypoint, config.kwargs)
        return cls(model=model)

    def predict(self, xs: Sequence[str]) -> list[float]:
        if hasattr(self.model, "predict"):
            return list(self.model.predict(xs))
        if callable(self.model):
            return list(self.model(xs))
        raise ModelError("Local model is not callable and has no predict method")


@dataclass
class HTTPEndpointModel(BaseModelAdapter):
    base_url: str
    predict_path: str
    timeout_s: float
    transport: httpx.BaseTransport | None = None

    @classmethod
    def from_config(cls, config: EndpointModelConfig) -> "HTTPEndpointModel":
        return cls(config.base_url, config.predict_path, config.timeout_s)

    def _run_async(self, coro: "asyncio.Future[Any]") -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError("Async transport requires an async call path")

    async def _async_post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_s, transport=self.transport) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    async def _async_post_raw(
        self, url: str, raw_body: str, headers: dict[str, str]
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_s, transport=self.transport) as client:
            response = await client.post(url, content=raw_body, headers=headers)
            response.raise_for_status()
            return response.json()

    def _post_json(self, payload: dict[str, Any]) -> list[float]:
        url = f"{self.base_url.rstrip('/')}{self.predict_path}"
        if self.transport is not None and hasattr(self.transport, "__aenter__"):
            data = self._run_async(self._async_post(url, payload))
        else:
            with httpx.Client(timeout=self.timeout_s, transport=self.transport) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        if "scores" not in data or not isinstance(data["scores"], Iterable):
            raise ModelError("Endpoint response missing 'scores' list")
        return [float(x) for x in data["scores"]]

    def post_raw(self, raw_body: str, headers: dict[str, str] | None = None) -> list[float]:
        url = f"{self.base_url.rstrip('/')}{self.predict_path}"
        headers = headers or {"content-type": "application/json"}
        if self.transport is not None and hasattr(self.transport, "__aenter__"):
            data = self._run_async(self._async_post_raw(url, raw_body, headers))
        else:
            with httpx.Client(timeout=self.timeout_s, transport=self.transport) as client:
                response = client.post(url, content=raw_body, headers=headers)
                response.raise_for_status()
                data = response.json()
        if "scores" not in data or not isinstance(data["scores"], Iterable):
            raise ModelError("Endpoint response missing 'scores' list")
        return [float(x) for x in data["scores"]]

    def predict(self, xs: Sequence[str]) -> list[float]:
        return self._post_json({"inputs": list(xs)})


def load_entrypoint(entrypoint: str, kwargs: dict[str, Any] | None = None) -> Any:
    kwargs = kwargs or {}
    if ":" in entrypoint:
        module_name, attr_name = entrypoint.split(":", 1)
    else:
        module_name, attr_name = entrypoint.rsplit(".", 1)
    module = importlib.import_module(module_name)
    target = getattr(module, attr_name)
    if callable(target):
        try:
            return target(**kwargs)
        except TypeError:
            return target
    raise ModelError(f"Entrypoint {entrypoint} is not callable")


def build_adapter(config: LocalModelConfig | EndpointModelConfig) -> BaseModelAdapter:
    if isinstance(config, LocalModelConfig):
        return LocalModelAdapter.from_config(config)
    return HTTPEndpointModel.from_config(config)
