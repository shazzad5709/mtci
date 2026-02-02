from __future__ import annotations

import httpx

from mtci.adapters import HTTPEndpointModel
from mtci.mrs.idempotence import IdempotenceMR
from mtci.server import create_app
from mtci.config import Tolerance


def test_endpoint_mr_with_fastapi(monkeypatch):
    monkeypatch.setenv("MTCI_LIGHT_MODEL", "1")
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    model = HTTPEndpointModel(
        base_url="http://test",
        predict_path="/predict",
        timeout_s=5.0,
        transport=transport,
    )
    mr = IdempotenceMR()
    result = mr.run(model, ["good", "bad"], max_examples=2, tolerance=Tolerance())
    assert result.passed
