from __future__ import annotations

import httpx

from sentiment_ltr.data.refinitiv_session import patch_lseg_httpx_proxy_compat


def test_lseg_empty_proxy_map_is_accepted(monkeypatch):
    from lseg.data._core.session import http_service

    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(httpx, "Client", FakeClient)
    monkeypatch.delattr(
        http_service.get_httpx_client, "_sentiment_ltr_proxy_compat", raising=False
    )
    patch_lseg_httpx_proxy_compat()
    http_service.get_httpx_client({}, headers={"Accept": "application/json"})
    assert "proxy" not in captured
    assert captured["headers"] == {"Accept": "application/json"}


def test_lseg_single_proxy_map_becomes_httpx_proxy(monkeypatch):
    from lseg.data._core.session import http_service

    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(httpx, "Client", FakeClient)
    patch_lseg_httpx_proxy_compat()
    http_service.get_httpx_client({"https://": "http://proxy.example:8080"})
    assert captured["proxy"] == "http://proxy.example:8080"
