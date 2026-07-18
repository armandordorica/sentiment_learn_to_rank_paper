from fastapi.testclient import TestClient

from webapp.api import paper_validation as pv
from webapp.main import app


def test_universe_has_expected_shape():
    universe = pv.load_universe()
    assert len(universe) == 1000
    assert universe["permno"].nunique() == 1000


def test_universe_checks_surface_expected_data_quality_failures():
    checks = {item["check"]: item["passed"] for item in pv.validation_checks(pv.load_universe())}
    assert len(checks) == 6
    assert checks["Exactly 1,000 candidate rows"] is True
    assert checks["Volume rank is unique"] is True
    assert checks["Only CRSP common-share codes 10/11"] is False
    assert checks["Only NYSE/AMEX/Nasdaq exchange codes"] is False


def test_page_context_has_top20_and_charts():
    ctx = pv.page_context()
    assert len(ctx["top20_rows"]) == 20
    assert "plotly" in ctx["top20_chart"].lower()
    assert "plotly" in ctx["volume_chart"].lower()


def test_paper_validation_route_renders():
    response = TestClient(app).get("/paper-validation")
    assert response.status_code == 200
    assert "7A Universe summary" in response.text


def test_price_partial_renders_and_rejects_unknown_ticker():
    client = TestClient(app)
    assert "plotly" in client.get("/paper-validation/prices?ticker=C").text.lower()
    assert "No monthly price data" in client.get("/paper-validation/prices?ticker=NOPE").text
