from __future__ import annotations

from webapp.api import finetune_analysis as fa


def test_tokenization_context_has_shared_contract():
    ctx = fa.tokenization_context()
    assert ctx["tokenizer_class"] == "DistilBertTokenizerFast"
    assert int(ctx["max_length"]) >= 32
    assert ctx["rows"]


def test_static_analysis_context_shape():
    ctx = fa.static_analysis_context()
    assert ctx["ticker"] == "AAPL"
    assert "tokenization" in ctx
    assert "hyperparams" in ctx
