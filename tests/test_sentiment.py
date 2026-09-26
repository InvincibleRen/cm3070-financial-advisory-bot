"""Offline tests for Extension A (FinBERT news sentiment).

No model download and no network: an ``InMemoryNewsSource`` supplies headlines
and a deterministic fake classifier stands in for the FinBERT pipeline, so the
leakage window, the signed-score aggregation and the feature assembly are all
verified reproducibly.
"""
import numpy as np
import pandas as pd

from src.selection import features as F
from src.sentiment.finbert import FinBERTScorer
from src.sentiment.news_source import Headline, InMemoryNewsSource



def _fake_classifier(texts):
    out = []
    for text in texts:
        t = text.lower()
        if any(w in t for w in ("surge", "beat", "record", "jumps")):
            out.append([{"label": "positive", "score": 0.9},
                        {"label": "negative", "score": 0.05},
                        {"label": "neutral", "score": 0.05}])
        elif any(w in t for w in ("plunge", "miss", "falls", "cuts")):
            out.append([{"label": "positive", "score": 0.05},
                        {"label": "negative", "score": 0.9},
                        {"label": "neutral", "score": 0.05}])
        else:
            out.append([{"label": "positive", "score": 0.1},
                        {"label": "negative", "score": 0.1},
                        {"label": "neutral", "score": 0.8}])
    return out


def _scorer():
    return FinBERTScorer(classifier=_fake_classifier)



def test_headline_url_defaults_empty():
    h = Headline("AAPL", pd.Timestamp("2020-01-01"), "t", "x")
    assert h.url == ""


def test_parse_news_item_extracts_url_both_schemas():
    from src.sentiment.news_source import _parse_yf_news_item

    old = {"title": "T", "providerPublishTime": 1590000000, "link": "https://old/a"}
    text, ts, url = _parse_yf_news_item(old)
    assert text == "T" and ts is not None and url == "https://old/a"

    new = {"content": {"title": "N", "pubDate": "2020-05-20T00:00:00Z",
                       "canonicalUrl": {"url": "https://new/b"}}}
    text2, ts2, url2 = _parse_yf_news_item(new)
    assert text2 == "N" and ts2 is not None and url2 == "https://new/b"


def test_news_source_respects_asof_and_lookback():
    heads = [
        Headline("AAPL", pd.Timestamp("2020-05-28"), "Apple shares surge", "x"),
        Headline("AAPL", pd.Timestamp("2020-05-20"), "old news", "x"),
        Headline("AAPL", pd.Timestamp("2020-06-10"), "future news", "x"),
        Headline("MSFT", pd.Timestamp("2020-05-30"), "other ticker", "x"),
    ]
    source = InMemoryNewsSource(heads)
    got = source.fetch_headlines("AAPL", pd.Timestamp("2020-06-01"), lookback_days=7)
    assert [h.text for h in got] == ["Apple shares surge"]



def test_score_texts_signs_and_aggregates():
    scorer = _scorer()
    assert scorer.score_texts(["NVDA shares surge to record"]) > 0.8
    assert scorer.score_texts(["Company misses and cuts guidance"]) < -0.8
    assert abs(scorer.score_texts(["a routine filing"])) < 0.01
    assert abs(scorer.score_texts(["surge", "plunge"])) < 0.01


def test_empty_batch_is_neutral():
    assert _scorer().score_texts([]) == 0.0
    assert _scorer().score_texts(["", "   "]) == 0.0



def test_build_sentiment_feature_shape_and_values():
    heads = [
        Headline("AAPL", pd.Timestamp("2020-05-30"), "Apple beats and jumps", "x"),
        Headline("MSFT", pd.Timestamp("2020-05-30"), "Microsoft misses, falls", "x"),
    ]
    source = InMemoryNewsSource(heads)
    t = pd.Timestamp("2020-06-01")

    feat = _scorer().build_sentiment_feature(source, ["AAPL", "MSFT", "TSLA"], [t], lookback_days=7)

    assert list(feat.columns) == ["sentiment"]
    assert list(feat.index.names) == ["rebalance_date", "ticker"]
    assert feat.loc[(t, "AAPL"), "sentiment"] > 0.8
    assert feat.loc[(t, "MSFT"), "sentiment"] < -0.8
    assert (t, "TSLA") not in feat.index


def test_future_headline_does_not_leak_into_earlier_date():
    heads = [
        Headline("AAPL", pd.Timestamp("2020-05-30"), "Apple beats and jumps", "x"),
        Headline("AAPL", pd.Timestamp("2020-06-20"), "Apple plunge and cuts", "x"),
    ]
    source = InMemoryNewsSource(heads)
    t = pd.Timestamp("2020-06-01")
    feat = _scorer().build_sentiment_feature(source, ["AAPL"], [t], lookback_days=7)
    assert feat.loc[(t, "AAPL"), "sentiment"] > 0.8



def test_sentiment_merges_into_core_feature_matrix():
    prices = {"AAPL": pd.DataFrame(
        {"Close": [100.0] * 200},
        index=pd.date_range("2019-06-01", periods=200, freq="D"),
    )}
    t = pd.Timestamp("2019-12-01")

    sentiment = _scorer().build_sentiment_feature(
        InMemoryNewsSource([Headline("AAPL", pd.Timestamp("2019-11-28"), "AAPL jumps to record", "x")]),
        ["AAPL"], [t], lookback_days=7,
    )
    matrix = F.build_feature_matrix(prices, pd.DataFrame(), [t], sentiment=sentiment)

    assert "sentiment" in matrix.columns
    assert matrix.loc[(t, "AAPL"), "sentiment"] > 0.8
