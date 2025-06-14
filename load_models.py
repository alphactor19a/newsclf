import os
import random
from typing import Tuple

import torch
import requests

from news_clf import (
    PretrainedModelForOrdinalSequenceClassification,
    PretrainedModelForUnorderedSequenceClassification,
)
from news_clf.text_utils import shorten_to_n_words, format_prompt_with_article

# Paths to model checkpoints relative to this file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_3CLASS = os.path.join(ROOT_DIR, "3class_model_best_checkpoint.safetensors")
CHECKPOINT_ORDINAL = os.path.join(ROOT_DIR, "ordinal_model_best_checkpoint.safetensors")


def load_models(device_map: str = "auto") -> Tuple[
    PretrainedModelForUnorderedSequenceClassification,
    PretrainedModelForOrdinalSequenceClassification,
]:
    """Load both classification models from their checkpoints."""
    clf = PretrainedModelForUnorderedSequenceClassification(
        device_map=device_map, checkpoint=CHECKPOINT_3CLASS
    )
    ordinal = PretrainedModelForOrdinalSequenceClassification(
        device_map=device_map, checkpoint=CHECKPOINT_ORDINAL
    )
    return clf, ordinal


# Instantiate models lazily so import time stays small
_models = None


def get_models() -> Tuple[
    PretrainedModelForUnorderedSequenceClassification,
    PretrainedModelForOrdinalSequenceClassification,
]:
    global _models
    if _models is None:
        _models = load_models()
    return _models


def _prepare_inputs(tokenizer, title: str, body: str):
    """Tokenize an article with the same preprocessing as the notebooks."""
    prompt = format_prompt_with_article(title, body, max_words=1500)
    encoded = tokenizer(
        [prompt],
        padding="longest",
        truncation=True,
        max_length=1500,
        return_tensors="pt",
    )
    return encoded


def classify_article(title: str, body: str) -> dict:
    """Return framing label and ordinal score for a single article."""
    clf, ordinal = get_models()

    inputs = _prepare_inputs(clf.tokenizer, title, body)
    inputs = {k: v.to(clf.device) for k, v in inputs.items()}
    with torch.no_grad():
        out_clf = clf(**inputs)
    probs = out_clf.logits.softmax(dim=-1)
    pred_idx = int(probs.argmax(dim=-1).cpu()[0])
    framing_label = clf.class_labels[pred_idx]

    inputs_ord = _prepare_inputs(ordinal.tokenizer, title, body)
    inputs_ord = {k: v.to(ordinal.device) for k, v in inputs_ord.items()}
    with torch.no_grad():
        out_ord = ordinal(**inputs_ord)
    # Map the raw score to a 0-100 range using a sigmoid
    score_raw = out_ord.article_score.squeeze(0)
    intensity = float(torch.sigmoid(score_raw).cpu() * 100)

    return {"framing": framing_label, "score": intensity}


def fetch_random_article(topic: str) -> dict:
    """Retrieve a random recent article about ``topic`` using NewsAPI."""
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise RuntimeError("NEWSAPI_KEY environment variable is not set")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 20,
        "apiKey": api_key,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    articles = data.get("articles", [])
    if not articles:
        raise ValueError("No articles found for topic")
    article = random.choice(articles)
    body = article.get("content") or article.get("description") or ""
    return {"title": article.get("title", ""), "url": article.get("url", ""), "body": body}

