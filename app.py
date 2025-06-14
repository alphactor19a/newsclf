from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from load_models import classify_article, fetch_random_article, get_models

app = FastAPI(title="News Framing API")

# Load models once at startup
get_models()

class TopicRequest(BaseModel):
    topic: str


@app.post("/get_article_framing")
def get_article_framing(data: TopicRequest):
    try:
        article = fetch_random_article(data.topic)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    result = classify_article(article["title"], article["body"])
    return {
        "title": article["title"],
        "url": article["url"],
        "framing_class": result["framing"],
        "emotional_score": result["score"],
    }

