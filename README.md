# News Classifier API

This repository provides a simple FastAPI backend that loads two pretrained models
for news article framing classification. The models are shipped as
`safetensors` checkpoints in this repository.

## Running the API

1. Install the requirements
   ```bash
   pip install -r requirements.txt
   ```

2. Set a `NEWSAPI_KEY` environment variable so the API can fetch recent
   articles from [NewsAPI.org](https://newsapi.org/).

3. Start the server
   ```bash
   uvicorn app:app --reload
   ```

4. Send a POST request to `/get_article_framing` with JSON payload `{"topic": "economy"}`.
   The response will contain the article title, its URL, the predicted framing
   class and the emotional intensity score from the ordinal model.

