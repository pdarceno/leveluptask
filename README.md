# CEO Chat Analyzer API

This Flask API analyzes chat logs to identify:
- Top discussed topics (keywords)
- Sentiment breakdown (positive, neutral, negative)
- Executive summary using OpenAI's GPT model

## Features

- Accepts a JSON file of chat messages via POST
- Extracts top 3 keywords (ignoring common stopwords)
- Performs sentiment analysis using TextBlob
- Generates a summary using OpenAI GPT-3.5 for CEO-style insights

