from flask import Flask, request, jsonify
from collections import Counter
from textblob import TextBlob
from transformers import pipeline
import openai
import json
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Get API key from environment variable
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = {
        "the", "and", "is", "it", "in", "a", "to", "for", "of", "with",
        "on", "at", "but", "this", "that", "our", "we", "they", "i"
    }
    return [word for word in words if word not in stopwords]

def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

@app.route("/analyze", methods=["POST"])
def analyze_uploaded_file():
    if 'file' not in request.files:
        return jsonify({"error": "Missing file in request"}), 400

    file = request.files['file']
    try:
        messages = json.load(file)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON file: {str(e)}"}), 400

    all_keywords = []
    all_text = ""
    sentiment_analysis = {"positive": 0, "neutral": 0, "negative": 0}

    for msg in messages:
        text = msg["message"]
        all_text += " " + text
        all_keywords.extend(extract_keywords(text))
        sentiment = analyze_sentiment(text)
        sentiment_analysis[sentiment] += 1

    top_topics = [word for word, _ in Counter(all_keywords).most_common(3)]

    # Send all messages to OpenAI to generate CEO-style summary
    try:
        system_prompt = (
            "You are a business analyst. Summarize the employee chat logs below "
            "to generate a concise summary based on the messages, written in a format suitable for the CEO."
        )

        chat_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": all_text.strip()}
            ],
            max_tokens=200
        )

        summary = chat_response.choices[0].message.content.strip()
    except Exception as e:
        summary = f"Error generating summary with OpenAI: {str(e)}"

    return jsonify({
        "top_topics": top_topics,
        "summary": summary,
        "sentiment_analysis": sentiment_analysis
    })

if __name__ == "__main__":
    app.run(port=3000)
