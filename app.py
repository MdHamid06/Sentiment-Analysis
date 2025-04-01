from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import requests
import re
import os
from dotenv import load_dotenv
import hashlib

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
CONFIG = {
    "YOUTUBE_API_KEY": os.getenv("YOUTUBE_API_KEY", ""),
    "COLLECT_API_KEY": os.getenv("COLLECT_API_KEY", ""),
    "CACHE_ENABLED": os.getenv("CACHE_ENABLED", "false").lower() == "true"
}

# Load model and tokenizer only once
try:
    model = load_model("sentiment_model.h5")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    tokenizer = None

def extract_video_id(youtube_url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r"(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^?]+)",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^?]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    return None

def predict_sentiment(text):
    """Predict sentiment of a single text"""
    if model is None or tokenizer is None:
        return "Unknown"
    
    try:
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=200)
        prediction = model.predict(padded_sequence, verbose=0)[0][0]
        return "Positive" if prediction > 0.5 else "Negative"
    except Exception as e:
        print(f"Error in sentiment prediction: {e}")
        return "Unknown"

def get_unique_cache_key(platform, input_value):
    """Generate unique cache key based on platform and input"""
    key_str = f"{platform}:{input_value}"
    return hashlib.md5(key_str.encode()).hexdigest()

def analyze_sentiments(texts):
    """Analyze sentiment for a list of texts with proper uniqueness"""
    if not texts or not isinstance(texts, list):
        return {
            "error": "No valid texts provided for analysis",
            "total_reviews": 0,
            "positive_percentage": 0,
            "overall_sentiment": "Unknown"
        }
    
    positive_count = 0
    unique_texts = list(set(texts))  # Remove duplicates
    
    for text in unique_texts:
        if predict_sentiment(text) == "Positive":
            positive_count += 1
    
    total = len(unique_texts)
    percentage = round((positive_count / total) * 100, 2) if total > 0 else 0
    overall = "Positive" if percentage > 50 else "Negative"
    
    return {
        "total_reviews": total,
        "positive_percentage": percentage,
        "overall_sentiment": overall,
        "positive_count": positive_count,
        "negative_count": total - positive_count
    }

def get_youtube_comments(video_id, max_comments=100):
    """Fetch unique YouTube comments for a specific video"""
    if not CONFIG["YOUTUBE_API_KEY"]:
        return {"error": "YouTube API not configured"}
    
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "key": CONFIG["YOUTUBE_API_KEY"],
        "maxResults": min(max_comments, 100),
        "textFormat": "plainText",
        "order": "relevance"  # Get most relevant comments
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            print(f"YouTube API error: {response.json()}")
            return {"error": "Failed to fetch comments"}
        
        items = response.json().get("items", [])
        comments = [
            item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            for item in items
        ]
        
        # Ensure we have unique comments
        unique_comments = list(set(comments))
        return unique_comments if unique_comments else ["No comments available."]
    except Exception as e:
        print(f"Error fetching YouTube comments: {e}")
        return {"error": "Error fetching comments"}

def get_imdb_reviews(movie_name):
    """Fetch unique IMDb reviews for a specific movie"""
    if not CONFIG["COLLECT_API_KEY"]:
        return {"error": "CollectAPI not configured"}
    
    url = f"https://api.collectapi.com/imdb/imdbSearchByName?query={movie_name}"
    headers = {
        "authorization": f"apikey {CONFIG['COLLECT_API_KEY']}",
        "content-type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"IMDb API error: {response.json()}")
            return {"error": "Failed to fetch reviews"}
        
        data = response.json()
        reviews = [
            movie["imdbContent"] 
            for movie in data.get("result", []) 
            if "imdbContent" in movie
        ]
        
        # Ensure unique reviews
        unique_reviews = list(set(reviews))
        return unique_reviews if unique_reviews else ["No reviews available."]
    except Exception as e:
        print(f"Error fetching IMDb reviews: {e}")
        return {"error": "Error fetching reviews"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        platform = data.get("platform", "").lower()
        input_value = data.get("input", "").strip()
        
        if not platform or not input_value:
            return jsonify({"error": "Missing platform or input"}), 400
        
        if platform == "youtube":
            video_id = extract_video_id(input_value)
            if not video_id:
                return jsonify({"error": "Invalid YouTube URL"}), 400
            
            comments = get_youtube_comments(video_id)
            if "error" in comments:
                return jsonify(comments), 400
                
            results = analyze_sentiments(comments)
            results["sample_comments"] = comments[:5] if len(comments) > 5 else comments
            results["platform"] = "youtube"
            
        elif platform == "imdb":
            reviews = get_imdb_reviews(input_value)
            if "error" in reviews:
                return jsonify(reviews), 400
                
            results = analyze_sentiments(reviews)
            results["sample_reviews"] = reviews[:5] if len(reviews) > 5 else reviews
            results["platform"] = "imdb"
            
        else:
            return jsonify({"error": "Unsupported platform"}), 400
        
        # Add unique identifier to results
        results["analysis_id"] = get_unique_cache_key(platform, input_value)
        return jsonify(results)
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return jsonify({"error": "Analysis failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)