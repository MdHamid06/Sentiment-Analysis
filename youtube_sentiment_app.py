from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import requests
import re  # ✅ Added for extracting video ID

print("Starting YouTube Sentiment Analysis App...")

app = Flask(__name__)
print("Flask app created.")

 #Load the trained sentiment model and tokenizer
try:
    model = load_model("sentiment_model.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Prevent crashes if model fails to load

try:
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    tokenizer = None  # Prevent crashes if tokenizer fails to load

# Your YouTube API Key
YOUTUBE_API_KEY = "AIzaSyCNGvGqXv2v_mk1VCY8LVNnrgWLoezFgF4"

# Function to extract video ID from YouTube URL
def extract_video_id(youtube_url):
    """Extract the video ID from a YouTube URL."""
    match = re.search(r"(?<=v=)[^&]+", youtube_url)  # Works for standard URLs
    if not match:
        match = re.search(r"(?<=youtu.be/)[^?]+", youtube_url)  # Works for shortened URLs
    return match.group(0) if match else None

# Function to fetch YouTube comments
def get_youtube_comments(video_url, max_comments=500):
    """Fetch more YouTube comments by paginating results."""

    video_id = extract_video_id(video_url)
    if not video_id:
        return ["Invalid YouTube URL! Please enter a valid URL."]

    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    
    params = {
        "part": "snippet",
        "videoId": video_id,
        "key": YOUTUBE_API_KEY,
        "maxResults": 100,  # Fetch the max (100 per request)
        "textFormat": "plainText"
    }

    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        if next_page_token:
            params["pageToken"] = next_page_token  # Request next page of comments

        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"Error fetching comments: {response.json()}")
            break

        data = response.json()
        comments.extend(
            item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            for item in data.get("items", [])
        )

        next_page_token = data.get("nextPageToken")  #  Check for next page
        if not next_page_token:
            break  # Stop if no more pages

    return comments[:max_comments] if comments else ["No comments available."]

# Function to predict sentiment of a comment
def predict_sentiment(comment):
    if model is None or tokenizer is None:
        return "Unknown"  # Prevent crashes if model/tokenizer failed to load

    sequence = tokenizer.texts_to_sequences([comment])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)[0][0]  # Ensure correct indexing
    return "Positive" if prediction > 0.5 else "Negative"

#  Function to analyze overall sentiment of a YouTube video
def analyze_video_sentiment(video_url):
    comments = get_youtube_comments(video_url)

    if not comments or comments == ["No comments available."]:
        return {"error": "No comments found for this video!"}

    positive_count = sum(1 for comment in comments if predict_sentiment(comment) == "Positive")
    total_comments = len(comments)
    percentage = round((positive_count / total_comments) * 100, 2)

    overall_sentiment = "Positive" if percentage > 50 else "Negative"

    return {
        "total_comments": total_comments,
        "positive_percentage": percentage,
        "overall_sentiment": overall_sentiment,
        "sample_comments": comments[:5]  # Send 5 sample comments in response
    }

# Home Route
@app.route("/")
def home():
    print("Home route accessed.")
    return render_template("index.html")

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "video_url" not in data:
            return jsonify({"error": "No YouTube URL provided!"}), 400

        video_url = data["video_url"].strip()
        result = analyze_video_sentiment(video_url)

        return jsonify(result)
    except Exception as e:
        print(f"Error in /predict: {e}")  # ✅ Print the exact error
        return jsonify({"error": f"Something went wrong! {str(e)}"}), 500  # ✅ Send error message

# Run Flask App
if __name__ == "__main__":
    print("Starting YouTube Sentiment Analysis Server...")
    app.run(debug=True)
