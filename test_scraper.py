import requests

# Replace with your TMDb API key
TMDB_API_KEY = "9658ee8872194811b6ee038e1256e03a"

def get_movie_reviews(movie_name):
    """Fetch movie reviews from TMDb API instead of IMDB scraping."""
    
    # Step 1: Get the movie ID
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"
    response = requests.get(search_url)
    
    if response.status_code != 200:
        return {"error": "Failed to fetch movie data!"}

    movies = response.json().get("results", [])
    if not movies:
        return {"error": "Movie not found!"}

    movie_id = movies[0]["id"]  # Get the first matching movie ID

    # Step 2: Get movie reviews
    reviews_url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={TMDB_API_KEY}"
    response = requests.get(reviews_url)

    if response.status_code != 200:
        return {"error": "Failed to fetch reviews!"}

    reviews = response.json().get("results", [])

    if not reviews:
        return {"error": "No reviews found!"}

    return [review["content"] for review in reviews[:10]]  # Return top 10 reviews
