import requests

COLLECT_API_KEY = "3DKrj6HmKvKVFqFczOmuzz:2Y2Yj6JFDfdYkHRhCHzLJg"

url = "https://api.collectapi.com/imdb/imdbSearchByName?query=Inception"
headers = {
    "authorization": "apikey " + COLLECT_API_KEY,  
    "content-type": "application/json"
}

response = requests.get(url, headers=headers)
print(response.json())  
