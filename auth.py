# auth.py
import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from flask import Flask, request

# Load credentials from .env
load_dotenv()
CLIENT_ID     = os.getenv("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI  = os.getenv("SPOTIPY_REDIRECT_URI")
SCOPE         = "user-modify-playback-state user-read-playback-state"

# Use a token cache file
CACHE_PATH = ".cache-spotify-token.json"

app = Flask(__name__)

sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
    cache_path=CACHE_PATH
)

@app.route("/login")
def login():
    auth_url = sp_oauth.get_authorize_url()
    return f'<a href="{auth_url}">Click here to authorize Spotify</a>'

@app.route("/callback")
def callback():
    code = request.args.get("code")
    token_info = sp_oauth.get_access_token(code, as_dict=True)
    return "✔️ Authorization successful! You can close this tab."

if __name__ == "__main__":
    print("Visit http://localhost:8888/login to authorize.")
    app.run(port=8888)
