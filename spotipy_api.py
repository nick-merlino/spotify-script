from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from spotipy.client import SpotifyException
from pprint import pprint
from tqdm import tqdm
import logging
from retrying import retry
from config import *

SCOPE_LIST = 'playlist-read-private playlist-modify-public playlist-modify-private'

# Configure the logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")
logging.getLogger('spotipy').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)

auth_manager = SpotifyOAuth(scope=SCOPE_LIST, client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET, redirect_uri=SPOTIFY_REDIRECT_URI, cache_path=".spotipy_auth.cache")
sp = Spotify(auth_manager=auth_manager)

# Handles rate limit policy below
#   Spotify's API rate limit is calculated based on the number of calls that your app makes to Spotify
#   in a rolling 30 second window. If your app exceeds the rate limit for your app then you'll begin to 
#   see 429 error responses from Spotify's Web API
def retry_on_rate_limit_error(exc):
    """Retry on Spotify API rate limit error (status code 429)."""
    return isinstance(exc, SpotifyException) and exc.http_status == 429

@retry(retry_on_exception=retry_on_rate_limit_error, wait_fixed=31000, stop_max_attempt_number=5)
def get_user_playlists():
    playlists = sp.current_user_playlists()
    playlists = [playlist for playlist in playlists['items'] if playlist['owner']['id'] == SPOTIFY_USER_ID]
    return playlists

@retry(retry_on_exception=retry_on_rate_limit_error, wait_fixed=31000, stop_max_attempt_number=5)
def get_category_name_to_ids():
    try:
        category_names_to_ids = {}
        categories_container = sp.categories(locale="en_US", country="US", limit=50)
        while categories_container:
            categories = categories_container.get("categories", {})
            for item in categories.get("items", []):
                category_names_to_ids[item["name"]] = item["id"]
            categories_container = sp.next(categories)

        return category_names_to_ids
    except SpotifyException as e:
        logging.info(f"Can't get categories - {e}")
        exit()

@retry(retry_on_exception=retry_on_rate_limit_error, wait_fixed=31000, stop_max_attempt_number=5)
def get_featured_playlist_ids():
    try:    
        # Limited to top 50 featured playlists because it's only returning 10 anyway
        playlists_container = sp.featured_playlists(locale="en_US", country="US", limit=50)
        playlists = playlists_container.get("playlists", {})
        return [x['id'] for x in playlists.get("items", [])]
    except SpotifyException as e:
        logging.info(f"Can't get featured playlists - {e}")

@retry(retry_on_exception=retry_on_rate_limit_error, wait_fixed=31000, stop_max_attempt_number=5)
def get_category_playlist_ids(category_id):
    try:
        # Get top 50 playlists per category. Could be increased but API would be hit more
        playlists_container = sp.category_playlists(category_id, limit=20)
        playlists = playlists_container.get("playlists", {})
        return [x['id'] for x in playlists.get("items", []) if x is not None]
    except SpotifyException as e:
        logging.info(f"Can't get category playlists - {e}")

@retry(retry_on_exception=retry_on_rate_limit_error, wait_fixed=31000, stop_max_attempt_number=5)
def get_playlist_tracks(playlist_id):
    try:
        tracks = []
        results = sp.user_playlist_tracks(SPOTIFY_USER_ID, playlist_id)
        while results:
            tracks.extend([item["track"] for item in results['items'] if item["track"] is not None])
            results = sp.next(results)
        return tracks
    except SpotifyException as e:
        logging.error(f"Can't get playlists tracks - {e}")
        exit()
    
@retry(retry_on_exception=retry_on_rate_limit_error, wait_fixed=31000, stop_max_attempt_number=5)
def get_artists(new_artist_ids):
    try:
        artist_id_to_name = {}
        for chunk in tqdm(divide_chunks_list(list(new_artist_ids), 50), total=len(new_artist_ids)//50, desc="Fetching Artists"):
            artists_info = sp.artists(chunk)
            for artist in artists_info["artists"]:
                if artist:
                    artist_id_to_name[artist["id"]] = artist["name"]
        return artist_id_to_name
    except SpotifyException as e:
        logging.error(f"Can't get artists - {e}")
        exit()

def divide_chunks_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

@retry(retry_on_exception=retry_on_rate_limit_error, wait_fixed=31000, stop_max_attempt_number=5)
def get_highest_popularity_track(track_name, artists):
    try:
        artist_names = " ".join(artists[:3])

        # Search for the track
        search_results = sp.search(q=f"track:{track_name} artist:{artist_names}", type="track", limit=10)

        # Sort the search results by popularity in descending order
        if search_results and "tracks" in search_results and "items" in search_results["tracks"]:
            search_results = sorted(search_results['tracks']['items'], key=lambda x: x.get('popularity', 0), reverse=True)

        # Return the most popular version of the track
        return search_results[0] if search_results else None
    except SpotifyException as e:
        logging.error(f"Can't get highest popularity track - {e}")
        exit()

@retry(retry_on_exception=retry_on_rate_limit_error, wait_fixed=31000, stop_max_attempt_number=5)
def get_track_audio_features(track_ids):
    try:
        track_id_to_attributes = {}
        for chunk in divide_chunks_list(list(track_ids), 100):
            features = {}
            for features in sp.audio_features(chunk):
                if features is not None:
                    track_id = features["id"]
                    track_id_to_attributes[track_id] = features
        return track_id_to_attributes
    except SpotifyException as e:
        logging.error(f"Can't get audio features for track - {e}")
        exit()
    

@retry(retry_on_exception=retry_on_rate_limit_error, wait_fixed=31000, stop_max_attempt_number=5)
def upload_playlist(playlist_id, playlist_name, new_track_id_list):
    try:
        backup_playlist = sp.user_playlist_create(SPOTIFY_USER_ID, f"{playlist_name} (Backup)", public=False, description="Backup in case things go bad")
        backup_playlist_id = backup_playlist["id"]
        
        # Add the sorted tracks to the new playlist in batches of 100 (Spotify API limit)
        for i in range(0, len(new_track_id_list), 100):
            sp.playlist_add_items(backup_playlist_id, new_track_id_list[i:i+100])

        # Extract the existing track ids
        existing_track_ids = [track["id"] for track in get_playlist_tracks(playlist_id)]

        # Remove the existing tracks from the playlist
        for chunk in divide_chunks_list(existing_track_ids, 100):
            sp.playlist_remove_all_occurrences_of_items(playlist_id, chunk)

        # Add the sorted tracks to the new playlist in batches of 100 (Spotify API limit)
        for i in range(0, len(new_track_id_list), 100):
            sp.playlist_add_items(playlist_id, new_track_id_list[i:i+100])

        # Delete the new playlist since it's no longer needed
        sp.user_playlist_unfollow(SPOTIFY_USER_ID, backup_playlist_id)

    except SpotifyException as e:
        logging.error(f"Can't update playlists - {e}")
        sp.user_playlist_unfollow(SPOTIFY_USER_ID, backup_playlist_id)
        exit()

def create_predicted_playlist(predicted_playlist_name, track_id_list, popular_on_spotify_id):
    try:
        predicted_playlist = sp.user_playlist_create(SPOTIFY_USER_ID, f"{predicted_playlist_name} (Predicted)", public=False, description="Predicted Playlist")
        predicted_playlist_id = predicted_playlist["id"]

        # Add the tracks to the new playlist in batches of 100 (Spotify API limit)
        for i in range(0, len(track_id_list), 100):
            sp.playlist_add_items(predicted_playlist_id, track_id_list[i:i+100])

        # Remove the existing tracks from the playlist
        for chunk in divide_chunks_list(track_id_list, 100):
            sp.playlist_remove_all_occurrences_of_items(popular_on_spotify_id, chunk)

    except SpotifyException as e:
        logging.error(f"Can't created predicted playlists - {e}")
        exit()