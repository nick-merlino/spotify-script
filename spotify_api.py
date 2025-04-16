import asyncio
import pickle
import logging
from typing import Any, Dict, List, Optional, Callable
import diskcache as dc
from spotipy import Spotify, SpotifyException
from spotipy.oauth2 import SpotifyOAuth
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import requests
from requests.adapters import HTTPAdapter

from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI, SPOTIFY_USER_ID, SCOPE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a custom requests session with a larger connection pool
session = requests.Session()
default_adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
accounts_adapter = HTTPAdapter(pool_connections=20, pool_maxsize=50)  # Larger pool for accounts.spotify.com
session.mount("http://", default_adapter)
session.mount("https://", default_adapter)
session.mount("https://accounts.spotify.com", accounts_adapter)

# Persistent cache using diskcache (default TTL 24 hours)
persistent_cache = dc.Cache("spotify_cache")

def wait_for_retry(retry_state) -> float:
    exc = retry_state.outcome.exception()
    if hasattr(exc, "response") and exc.response is not None:
        status_code = exc.response.status_code
        if status_code == 429:
            headers = exc.response.headers
            logger.debug(f"429 encountered; response headers: {headers}")
            retry_after = headers.get("Retry-After")
            if retry_after is not None:
                try:
                    wait_time = float(retry_after)
                    logger.debug(f"Using Retry-After header: waiting for {wait_time} seconds")
                    return wait_time
                except ValueError:
                    logger.debug("Retry-After header could not be converted to float")
        elif status_code == 502:
            logger.debug("502 encountered; waiting for 5 seconds before retrying")
            return 5
    fallback = wait_exponential(multiplier=2, min=10, max=120)(retry_state)
    logger.debug(f"No valid Retry-After header; falling back to waiting {fallback} seconds")
    return fallback

def persistent_async_cache(ttl: int = 86400) -> Callable:
    """
    Decorator for persistent async caching.
    The key is based on function name and pickled arguments.
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(self, *args, **kwargs):
            key = pickle.dumps((func.__name__, args, tuple(sorted(kwargs.items()))))
            if key in persistent_cache:
                logger.debug(f"Persistent cache hit for {func.__name__}")
                return persistent_cache[key]
            logger.debug(f"Persistent cache miss for {func.__name__}")
            result = await func(self, *args, **kwargs)
            persistent_cache.set(key, result, expire=ttl)
            return result
        return wrapper
    return decorator

class SpotifyAPI:
    def __init__(self) -> None:
        self.sp = Spotify(auth_manager=SpotifyOAuth(
            scope=SCOPE,
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET,
            redirect_uri=SPOTIFY_REDIRECT_URI,
            open_browser=False,
            cache_path=".cache-spotify"
        ), requests_session=session)
        # In-memory per-item caches for list-based endpoints
        self._audio_features_cache: Dict[str, Dict[str, Any]] = {}
        self._artists_cache: Dict[str, str] = {}
        # Global concurrency limit for API calls
        self._semaphore = asyncio.Semaphore(5)

    def _safe_api_call(self, func: Callable, *args, **kwargs):
        @retry(
            retry=retry_if_exception_type(SpotifyException),
            wait=wait_for_retry,
            stop=stop_after_attempt(10),  # increased from 5 to 10 retries
            reraise=True,
        )
        def call():
            return func(*args, **kwargs)
        return call()

    async def is_playlist_owned_by_user(self, playlist_id: str) -> bool:
        playlist = await asyncio.to_thread(lambda: self._safe_api_call(self.sp.playlist, playlist_id))
        return playlist["owner"]["id"] == SPOTIFY_USER_ID

    async def get_user_playlists(self) -> List[Dict[str, Any]]:
        async with self._semaphore:
            return await asyncio.to_thread(lambda: self._safe_api_call(self._get_user_playlists))

    def _get_user_playlists(self) -> List[Dict[str, Any]]:
        playlists = self._safe_api_call(self.sp.current_user_playlists)
        return [pl for pl in playlists['items'] if pl['owner']['id'] == SPOTIFY_USER_ID]

    async def get_playlist_tracks(self, playlist_id: str) -> List[Dict[str, Any]]:
        async with self._semaphore:
            return await asyncio.to_thread(lambda: self._safe_api_call(self._get_playlist_tracks, playlist_id))

    def _get_playlist_tracks(self, playlist_id: str) -> List[Dict[str, Any]]:
        tracks = []
        results = self._safe_api_call(self.sp.playlist_items, playlist_id, limit=100, offset=0)
        while results:
            tracks.extend([item["track"] for item in results.get("items", []) if item.get("track") is not None])
            if results.get("next"):
                results = self._safe_api_call(self.sp.next, results)
            else:
                break
        return tracks

    async def get_audio_features(self, track_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        track_ids = list(set(track_ids))
        missing = [tid for tid in track_ids if tid not in self._audio_features_cache]
        if missing:
            try:
                async with self._semaphore:
                    new_features = await asyncio.to_thread(lambda: self._safe_api_call(self._get_audio_features, missing))
                self._audio_features_cache.update(new_features)
                logger.info(f"Fetched audio features for {len(missing)} tracks")
            except SpotifyException as e:
                logger.error(f"Error fetching audio features: {e}")
        return {tid: self._audio_features_cache[tid] for tid in track_ids if tid in self._audio_features_cache}

    def _get_audio_features(self, track_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        features = self._safe_api_call(self.sp.audio_features, track_ids)
        return {feat["id"]: feat for feat in features if feat is not None}

    async def get_new_releases(self) -> List[Dict[str, Any]]:
        async with self._semaphore:
            new_albums = await asyncio.to_thread(lambda: self._safe_api_call(self.sp.new_releases, country="US", limit=20))
        tracks = []
        for album in new_albums["albums"]["items"]:
            album_tracks = await asyncio.to_thread(lambda: self._safe_api_call(self.sp.album_tracks, album["id"]))
            for item in album_tracks["items"]:
                item["album"] = album
                tracks.append(item)
        return tracks

    async def get_artists(self, artist_ids: List[str]) -> Dict[str, str]:
        artist_ids = list(set(artist_ids))
        missing = [aid for aid in artist_ids if aid not in self._artists_cache]
        if missing:
            try:
                async with self._semaphore:
                    new_artists = await asyncio.to_thread(lambda: self._safe_api_call(self._get_artists, missing))
                self._artists_cache.update(new_artists)
                logger.info(f"Fetched artist info for {len(missing)} artists")
            except SpotifyException as e:
                logger.error(f"Error fetching artists: {e}")
        return {aid: self._artists_cache[aid] for aid in artist_ids if aid in self._artists_cache}

    def _get_artists(self, artist_ids: List[str]) -> Dict[str, str]:
        result = {}
        for i in range(0, len(artist_ids), 50):
            chunk = artist_ids[i:i+50]
            artists_info = self._safe_api_call(self.sp.artists, chunk)
            for artist in artists_info["artists"]:
                if artist:
                    result[artist["id"]] = artist["name"]
        return result

    async def search_track(self, track_name: str, artist_names: List[str], limit: int = 10) -> Optional[Dict[str, Any]]:
        async with self._semaphore:
            return await asyncio.to_thread(lambda: self._safe_api_call(self._search_track, track_name, artist_names, limit))

    def _search_track(self, track_name: str, artist_names: List[str], limit: int) -> Optional[Dict[str, Any]]:
        query = f"track:{track_name} artist:{' '.join(artist_names[:3])}"
        results = self._safe_api_call(self.sp.search, q=query, type="track", limit=limit)
        items = results.get("tracks", {}).get("items", [])
        items.sort(key=lambda x: x.get("popularity", 0), reverse=True)
        return items[0] if items else None

    async def upload_playlist(self, playlist_id: str, new_track_ids: List[str]) -> None:
        async with self._semaphore:
            # Await the async function directly rather than using asyncio.to_thread
            owned = await self.is_playlist_owned_by_user(playlist_id)
            if not owned:
                logger.info(f"Skipping playlist {playlist_id} - not owned by the user.")
                return
            # _upload_playlist is synchronous so we offload it to a thread
            return await asyncio.to_thread(self._upload_playlist, playlist_id, new_track_ids)

    def _unfollow_playlist(self, playlist_id: str) -> None:
        self._safe_api_call(self.sp.user_playlist_unfollow, SPOTIFY_USER_ID, playlist_id)
        logger.info(f"Backup playlist {playlist_id} unfollowed successfully.")

    
    def _upload_playlist(self, playlist_id: str, new_track_ids: List[str]) -> None:
        logger.info(f"Uploading playlist {playlist_id} with {len(new_track_ids)} tracks")
        # Retrieve the original playlist details.
        original_playlist = self._safe_api_call(self.sp.playlist, playlist_id)
        if original_playlist.get("collaborative"):
            logger.info(f"Skipping playlist {playlist_id} because it is collaborative and cannot be modified via the API.")
            return
        playlist_name = original_playlist.get("name", f"Playlist {playlist_id}")
        # Create a backup playlist.
        backup_playlist = self._safe_api_call(
            self.sp.user_playlist_create,
            SPOTIFY_USER_ID,
            f"Backup of {playlist_name}",
            public=False,
            description="Backup in case of errors"
        )
        backup_playlist_id = backup_playlist["id"]
        try:
            # Backup the new track order in batches of 100.
            for i in range(0, len(new_track_ids), 100):
                batch = new_track_ids[i:i+100]
                self._safe_api_call(self.sp.playlist_add_items, backup_playlist_id, batch)
            # Remove current tracks from the target playlist.
            current_tracks = self._get_playlist_tracks(playlist_id)
            current_ids = [t["id"] for t in current_tracks]
            for i in range(0, len(current_ids), 100):
                batch = current_ids[i:i+100]
                self._safe_api_call(self.sp.playlist_remove_all_occurrences_of_items, playlist_id, batch)
            # Add new tracks to the target playlist.
            for i in range(0, len(new_track_ids), 100):
                batch = new_track_ids[i:i+100]
                self._safe_api_call(self.sp.playlist_add_items, playlist_id, batch)
            logger.info(f"Uploaded new playlist {playlist_id}")
        finally:
            try:
                self._safe_api_call(self._unfollow_playlist, backup_playlist_id)
            except Exception as e:
                logger.error(f"Failed to unfollow backup playlist {backup_playlist_id} after multiple attempts: {e}")

    async def get_category_name_to_ids(self) -> Dict[str, str]:
        async with self._semaphore:
            return await asyncio.to_thread(lambda: self._safe_api_call(self._get_category_name_to_ids))

    def _get_category_name_to_ids(self) -> Dict[str, str]:
        categories_container = self._safe_api_call(self.sp.categories, locale="en_US", country="US", limit=50)
        category_names_to_ids = {}
        while categories_container:
            for item in categories_container.get("categories", {}).get("items", []):
                category_names_to_ids[item["name"]] = item["id"]
            if categories_container.get("next"):
                categories_container = self._safe_api_call(self.sp.next, categories_container)
            else:
                break
        return category_names_to_ids

    async def get_featured_playlist_ids(self) -> List[str]:
        async with self._semaphore:
            return await asyncio.to_thread(lambda: self._safe_api_call(self._get_featured_playlist_ids))

    def _get_featured_playlist_ids(self) -> List[str]:
        playlists_container = self._safe_api_call(self.sp.featured_playlists, locale="en_US", country="US", limit=50)
        playlists = playlists_container.get("playlists", {})
        return [item['id'] for item in playlists.get("items", [])]

    async def get_category_playlist_ids(self, category_id: str) -> List[str]:
        async with self._semaphore:
            return await asyncio.to_thread(lambda: self._safe_api_call(self._get_category_playlist_ids, category_id))

    def _get_category_playlist_ids(self, category_id: str) -> List[str]:
        playlists_container = self._safe_api_call(self.sp.category_playlists, category_id, country="US", limit=20)
        playlists = playlists_container.get("playlists", {})
        return [item['id'] for item in playlists.get("items", []) if item is not None]
