import asyncio
import datetime
import logging
import os
import math
from typing import Any, Dict, List
import pandas as pd
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

from database import init_db, insert_or_update_track, get_all_tracks, get_track_by_id, insert_or_update_artist, get_all_artists, pool
from spotify_api import SpotifyAPI
from config import audio_features, playlist_priority_list, playlist_name_to_min_popularity, selected_category_names, playlist_to_process, playlist_to_process_id, spotify_playlists
from task_queue import TaskQueue

logger = logging.getLogger(__name__)
MODEL_FILENAME = "model.pkl"

def train_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list) -> pd.Series:
    """
    Trains a RandomForestClassifier on the training data and predicts playlist labels for the test data.
    Returns a pandas Series mapping track_id to the predicted playlist name.
    """
    label_encoder = LabelEncoder()
    train_df["label"] = label_encoder.fit_transform(train_df["playlist_name"])
    X_train = train_df[features]
    y_train = train_df["label"]
    X_test = test_df[features]
    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(X_train, y_train)
    predicted_labels = clf.predict(X_test)
    predicted_playlists = label_encoder.inverse_transform(predicted_labels)
    return pd.Series(predicted_playlists, index=test_df["track_id"])

class BusinessLogic:
    def __init__(self, spotify: SpotifyAPI = None) -> None:
        self.spotify = spotify if spotify is not None else SpotifyAPI()
        self.task_queue = TaskQueue()
        self.loop = asyncio.get_event_loop()

    async def initialize(self) -> None:
        # Initialize the pool first to avoid hangs during DB initialization
        await pool.init_pool()
        await init_db()

    def get_priority(self, playlist_name: str) -> int:
        try:
            return playlist_priority_list.index(playlist_name)
        except ValueError:
            # unknown playlist ⇒ highest priority
            return 7


    async def get_selected_category_ids(self) -> List[str]:
        """
        Retrieves the list of category IDs corresponding to the names specified in config.selected_category_names.
        """
        category_mapping = await self.spotify.get_category_name_to_ids()
        selected_ids = []
        for cat_name in selected_category_names:
            if cat_name in category_mapping:
                selected_ids.append(category_mapping[cat_name])
            else:
                logger.warning(f"Category '{cat_name}' not found in Spotify's category list.")
        return selected_ids

    async def update_playlist_info(self) -> None:
        try:
            playlists = await self.spotify.get_user_playlists()
            tasks = []
            for pl in tqdm(playlists, desc="Updating playlist info"):
                tasks.append(self._update_playlist_tracks(pl))
            await asyncio.gather(*tasks)
            logger.info("Playlist info updated")
        except Exception as e:
            logger.error(f"Error in update_playlist_info: {e}")

    async def _update_playlist_tracks(self, playlist: Dict[str, Any]) -> None:
        playlist_id = playlist["id"]
        playlist_name = playlist["name"]
        try:
            tracks = await self.spotify.get_playlist_tracks(playlist_id)
            for track in tqdm(tracks, desc=f"Processing tracks for {playlist_name}", leave=False):
                track_id = track["id"]
                existing = await get_track_by_id(track_id)
                new_priority = self.get_priority(playlist_name)
                if existing:
                    current_playlist = existing[2]  # playlist_name column
                    current_priority = self.get_priority(current_playlist)
                    if new_priority >= current_priority:
                        continue
                track_data = {
                    "track_id": track_id,
                    "playlist_id": playlist_id,
                    "playlist_name": playlist_name,
                    "track_name": track["name"],
                    "duration_ms": track["duration_ms"],
                    "popularity": track["popularity"],
                    "artists": ",".join([artist["id"] for artist in track["artists"]]),
                    "release_year": int(track["album"]["release_date"].split("-")[0]) if track["album"]["release_date"] else None,
                    "explicit": 1 if track["explicit"] else 0,
                    "danceability": None,
                    "energy": None,
                    "key": None,
                    "speechiness": None,
                    "loudness": None,
                    "mode": None,
                    "acousticness": None,
                    "instrumentalness": None,
                    "liveness": None,
                    "valence": None,
                    "tempo": None,
                    "time_signature": None,
                    "user_deleted": None,
                    "script_deleted": None,
                    "last_deleted_date": None
                }
                await insert_or_update_track(track_data)
        except Exception as e:
            logger.error(f"Error updating playlist {playlist_name}: {e}")

    async def find_new_music(self) -> None:
        try:
            new_count = 0
            
            new_releases_tracks = await self.spotify.get_new_releases()
            for track in tqdm(new_releases_tracks, desc="Processing new releases"):
                # Only insert track if it doesn't already exist in the db.
                if not await get_track_by_id(track["id"]):
                    track_data = {
                        "track_id": track["id"],
                        "playlist_id": playlist_to_process_id,
                        "playlist_name": playlist_to_process,
                        "track_name": track["name"],
                        "duration_ms": track["duration_ms"],
                        "popularity": track.get("popularity", 0),
                        "artists": ",".join([artist["id"] for artist in track["artists"]]),
                        "release_year": int(track["album"]["release_date"].split("-")[0]) if track["album"].get("release_date") else None,
                        "explicit": 1 if track.get("explicit") else 0,
                        "danceability": None,
                        "energy": None,
                        "key": None,
                        "speechiness": None,
                        "loudness": None,
                        "mode": None,
                        "acousticness": None,
                        "instrumentalness": None,
                        "liveness": None,
                        "valence": None,
                        "tempo": None,
                        "time_signature": None,
                        "user_deleted": None,
                        "script_deleted": None,
                        "last_deleted_date": None
                    }
                    await insert_or_update_track(track_data)
                    new_count += 1
            
            for playlist_name, playlist_id in tqdm(spotify_playlists.items(), desc="Processing additional playlists"):
                tracks = await self.spotify.get_playlist_tracks(playlist_id)
                for track in tqdm(tracks, desc=f"Processing tracks for {playlist_name}", leave=False):
                    # Only insert track if it doesn't already exist in the db.
                    if not await get_track_by_id(track["id"]):
                        track_data = {
                            "track_id": track["id"],
                            "playlist_id": playlist_to_process_id,
                            "playlist_name": playlist_to_process,
                            "track_name": track["name"],
                            "duration_ms": track["duration_ms"],
                            "popularity": track["popularity"],
                            "artists": ",".join([artist["id"] for artist in track["artists"]]),
                            "release_year": int(track["album"]["release_date"].split("-")[0])
                                             if track["album"].get("release_date") else None,
                            "explicit": 1 if track["explicit"] else 0,
                            "danceability": None,
                            "energy": None,
                            "key": None,
                            "speechiness": None,
                            "loudness": None,
                            "mode": None,
                            "acousticness": None,
                            "instrumentalness": None,
                            "liveness": None,
                            "valence": None,
                            "tempo": None,
                            "time_signature": None,
                            "user_deleted": None,
                            "script_deleted": None,
                            "last_deleted_date": None
                        }
                        await insert_or_update_track(track_data)
                        new_count += 1

            logger.info(f"New songs found: {new_count}")
        except Exception as e:
            logger.error(f"Error in find_new_music: {e}")

    async def update_artist_cache(self) -> None:
        try:
            tracks = await get_all_tracks()
            artist_ids = set()
            for row in tqdm(tracks, desc="Processing tracks for artist cache"):
                artists_str = row[6]
                if artists_str:
                    artist_ids.update(artists_str.split(","))
            current_artists = await get_all_artists()
            missing_artist_ids = list(artist_ids - set(current_artists.keys()))
            if missing_artist_ids:
                artists_info = await self.spotify.get_artists(missing_artist_ids)
                for artist_id, name in tqdm(artists_info.items(), desc="Updating artist cache", leave=False):
                    await insert_or_update_artist(artist_id, name)
            logger.info("Artist cache updated")
        except Exception as e:
            logger.error(f"Error in update_artist_cache: {e}")

    async def update_audio_features(self) -> None:
        try:
            tracks = await get_all_tracks()
            missing_ids = [row[0] for row in tracks if row[10] is None]
            for i in range(0, len(missing_ids), 100):
                chunk = missing_ids[i:i+100]
                features_dict = await self.spotify.get_audio_features(chunk)
                for track_id, features in features_dict.items():
                    self._update_track_audio_features(track_id, {
                        "danceability": features.get("danceability"),
                        "energy": features.get("energy"),
                        "key": features.get("key"),
                        "speechiness": features.get("speechiness"),
                        "loudness": features.get("loudness"),
                        "mode": features.get("mode"),
                        "acousticness": features.get("acousticness"),
                        "instrumentalness": features.get("instrumentalness"),
                        "liveness": features.get("liveness"),
                        "valence": features.get("valence"),
                        "tempo": features.get("tempo"),
                        "time_signature": features.get("time_signature")
                    })
            logger.info("Audio features updated")
        except Exception as e:
            logger.error(f"Error in update_audio_features: {e}")

    def _update_track_audio_features(self, track_id: str, update_data: Dict[str, Any]) -> None:
        try:
            conn = sqlite3.connect("spotify.db")
            columns = ", ".join(f"{k}=?" for k in update_data.keys())
            values = list(update_data.values())
            values.append(track_id)
            conn.execute(f"UPDATE tracks SET {columns} WHERE track_id=?", values)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error updating audio features for {track_id}: {e}")

    async def refresh_popularity_local(self) -> None:
        """
        Refresh local `popularity` for all tracks that are still active
        (script_deleted IS NULL AND user_deleted IS NULL) by batching Spotify API calls.
        """
        import sqlite3
        import asyncio
        from tqdm import tqdm

        # collect active track IDs
        conn = sqlite3.connect("spotify.db")
        cur = conn.cursor()
        cur.execute(
            "SELECT track_id FROM tracks "
            "WHERE script_deleted IS NULL AND user_deleted IS NULL"
        )
        ids = [row[0] for row in cur.fetchall()]
        conn.close()

        # batch refresh popularity in chunks of 50
        for i in tqdm(range(0, len(ids), 50),
                      desc="Refreshing popularity",
                      unit="batch",
                      dynamic_ncols=True):
            batch = ids[i: i + 50]
            results = await asyncio.to_thread(
                lambda: self.spotify._safe_api_call(self.spotify.sp.tracks, batch)
            )

            # update each track's popularity in the DB
            conn = sqlite3.connect("spotify.db")
            cur = conn.cursor()
            for tr in tqdm(results.get("tracks", []),
                           desc="Updating batch popularity",
                           unit="track",
                           leave=False,
                           dynamic_ncols=True):
                cur.execute(
                    "UPDATE tracks SET popularity = ? WHERE track_id = ?",
                    (tr.get("popularity", 0), tr["id"])
                )
            conn.commit()
            conn.close()


    async def dedupe_exact_local(self) -> None:
        """
        Deduplicate by exact (track_name, artists) for all active tracks
        (script_deleted IS NULL AND user_deleted IS NULL). Keep only the one with
        highest local popularity, marking others as user_deleted.
        """
        import sqlite3
        from datetime import datetime
        from collections import defaultdict
        from tqdm import tqdm

        # load active tracks
        conn = sqlite3.connect("spotify.db")
        cur = conn.cursor()
        cur.execute(
            "SELECT track_id, track_name, artists, popularity FROM tracks "
            "WHERE script_deleted IS NULL AND user_deleted IS NULL"
        )
        rows = cur.fetchall()
        conn.close()

        # group by exact metadata
        buckets = defaultdict(list)
        for track_id, name, artists, pop in tqdm(rows,
                                                desc="Grouping tracks",
                                                unit="row",
                                                dynamic_ncols=True):
            buckets[(name, artists)].append((track_id, pop))

        timestamp = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        # dedupe each cluster
        for (name, artists), items in tqdm(buckets.items(),
                                           desc="Deduplicating clusters",
                                           unit="cluster",
                                           dynamic_ncols=True):
            if len(items) <= 1:
                continue
            # choose the track with highest popularity
            best_id, _ = max(items, key=lambda x: x[1])
            # mark all other duplicates as user_deleted
            conn = sqlite3.connect("spotify.db")
            cur = conn.cursor()
            for tid, _ in items:
                if tid != best_id:
                    cur.execute(
                        "UPDATE tracks SET user_deleted = 1, last_deleted_date = ? "
                        "WHERE track_id = ? "
                        "AND script_deleted IS NULL AND user_deleted IS NULL",
                        (timestamp, tid)
                    )
            conn.commit()
            conn.close()


    async def ensure_best_versions(self) -> None:
        """
        Deduplicate: for each active track (script_deleted & user_deleted are null),
        find the most popular Spotify variant, upsert it, and mark all other copies as user_deleted.
        """
        try:
            import sqlite3
            import datetime
            import pandas as pd
            from collections import defaultdict
            from tqdm import tqdm
            # Load only active tracks
            conn = sqlite3.connect("spotify.db")
            df = pd.read_sql_query(
                "SELECT * FROM tracks WHERE script_deleted IS NULL AND user_deleted IS NULL",
                conn
            )
            conn.close()

            # Pre-fetch artist cache
            artist_cache = await get_all_artists()
            best_map: Dict[str, str] = {}

            # Determine best variant for each track
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Finding best variants"):
                orig_id = row["track_id"]
                artists = row["artists"].split(",") if row["artists"] else []
                artist_names = [artist_cache.get(a, "") for a in artists]
                track_query = row["track_name"]
                best = await self.spotify.search_track(track_query, artist_names)
                if best:
                    best_map[orig_id] = best["id"]
                    if best["id"] != orig_id and best["popularity"] > row["popularity"]:
                        # Upsert the more popular variant
                        best_data = {
                            "track_id": best["id"],
                            "playlist_id": row["playlist_id"],
                            "playlist_name": row["playlist_name"],
                            "track_name": best["name"],
                            "duration_ms": best["duration_ms"],
                            "popularity": best["popularity"],
                            "artists": ",".join([artist["id"] for artist in best["artists"]]),
                            "release_year": int(best["album"]["release_date"].split("-")[0]) if best["album"].get("release_date") else None,
                            "explicit": 1 if best["explicit"] else 0,
                            # preserve audio_features columns
                            **{feat: None for feat in audio_features if feat not in [
                                "track_id","playlist_id","playlist_name","track_name",
                                "duration_ms","popularity","artists","release_year","explicit"
                            ]}
                        }
                        await insert_or_update_track(best_data)
                else:
                    best_map[orig_id] = orig_id

            # Group originals by their best variant
            groups: Dict[str, List[str]] = defaultdict(list)
            for orig, best_id in best_map.items():
                groups[best_id].append(orig)

            # Mark non-best copies as user_deleted if still active
            timestamp = datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S")
            for best_id, orig_list in groups.items():
                for orig_id in orig_list:
                    if orig_id != best_id:
                        conn = sqlite3.connect("spotify.db")
                        conn.execute(
                            """
                            UPDATE tracks
                               SET user_deleted     = 1,
                                   last_deleted_date = ?
                             WHERE track_id       = ?
                               AND script_deleted IS NULL
                               AND user_deleted IS NULL
                            """,
                            (timestamp, orig_id)
                        )
                        conn.commit()
                        conn.close()

            logger.info("Best variants ensured and duplicates user_deleted.")
        except Exception as e:
            logger.error(f"Error in ensure_best_versions: {e}")

    def _mark_track_deleted(self, track_id: str, timestamp: str) -> None:
        """
        Marks a track as script_deleted=1 (and updates last_deleted_date)
        only if it has not already been marked as user_deleted.
        """
        try:
            conn = sqlite3.connect("spotify.db")
            conn.execute(
                """
                UPDATE tracks
                SET script_deleted   = 1,
                    last_deleted_date = ?
                WHERE track_id       = ?
                AND user_deleted IS NULL
                """,
                (timestamp, track_id)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error marking track {track_id} as deleted: {e}")


    def delete_duds(self) -> None:
        """
        Marks tracks as script deleted if their popularity is below the minimum threshold
        and automatically restores (un-deletes) tracks that were script-deleted if their
        popularity is now above the threshold.
        Only applies to tracks not manually deleted by the user.
        """
        try:
            conn = sqlite3.connect("spotify.db")
            cur = conn.cursor()
            cur.execute("SELECT track_id, playlist_name, popularity, script_deleted FROM tracks WHERE user_deleted IS NULL")
            rows = cur.fetchall()
            now = datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S")
            for row in tqdm(rows, desc="Processing dud tracks"):
                track_id, playlist_name, popularity, script_deleted = row
                min_pop = playlist_name_to_min_popularity.get(playlist_name, 0)
                if popularity < min_pop:
                    if script_deleted != 1:
                        conn.execute("UPDATE tracks SET script_deleted=1, last_deleted_date=? WHERE track_id=?", (now, track_id))
                else:
                    if script_deleted == 1:
                        conn.execute("UPDATE tracks SET script_deleted=NULL, last_deleted_date=NULL WHERE track_id=?", (track_id,))
            conn.commit()
            conn.close()
            logger.info("Dud tracks processed")
        except Exception as e:
            logger.error(f"Error deleting duds: {e}")

    def move_tracks(self) -> None:
        try:
            conn = sqlite3.connect("spotify.db")
            cur = conn.cursor()
            cur.execute("SELECT track_id, playlist_name, energy FROM tracks WHERE script_deleted IS NULL AND energy IS NOT NULL")
            rows = cur.fetchall()
            for row in tqdm(rows, desc="Moving tracks based on energy"):
                track_id, current_playlist, energy = row
                new_playlist = current_playlist
                if current_playlist in ["Slow it Down", "Spotify and Chill", "Just Good Music"]:
                    if energy > 0.75:
                        new_playlist = "Just Good Music"
                    elif energy > 0.45:
                        new_playlist = "Spotify and Chill"
                    else:
                        new_playlist = "Slow it Down"
                elif current_playlist in ["Poolside", "Dance the Night Away"]:
                    new_playlist = "Dance the Night Away" if energy > 0.8 else "Poolside"
                elif current_playlist in ["I'll Day Drink to That", "Kick'n Back"]:
                    new_playlist = "I'll Day Drink to That" if energy > 0.75 else "Kick'n Back"
                if new_playlist != current_playlist:
                    conn.execute("UPDATE tracks SET playlist_name=? WHERE track_id=?", (new_playlist, track_id))
            conn.commit()
            conn.close()
            logger.info("Tracks moved based on energy")
        except Exception as e:
            logger.error(f"Error moving tracks: {e}")

    def restore_deleted_songs(self) -> None:
        try:
            conn = sqlite3.connect("spotify.db")
            cur = conn.cursor()
            cur.execute("SELECT track_id, last_deleted_date FROM tracks WHERE last_deleted_date IS NOT NULL")
            rows = cur.fetchall()
            now = datetime.datetime.now()
            for track_id, last_deleted in tqdm(rows, desc="Restoring deleted songs"):
                try:
                    deleted_time = datetime.datetime.strptime(last_deleted, "%m/%d/%Y %H:%M:%S")
                    if (now - deleted_time) <= datetime.timedelta(hours=24):
                        conn.execute("UPDATE tracks SET script_deleted=NULL, user_deleted=NULL, last_deleted_date=NULL WHERE track_id=?", (track_id,))
                except Exception:
                    continue
            conn.commit()
            conn.close()
            logger.info("Deleted songs restored")
        except Exception as e:
            logger.error(f"Error restoring deleted songs: {e}")

    def restore_high_popularity_tracks(self) -> None:
        try:
            conn = sqlite3.connect("spotify.db")
            cur = conn.cursor()
            cur.execute("SELECT track_id, popularity FROM tracks WHERE script_deleted=1")
            rows = cur.fetchall()
            for track_id, popularity in rows:
                if popularity > 90:
                    conn.execute("UPDATE tracks SET playlist_name='Popular (Restored)', script_deleted=NULL, user_deleted=NULL, last_deleted_date=NULL WHERE track_id=?", (track_id,))
            conn.commit()
            conn.close()
            logger.info("High popularity tracks restored")
        except Exception as e:
            logger.error(f"Error restoring high popularity tracks: {e}")

    async def upload_playlists(self) -> None:
        """
        Uploads each playlist with only active tracks:
        - script_deleted IS NULL
        - user_deleted   IS NULL
        Deduplicates by (track_name, artists), keeping the highest-popularity version.
        """
        try:
            conn = sqlite3.connect("spotify.db")
            # only select tracks that haven't been script- or user-deleted
            df = pd.read_sql_query(
                "SELECT * FROM tracks "
                "WHERE script_deleted IS NULL AND user_deleted IS NULL",
                conn
            )
            conn.close()

            grouped = df.groupby("playlist_name")
            tasks = []
            for playlist_name, group in tqdm(
                list(grouped),
                desc="Uploading playlists",
                dynamic_ncols=True,
                leave=False
            ):
                # sort so the most popular version is first
                sorted_group = group.sort_values(by="popularity", ascending=False)
                # drop any exact duplicates (name+artists), keep the top one
                unique_group = sorted_group.drop_duplicates(
                    subset=["track_name", "artists"],
                    keep="first"
                )
                track_ids = unique_group["track_id"].tolist()
                playlist_id = unique_group["playlist_id"].iloc[0]
                tasks.append(self.spotify.upload_playlist(playlist_id, track_ids))

            if tasks:
                await asyncio.gather(*tasks)
            logger.info("Playlists uploaded")
        except Exception as e:
            logger.error(f"Error uploading playlists: {e}")


    def generate_playlist_predictions(self) -> None:
        try:
            conn = sqlite3.connect("spotify.db")
            df = pd.read_sql_query("SELECT * FROM tracks WHERE script_deleted IS NULL", conn)
            conn.close()
            df = df.dropna(subset=audio_features)
            if df.empty:
                logger.info("No data for ML prediction")
                return
            train_df = df[df["playlist_name"] != "ToProcess"]
            test_df = df[df["playlist_name"] == "ToProcess"]
            if train_df.empty or test_df.empty:
                logger.info("Insufficient data for training or testing")
                return

            # Offload heavy ML training to a separate process
            with ProcessPoolExecutor() as executor:
                future = executor.submit(train_and_predict, train_df, test_df, audio_features)
                predictions = future.result()

            # Update predictions in the database
            for track_id, predicted_playlist in predictions.items():
                conn = sqlite3.connect("spotify.db")
                conn.execute("UPDATE tracks SET playlist_name=? WHERE track_id=?", (predicted_playlist, track_id))
                conn.commit()
                conn.close()
            logger.info("Playlist predictions generated and applied")
        except Exception as e:
            logger.error(f"Error generating playlist predictions: {e}")

    async def plot_playlists(self) -> None:
        """
        Generates multiple visualizations of playlist audio features:
        - A heatmap of average audio features per playlist.
        - A radar chart overlay comparing audio feature profiles.
        - A scatter plot of average energy vs. average popularity.
        - A parallel coordinates plot of audio features by playlist.
        Saves the combined figure as 'audio_features_plot.png'.
        """
        try:
            # Retrieve all tracks from the database
            tracks = await get_all_tracks()
            import matplotlib.pyplot as plt
            import seaborn as sns
            from pandas.plotting import parallel_coordinates

            columns = [
                "track_id", "playlist_id", "playlist_name", "track_name", "duration_ms", "popularity",
                "artists", "release_year", "explicit", "danceability", "energy", "key", "speechiness",
                "loudness", "mode", "acousticness", "instrumentalness", "liveness", "valence", "tempo",
                "time_signature", "user_deleted", "script_deleted", "last_deleted_date"
            ]
            df = pd.DataFrame(tracks, columns=columns)
            df = df[df["script_deleted"].isnull()]

            # Group by playlist and compute average audio features
            group = df.groupby("playlist_name").mean()
            avg_features = group[audio_features]

            # Create a figure with multiple subplots
            fig = plt.figure(constrained_layout=True, figsize=(18, 20))
            gs = fig.add_gridspec(3, 2)

            # Heatmap of audio features per playlist
            ax1 = fig.add_subplot(gs[0, :])
            sns.heatmap(avg_features, annot=True, cmap="viridis", ax=ax1)
            ax1.set_title("Average Audio Features per Playlist (Heatmap)")

            # Radar chart overlay of audio feature profiles
            labels = audio_features
            num_vars = len(labels)
            angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
            angles += angles[:1]
            ax2 = fig.add_subplot(gs[1, 0], polar=True)
            for playlist in avg_features.index:
                values = avg_features.loc[playlist].tolist()
                values += values[:1]
                ax2.plot(angles, values, label=playlist)
                ax2.fill(angles, values, alpha=0.25)
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(labels, fontsize=8)
            ax2.set_title("Radar Chart of Audio Profiles", y=1.1)
            ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

            # Scatter plot of average energy vs. average popularity
            avg_energy = group["energy"]
            avg_popularity = group["popularity"]
            ax3 = fig.add_subplot(gs[1, 1])
            sns.regplot(x=avg_energy, y=avg_popularity, ax=ax3)
            for playlist in group.index:
                ax3.text(avg_energy.loc[playlist], avg_popularity.loc[playlist], playlist, fontsize=9)
            ax3.set_xlabel("Average Energy")
            ax3.set_ylabel("Average Popularity")
            ax3.set_title("Energy vs. Popularity per Playlist")

            # Parallel coordinates plot of audio features by playlist
            par_df = avg_features.reset_index()
            ax4 = fig.add_subplot(gs[2, :])
            parallel_coordinates(par_df, 'playlist_name', colormap='viridis', ax=ax4)
            ax4.set_title("Parallel Coordinates Plot of Audio Features by Playlist")
            ax4.legend(loc='upper right', fontsize=8)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig("audio_features_plot.png")
            plt.close()
            logger.info("Enhanced playlist plots saved as 'audio_features_plot.png'")
        except Exception as e:
            logger.error(f"Error plotting playlists: {e}")

    async def mark_user_deleted(self) -> None:
        """
        Marks tracks as user_deleted = 1 if they were in your DB (hence in a playlist before),
        but now appear in no playlists, and have not already been script-deleted.
        Shows a tqdm progress bar for the updates.
        """
        # 1) Gather every track ID currently in any of your playlists
        playlists = await self.spotify.get_user_playlists()
        current_ids = set()
        for pl in playlists:
            tracks = await self.spotify.get_playlist_tracks(pl["id"])
            current_ids.update(t["id"] for t in tracks)

        # 2) Load every track ID from your database
        all_tracks = await get_all_tracks()  # returns list of rows, row[0] is track_id
        db_ids = {row[0] for row in all_tracks}

        # 3) Compute which IDs have disappeared
        gone_ids = list(db_ids - current_ids)
        if not gone_ids:
            return  # nothing to mark

        # 4) Only mark those that aren't script-deleted, with a tqdm bar
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        async with pool.get_connection() as conn:
            for tid in tqdm(
                gone_ids,
                desc="Marking user_deleted tracks",
                unit="track",
                dynamic_ncols=True,
                leave=False
            ):
                await conn.execute(
                    """
                    UPDATE tracks
                       SET user_deleted     = 1,
                           last_deleted_date = ?
                     WHERE track_id         = ?
                       AND script_deleted IS NULL
                    """,
                    (timestamp, tid)
                )
            await conn.commit()


    async def run_full_pipeline(self) -> None:
        await self.update_playlist_info()
        await self.mark_user_deleted()
        # await self.find_new_music()
        await self.update_artist_cache()
        # await self.update_audio_features()
        await self.refresh_popularity_local()
        await self.dedupe_exact_local()
        self.delete_duds()
        self.move_tracks()
        # self.restore_deleted_songs()
        # self.restore_high_popularity_tracks()
        # self.task_queue.add_task(self.async_generate_predictions)
        await self.task_queue.run(num_workers=2)
        await self.upload_playlists()

    async def async_generate_predictions(self) -> None:
        self.generate_playlist_predictions()
