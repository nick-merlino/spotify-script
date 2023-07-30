from argparse import ArgumentParser
import logging
from pandas import DataFrame, read_csv, concat
from pathlib import Path
from pickle import dump, load
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from config import *
from spotipy_api import *

SPOTIFY_DATABASE = "spotify-database.csv"
ARTISTS_CACHE = ".artists_cache.pkl"

playlist_name_to_id = dict()
artist_id_to_name = dict()

# Configure logging
logging.basicConfig(level=logging.CRITICAL, format="%(levelname)s - %(message)s")
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def load_song_database():
    global spotify_database
    
    try:
        if not Path(SPOTIFY_DATABASE).exists():
            raise FileNotFoundError("Spotify database not found. Exiting...")
        
        spotify_database = read_csv(SPOTIFY_DATABASE, index_col=0, encoding_errors="replace", dtype={
            "playlist_id": "object", 
            "playlist_name": "object", 
            "track_name": "object", 
            "duration_ms": "float64", 
            "popularity": "float64", 
            "artists": "object",
            "danceability": "float64", 
            "valence": "float64", 
            "energy": "float64", 
            "tempo": "float64", 
            "loudness": "float64", 
            "speechiness": "float64", 
            "instrumentalness": "float64", 
            "liveness": "float64", 
            "acousticness": "float64", 
            "key": "float64", 
            "mode": "float64", 
            "time_signature": "float64", 
            "user_deleted": "object", 
            "script_deleted": "object", 
            "last_deleted_date": "object",
            "release_year": "float64",
            "explicit": "float64"
            })
        
        # ~~~~~ This section is just in case there are duplicate track_id entries, which are the index values

        # Assign a priority value based on the playlist_priority_list
        priority_dict = {playlist_name: i for i, playlist_name in enumerate(playlist_priority_list, start=1)}

        # Apply the priority to the playlist_name column
        spotify_database["playlist_priority"] = spotify_database["playlist_name"].map(lambda x: priority_dict.get(x, len(priority_dict) + 1))

        # Sort the DataFrame based on the priority value
        spotify_database.sort_values(by="playlist_priority", inplace=True)

        # Drop the temporary priority column
        spotify_database.drop(columns=["playlist_priority"], inplace=True)

        # Drop duplicate index values, keeping the first occurrence (highest priority playlist name)
        spotify_database.drop_duplicates(subset=spotify_database.index.name, keep="first", inplace=True)
    except FileNotFoundError as e:
        logging.error(e)
        exit()

def load_artists_cache():
    global artist_id_to_name
    try:
        if not Path(ARTISTS_CACHE).exists():
            raise FileNotFoundError("Artists cache not found. Continuing...")
        
        artist_id_to_name = load(open(ARTISTS_CACHE, "rb"))
    except FileNotFoundError as e:
        logging.info(e)
   

def find_new_music():
    global spotify_database

    playlist_id_set = set()

    for _, category_id in category_dict.items():
        playlist_id_set.update(get_category_playlist_ids(category_id))

    playlist_id_set.update(get_featured_playlist_ids())
    playlist_id_set.update(popular_playlists.values())

    num_new_songs = 0
    for playlist_id in tqdm(list(playlist_id_set), desc="Getting new playlist tracks"):
        tracks = get_playlist_tracks(playlist_id)
        for track in tracks:
            track_id = track["id"]
            
            if track_id not in spotify_database.index:
                num_new_songs += 1
                new_df = DataFrame({
                    "playlist_id": playlist_name_to_id["Popular on Spotify"],
                    "playlist_name": "Popular on Spotify",
                    "track_name": track["name"],
                    "duration_ms": track["duration_ms"],
                    "popularity": track["popularity"],
                    "artists": ','.join([x["id"] for x in track["artists"]]),
                    "release_year": track["album"]["release_date"].split('-')[0] if track["album"]["release_date"] else None,
                    "explicit": 1 if track["explicit"] else 0
                }, index=[track_id])
                spotify_database = concat([spotify_database, new_df])
    
    logging.info(f"New songs: {num_new_songs}")


def update_playlist_info(playlists):    
    global spotify_database
    
    playlist_id_to_track_id_set = dict()

    # Create a dictionary to map playlist names to their corresponding index in the order_list
    playlist_order = {playlist_name: index for index, playlist_name in enumerate(playlist_priority_list)}

    # Get the maximum index in the playlist_order dictionary
    max_index = max(playlist_order.values())
    
    # Sort the playlists using the custom sorting key based on their order in the order_list
    ordered_playlists = sorted(playlists, key=lambda playlist: playlist_order.get(playlist['name'], max_index + 1), reverse=True)
        
    for playlist in tqdm(ordered_playlists, desc="Updating database with current playlist info"):
        playlist_name_to_id[playlist["name"]] = playlist["id"]

        for track in get_playlist_tracks(playlist["id"]):
            track_id = track["id"]
            
            if playlist["id"] not in playlist_id_to_track_id_set:
                playlist_id_to_track_id_set[playlist["id"]] = set()
            playlist_id_to_track_id_set[playlist["id"]].add(track_id)

            if track_id not in spotify_database.index:
                new_df = DataFrame({
                    "playlist_id": playlist["id"],
                    "playlist_name": playlist["name"],
                    "track_name": track["name"],
                    "duration_ms": track["duration_ms"],
                    "popularity": track["popularity"],
                    "release_year": track["album"]["release_date"].split('-')[0] if track["album"]["release_date"] else None,
                    "explicit": 1 if track["explicit"] else 0,
                    "artists": ','.join([x["id"] for x in track["artists"]])
                }, index=[track_id])
                spotify_database = concat([spotify_database, new_df])
            else:
                spotify_database.at[track_id, "playlist_id"] = playlist["id"]
                spotify_database.at[track_id, "playlist_name"] = playlist["name"]
                spotify_database.at[track_id, "track_name"] = track["name"]
                spotify_database.at[track_id, "explicit"] = 1 if track["explicit"] else 0
                spotify_database.at[track_id, "release_year"] = track["album"]["release_date"].split('-')[0] if track["album"]["release_date"] else None
                spotify_database.at[track_id, "popularity"] = track["popularity"]
                spotify_database.at[track_id, "duration_ms"] = track["duration_ms"]
                spotify_database.at[track_id, "artists"] = ','.join([x["id"] for x in track["artists"]])
                spotify_database.at[track_id, "user_deleted"] = np.NaN
                spotify_database.at[track_id, "script_deleted"] = np.NaN
                spotify_database.at[track_id, "last_deleted_date"] = np.NaN



    for track_id, track in spotify_database[spotify_database["last_deleted_date"].isnull()].iterrows():
        if track["playlist_id"] not in playlist_id_to_track_id_set or track_id not in playlist_id_to_track_id_set[track["playlist_id"]]:
            spotify_database.at[track_id, "user_deleted"] = True
            spotify_database.at[track_id, "last_deleted_date"] = datetime.now().strftime("%m/%d/%Y %H:%M:%S")


def update_artist_cache():
    global artist_id_to_name
    
    artist_ids = set()
    for artist_id_string in set(spotify_database["artists"]):
        for artist_id in artist_id_string.split(','):
            artist_ids.add(artist_id)
    
    new_artist_ids = set(artist_ids) - set(artist_id_to_name.keys())
    if new_artist_ids:
        num_new_artists = len(new_artist_ids)
        logging.info(f"{num_new_artists} new artists")
        artist_id_to_name.update(get_artists(new_artist_ids))

    dump(artist_id_to_name, open(ARTISTS_CACHE, "wb"))
   
def get_popular_track_verions():
    global spotify_database
    
    # Greedy choice to only do this for non-deleted tracks
    # It does disregard any tracks that might be more popular than it was in the past
    user_tracks_df = spotify_database[spotify_database["last_deleted_date"].isnull()]
    
    # Create a tqdm progress bar
    progress_bar = tqdm(total=len(user_tracks_df), desc="Getting most popular track versions")

    for track_id, track in user_tracks_df.iterrows():
        progress_bar.update(1)
        original_artists_ids = track["artists"].split(',')
        original_artists = [artist_id_to_name[artist_id] for artist_id in original_artists_ids]

        highest_popularity_track = get_highest_popularity_track(track["track_name"], original_artists)
            
        if highest_popularity_track and track_id != highest_popularity_track["id"]:
            if highest_popularity_track["id"] not in spotify_database.index:
                new_df = DataFrame({
                    "playlist_id": track["playlist_id"],
                    "playlist_name": track["playlist_name"],
                    "track_name": highest_popularity_track["name"],
                    "duration_ms": highest_popularity_track["duration_ms"],
                    "popularity": highest_popularity_track["popularity"],
                    "artists": ','.join([x["id"] for x in highest_popularity_track["artists"]]),
                    "release_year": highest_popularity_track["album"]["release_date"].split('-')[0],
                    "explicit": 1 if highest_popularity_track["explicit"] else 0
                }, index=[highest_popularity_track["id"]])
                spotify_database = concat([spotify_database, new_df])

            spotify_database.at[track_id, "last_deleted_date"] = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
            spotify_database.at[track_id, "script_deleted"] = True

def delete_duds():
    for track_id, track_df in spotify_database[spotify_database["last_deleted_date"].isna()].iterrows():
        min_popularity = playlist_name_to_min_popularity[track_df["playlist_name"]]
        if track_df["popularity"] < min_popularity:
            spotify_database.at[track_id, "script_deleted"] = True
            spotify_database.at[track_id, "last_deleted_date"] = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

def update_track_audio_features():
    no_attribute_ids = spotify_database[(spotify_database["last_deleted_date"].isna()) & (spotify_database["energy"].isnull())].index.tolist()
    track_id_to_attributes = get_track_audio_features(no_attribute_ids)

    for track_id, attributes in tqdm(track_id_to_attributes.items(), desc="Updating audio features"):
        spotify_database.at[track_id, "danceability"] = attributes["danceability"]
        spotify_database.at[track_id, "energy"] = attributes["energy"]
        spotify_database.at[track_id, "key"] = attributes["key"]
        spotify_database.at[track_id, "speechiness"] = attributes["speechiness"]
        spotify_database.at[track_id, "loudness"] = attributes["loudness"]
        spotify_database.at[track_id, "mode"] = attributes["mode"]
        spotify_database.at[track_id, "acousticness"] = attributes["acousticness"]
        spotify_database.at[track_id, "instrumentalness"] = attributes["instrumentalness"]
        spotify_database.at[track_id, "liveness"] = attributes["liveness"]
        spotify_database.at[track_id, "valence"] = attributes["valence"]
        spotify_database.at[track_id, "tempo"] = attributes["tempo"]
        spotify_database.at[track_id, "time_signature"] = attributes["time_signature"]



def upload_playlists():
    for playlist_id, group_df in tqdm(spotify_database.groupby("playlist_id"), desc="Uploading playlists"):
        not_deleted_df = group_df[group_df["last_deleted_date"].isna()]
        sorted_df = not_deleted_df.sort_values(by=["popularity"], ascending=False)
        upload_playlist(playlist_id, group_df["playlist_name"].unique()[0], sorted_df.index.to_list())

def plot_playlists():
    audio_features = ["duration_ms", "popularity", "danceability", "valence", "energy", "tempo", "loudness", "speechiness", "instrumentalness", "liveness", "acousticness", "key", "mode", "time_signature", "release_year", "explicit"]

    user_tracks_df = spotify_database[spotify_database["last_deleted_date"].isnull()]

    playlist_names = []
    audio_feature_averages = []
    
    for audio_feature in audio_features:
        averages = []
        playlist_names = [] # Cheating to clear it - hopefully they're the same
        for playlist_name, playlist_df in user_tracks_df.groupby("playlist_name"):
            playlist_names.append(playlist_name)
            averages.append(playlist_df[audio_feature].mean())
        audio_feature_averages.append(averages)
    
    # Plotting the average values for each audio feature
    rows, cols = 4, 4
    fig, axs = plt.subplots(rows, cols, figsize=(20, 15))
    fig.suptitle("Average Audio Features for each Playlist", fontsize=20)

    for i, audio_feature in enumerate(audio_features):
        row, col = i // cols, i % cols
        axs[row, col].scatter(range(len(audio_feature_averages[i])), audio_feature_averages[i], marker='o', s=100)
        axs[row, col].set_title(f"{audio_feature.capitalize()}")
        axs[row, col].set_xticks(range(len(audio_feature_averages[i])))
        axs[row, col].set_xticklabels(playlist_names, rotation=45, ha='right')
        axs[row, col].set_xlabel("Playlist Name")
        axs[row, col].set_ylabel(f"Average {audio_feature.capitalize()}")

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"audio_features_plot.png")

# Untested
def restore_deleted_songs():
    for track_id, track_df in spotify_database[(spotify_database["last_deleted_date"].notnull()) & (spotify_database["script_deleted"].notnull())].iterrows():
        curr_time = datetime.now()
        time = datetime.strptime(track_df["last_deleted_date"], "%m/%d/%Y %H:%M:%S")
        if (curr_time - time) <= timedelta(hours=24):
            spotify_database.at[track_id, "script_deleted"] = np.NaN
            spotify_database.at[track_id, "last_deleted_date"] = np.NaN

if __name__ == "__main__":
    parser = ArgumentParser()
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-n", action="store_true", help="Find new music")
    group.add_argument("-d", action="store_true", help="Delete duds")
    
    group2 = group.add_mutually_exclusive_group()
    group2.add_argument("-r", action="store_true", help="Restore deleted songs from today")
    
    group3 = group.add_mutually_exclusive_group()
    group3.add_argument("-c", action="store_true", help="Print current spotify categories")

    group4 = group.add_mutually_exclusive_group()
    group.add_argument("-p", action="store_true", help="Plot attributes")

    args = parser.parse_args()

    if args.c:
        logging.info("Getting current categories")
        for category_name, category_id in get_category_name_to_ids().items():
            print(f"{category_name}: {category_id}")
        exit()
    
    load_song_database()
    
    if args.p:
        plot_playlists()
        exit()

    load_artists_cache()
    playlists = get_user_playlists()    
    update_playlist_info(playlists)
    spotify_database.to_csv("spotify-database.csv")
    
    if args.r:
        print("Restoring deleted tracks from today")
        restore_deleted_songs()
        exit()

    if args.n:
        logging.info("Finding New Music")
        find_new_music()
        spotify_database.to_csv("spotify-database.csv")

    update_artist_cache()

    if args.d:
        get_popular_track_verions()
        delete_duds()

    update_track_audio_features()
    spotify_database.to_csv("spotify-database.csv")
    
    # logging.info("Moving Songs")
    # moveTracks()

    upload_playlists()


    # print("Create Playlist Clusters")
    # clusterPlaylists()











# DANCEABILITY_QUALIFIER = 0.65
# DANCE_ENERGY_QUALIFIER = 0.8

# HIGH_ENERGY_QUALIFIER = 0.8
# LOW_ENERGY_QUALIFIER = 0.5
# NEW_MUSIC_QUALIFIER = 85




# def process_energy(just_good_music, spotify_and_chill, slow_it_down):
#     move_to_just_good_music_track_id_set = set()
#     move_to_spotify_and_chill_track_id_set = set()
#     move_to_slow_it_down_track_id_set = set()

#     for track_id in playlist_id_to_attributes[just_good_music]["track_id_set"]:
#         track_energy = track_id_to_attributes[track_id]["features"]["energy"]
#         if track_energy < HIGH_ENERGY_QUALIFIER and track_energy >= LOW_ENERGY_QUALIFIER:
#             move_to_spotify_and_chill_track_id_set.add(track_id)
#             print("Moved: " + track_id_to_attributes[track_id]["name"] +
#                   " from just good music to Spotify and Chill for its energy of " + str(track_energy))
#         elif track_energy < LOW_ENERGY_QUALIFIER:
#             move_to_slow_it_down_track_id_set.add(track_id)
#             print("Moved: " + track_id_to_attributes[track_id]["name"] +
#                   " from just good music to Slow it Down for its energy of " + str(track_energy))

#     for track_id in playlist_id_to_attributes[spotify_and_chill]["track_id_set"]:
#         track_energy = track_id_to_attributes[track_id]["features"]["energy"]
#         if track_energy >= HIGH_ENERGY_QUALIFIER:
#             move_to_just_good_music_track_id_set.add(track_id)
#             print("Moved: " + track_id_to_attributes[track_id]["name"] +
#                   " from Spotify and Chill to Just Good Music for its energy of " + str(track_energy))
#         elif track_energy < LOW_ENERGY_QUALIFIER:
#             move_to_slow_it_down_track_id_set.add(track_id)
#             print("Moved: " + track_id_to_attributes[track_id]["name"] +
#                   " from Spotify and Chill to Slow it Down for its energy of " + str(track_energy))

#     for track_id in playlist_id_to_attributes[slow_it_down]["track_id_set"]:
#         track_energy = track_id_to_attributes[track_id]["features"]["energy"]
#         if track_energy >= HIGH_ENERGY_QUALIFIER:
#             move_to_just_good_music_track_id_set.add(track_id)
#             print("Moved: " + track_id_to_attributes[track_id]["name"] +
#                   " from Slow it Down to Just Good Music for its energy of " + str(track_energy))
#         elif track_energy >= LOW_ENERGY_QUALIFIER and track_energy < HIGH_ENERGY_QUALIFIER:
#             move_to_spotify_and_chill_track_id_set.add(track_id)
#             print("Moved: " + track_id_to_attributes[track_id]["name"] +
#                   " from Slow it Down to Spotify and Chill for its energy of " + str(track_energy))

#     if move_to_just_good_music_track_id_set:
#         for chunk in divide_chunks_list(list(move_to_just_good_music_track_id_set), 100):
#             spotify.playlist_add_items(just_good_music, chunk)
#             spotify.playlist_remove_all_occurrences_of_items(
#                 spotify_and_chill, chunk)
#             spotify.playlist_remove_all_occurrences_of_items(
#                 slow_it_down, chunk)

#         playlist_id_to_attributes[just_good_music]["track_id_set"].update(
#             move_to_just_good_music_track_id_set)
#         playlist_id_to_attributes[spotify_and_chill]["track_id_set"] = playlist_id_to_attributes[spotify_and_chill]["track_id_set"].difference(
#             move_to_just_good_music_track_id_set)
#         playlist_id_to_attributes[slow_it_down]["track_id_set"] = playlist_id_to_attributes[slow_it_down]["track_id_set"].difference(
#             move_to_just_good_music_track_id_set)

#     if move_to_spotify_and_chill_track_id_set:
#         for chunk in divide_chunks_list(list(move_to_spotify_and_chill_track_id_set), 100):
#             spotify.playlist_add_items(spotify_and_chill, chunk)
#             spotify.playlist_remove_all_occurrences_of_items(
#                 just_good_music, chunk)
#             spotify.playlist_remove_all_occurrences_of_items(
#                 slow_it_down, chunk)

#         playlist_id_to_attributes[spotify_and_chill]["track_id_set"].update(
#             move_to_spotify_and_chill_track_id_set)
#         playlist_id_to_attributes[just_good_music]["track_id_set"] = playlist_id_to_attributes[just_good_music]["track_id_set"].difference(
#             move_to_spotify_and_chill_track_id_set)
#         playlist_id_to_attributes[slow_it_down]["track_id_set"] = playlist_id_to_attributes[slow_it_down]["track_id_set"].difference(
#             move_to_spotify_and_chill_track_id_set)

#     if move_to_slow_it_down_track_id_set:
#         for chunk in divide_chunks_list(list(move_to_slow_it_down_track_id_set), 100):
#             spotify.playlist_add_items(slow_it_down, chunk)
#             spotify.playlist_remove_all_occurrences_of_items(
#                 spotify_and_chill, chunk)
#             spotify.playlist_remove_all_occurrences_of_items(
#                 just_good_music, chunk)

#         playlist_id_to_attributes[slow_it_down]["track_id_set"].update(
#             move_to_slow_it_down_track_id_set)
#         playlist_id_to_attributes[spotify_and_chill]["track_id_set"] = playlist_id_to_attributes[spotify_and_chill]["track_id_set"].difference(
#             move_to_slow_it_down_track_id_set)
#         playlist_id_to_attributes[just_good_music]["track_id_set"] = playlist_id_to_attributes[just_good_music]["track_id_set"].difference(
#             move_to_slow_it_down_track_id_set)


# def moveSongsUpToDanceTheNightAway(from_playlist_id, to_playlist_id):
#     moved_track_id_set = set()
#     for track_id in playlist_id_to_attributes[from_playlist_id]["track_id_set"]:
#         if "features" in track_id_to_attributes[track_id]:
#             if track_id_to_attributes[track_id]["features"]["danceability"] >= DANCEABILITY_QUALIFIER and track_id_to_attributes[track_id]["features"]["energy"] >= DANCE_ENERGY_QUALIFIER:
#                 moved_track_id_set.add(track_id)
#                 print("Moved: " + track_id_to_attributes[track_id]["name"] + " up for its danceability of " + str(
#                     track_id_to_attributes[track_id]["features"]["danceability"]))
#         else:
#             print("No features for " +
#                   track_id_to_attributes[track_id]["name"])

#     if moved_track_id_set:
#         for chunk in divide_chunks_list(list(moved_track_id_set), 100):
#             spotify.playlist_add_items(to_playlist_id, list(chunk))
#             spotify.playlist_remove_all_occurrences_of_items(
#                 from_playlist_id, list(chunk))

#         playlist_id_to_attributes[to_playlist_id]["track_id_set"].update(
#             moved_track_id_set)
#         playlist_id_to_attributes[from_playlist_id]["track_id_set"] = playlist_id_to_attributes[from_playlist_id]["track_id_set"].difference(
#             moved_track_id_set)


# def moveSongsDownToPoolside(from_playlist_id, to_playlist_id):
#     moved_track_id_set = set()
#     for track_id in playlist_id_to_attributes[from_playlist_id]["track_id_set"]:
#         if "features" in track_id_to_attributes[track_id]:
#             if track_id_to_attributes[track_id]["features"]["danceability"] < DANCEABILITY_QUALIFIER or track_id_to_attributes[track_id]["features"]["energy"] < DANCE_ENERGY_QUALIFIER:
#                 moved_track_id_set.add(track_id)
#                 print("Moved: " + track_id_to_attributes[track_id]["name"] + " down for its danceability of " + str(
#                     track_id_to_attributes[track_id]["features"]["danceability"]))
#         else:
#             print("No features for " +
#                   track_id_to_attributes[track_id]["name"])

#     if moved_track_id_set:
#         for chunk in divide_chunks_list(list(moved_track_id_set), 100):
#             spotify.playlist_add_items(to_playlist_id, chunk)
#             spotify.playlist_remove_all_occurrences_of_items(
#                 from_playlist_id, chunk)

#         playlist_id_to_attributes[to_playlist_id]["track_id_set"].update(
#             moved_track_id_set)
#         playlist_id_to_attributes[from_playlist_id]["track_id_set"] = playlist_id_to_attributes[from_playlist_id]["track_id_set"].difference(
#             moved_track_id_set)


# def moveSongs():
#     for playlist_id, playlist_attributes in playlist_id_to_attributes.items():
#         if playlist_attributes["name"] == "Just Good Music":
#             just_good_music = playlist_id
#         elif playlist_attributes["name"] == "Dance the Night Away":
#             dance_the_night_away = playlist_id
#         elif playlist_attributes["name"] == "Spotify and Chill":
#             spotify_and_chill = playlist_id
#         elif playlist_attributes["name"] == "Poolside":
#             poolside = playlist_id
#         elif playlist_attributes["name"] == "Slow it Down":
#             slow_it_down = playlist_id

#     process_energy(just_good_music, spotify_and_chill, slow_it_down)

#     moveSongsUpToDanceTheNightAway(poolside, dance_the_night_away)
#     moveSongsDownToPoolside(dance_the_night_away, poolside)




# def clusterPlaylists():
#     column_names = ["id", "danceability", "energy", "key", "loudness", "mode", "speechiness",
#                     "acousticness", "instrumentalness", "liveness", "valence", "tempo", "time_signature"]
#     all_songs = pd.DataFrame(columns=column_names)
#     for playlist in own_playlists:
#         if playlist["name"] != "Popular on Spotify":
#             songs = [x for x in get_playlist_items(playlist["id"]) if "track" in x and isinstance(
#                 x["track"], dict) and "id" in x["track"]]
#             for song_group in divide_chunks_list(songs, 100):
#                 songs_features = spotify.audio_features(
#                     [x["track"]["id"] for x in song_group])
#                 for song_features in songs_features:
#                     new_song = pd.DataFrame([[song_features["id"], song_features["danceability"], song_features["energy"], song_features["key"], song_features["loudness"], song_features["mode"], song_features["speechiness"],
#                                             song_features["acousticness"], song_features["instrumentalness"], song_features["liveness"], song_features["valence"], song_features["tempo"], song_features["time_signature"]]], columns=column_names)
#                     all_songs = all_songs.append(new_song, ignore_index=True)

#     # all_songs.plot()
#     # plt.show()

#     all_songs.set_index("id", inplace=True)

#     scaler = StandardScaler()
#     all_songs_scaled = scaler.fit_transform(all_songs)
#     all_songs_normalized = normalize(all_songs_scaled)
#     all_songs_normalized = pd.DataFrame(
#         all_songs_normalized, index=all_songs.index)

#     # print(all_songs_normalized.head())
#     # all_songs_normalized.plot()
#     # plt.show()

#     # test = PCA().fit(all_songs_normalized)
#     # plt.plot(np.cumsum(test.explained_variance_ratio_))
#     # plt.xlabel("number of components")
#     # plt.ylabel("cum exp variance")
#     # plt.show()

#     pca = PCA(n_components=0.95)
#     all_songs_principal = pca.fit_transform(all_songs_normalized)
#     all_songs_principal = pd.DataFrame(
#         all_songs_principal, index=all_songs_normalized.index)
#     all_songs_principal.columns = [
#         "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"]
#     # all_songs_principal.plot()
#     # plt.show()

#     # Building the clustering model
#     n_clusters = 20
#     spectral_model_rbf = SpectralClustering(
#         n_clusters=n_clusters, affinity="rbf")  # "nearest_neighbors"

#     cluster_song_lists = {}
#     for idx in range(0, n_clusters):
#         cluster_song_lists[idx] = []

    # # Training the model and Storing the predicted cluster labels
    # labels_rbf = spectral_model_rbf.fit_predict(all_songs_principal)
    # for idx, id in enumerate(all_songs_principal.index.values.tolist()):
    #     cluster_song_lists[labels_rbf[idx]].append(id)

    # # for idx in range (0, n_clusters):
    # #     playlist_id = spotify.user_playlist_create(user_id, "Cluster " + str(idx+1))["id"]
    # #     for chunk in divide_chunks_list(list(cluster_song_lists[idx]), 100):
    # #         spotify.playlist_add_items(playlist_id, chunk)

    # # cluster_column_names = ["playlist_name", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "time_signature"]
    # # cluster_features = pd.DataFrame(columns = column_names)
    # # for playlist in own_playlists:
    # #     if playlist["name"] != "Popular on Spotify":
    # #         playlist_column_names = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "time_signature"]
    # #         playlist_songs = pd.DataFrame(columns = playlist_column_names)
    # #         songs = [x for x in get_playlist_items(playlist["id"]) if "track" in x and isinstance(x["track"], dict) and "id" in x["track"]]
    # #         for song_group in divide_chunks_list(songs, 100):
    # #             songs_features = spotify.audio_features([x["track"]["id"] for x in song_group])
    # #             for song_features in songs_features:
    # #                 new_song = pd.DataFrame([[song_features["danceability"], song_features["energy"], song_features["key"], song_features["loudness"], song_features["mode"], song_features["speechiness"], song_features["acousticness"], song_features["instrumentalness"], song_features["liveness"], song_features["valence"], song_features["tempo"], song_features["time_signature"]]], columns=playlist_column_names)
    # #                 playlist_songs = playlist_songs.append(new_song, ignore_index=True)
    # #     danceability = playlist_songs["danceability"].mean()
    # #     energy = playlist_songs["danceability"].mean()
    # #     key = playlist_songs["danceability"].mean()
    # #     loudness = playlist_songs["danceability"].mean()
    # #     mode = playlist_songs["danceability"].mean()
    # #     speechiness = playlist_songs["danceability"].mean()
    # #     acousticness = playlist_songs["danceability"].mean()
    # #     instrumentalness = playlist_songs["danceability"].mean()
    # #     liveness = playlist_songs["danceability"].mean()
    # #     valance = playlist_songs["danceability"].mean()
    # #     tempo = playlist_songs["danceability"].mean()
    # #     time_signature = playlist_songs["danceability"].mean()
    # #     cluster_features = cluster_features.append(pd.DataFrame([[playlist["name"], danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valance, tempo, time_signature]], columns=cluster_column_names), ignore_index=True)
    # # cluster_features.set_index("playlist_name", inplace=True)




