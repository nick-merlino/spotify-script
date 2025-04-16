# main.py
import argparse
import asyncio
import logging
from business_logic import BusinessLogic
from database import pool
import subprocess

logging.getLogger("tenacity").setLevel(logging.ERROR)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

class Ignore429Filter(logging.Filter):
    def filter(self, record):
        # Ignore messages that include a 429 error response
        if "returned 429" in record.getMessage():
            return False
        return True
    
# Add the filter to the spotipy.client logger
spotipy_logger = logging.getLogger("spotipy.client")
spotipy_logger.addFilter(Ignore429Filter())
    
def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced Spotify Playlist Manager")
    parser.add_argument("-n", action="store_true", help="Update playlist info from user playlists")
    # parser.add_argument("-f", action="store_true", help="Find new music from categories and featured playlists")
    parser.add_argument("-a", action="store_true", help="Update artist cache")
    # parser.add_argument("-u", action="store_true", help="Update audio features")
    parser.add_argument("-b", action="store_true", help="Ensure best versions of tracks are used")
    parser.add_argument("-d", action="store_true", help="Delete duds (low popularity tracks)")
    parser.add_argument("-t", action="store_true", help="Move tracks between playlists based on energy")
    parser.add_argument("-r", action="store_true", help="Restore recently deleted songs")
    # parser.add_argument("-m", action="store_true", help="Generate playlist predictions using ML")
    parser.add_argument("-p", action="store_true", help="Upload playlists to Spotify (sorted by popularity)")
    # parser.add_argument("-plt", action="store_true", help="Plot enhanced playlist visualizations")
    parser.add_argument("-hfull", action="store_true", help="Run full pipeline")
    parser.add_argument("-plot", action="store_true", help="Launch interactive dashboard for playlists")
    args = parser.parse_args()
    
    if args.plot:
        try:
            # Launch the Streamlit app.
            subprocess.run(["streamlit", "run", "dashboard.py"])
        except Exception as e:
            logger.error(f"Failed to launch dashboard: {e}")
        return

    logic = BusinessLogic()
    loop = asyncio.get_event_loop()
    
    loop.run_until_complete(logic.initialize())
    
    tasks = []
    if args.n:
        tasks.append(logic.update_playlist_info())
    # if args.f:
    #     tasks.append(logic.find_new_music())
    if args.a:
        tasks.append(logic.update_artist_cache())
    # if args.u:
    #     tasks.append(logic.update_audio_features()) # DEPRECATED SINCE SPOTIFY CHANGED THEIR API ACCESS
    if tasks:
        loop.run_until_complete(asyncio.gather(*tasks))
    if args.b:
        loop.run_until_complete(logic.ensure_best_versions())
    if args.d:
        logic.delete_duds()
    if args.t:
        logic.move_tracks()
    if args.r:
        logic.restore_deleted_songs()
    # if args.m:
    #     logic.generate_playlist_predictions()
    if args.p:
        loop.run_until_complete(logic.upload_playlists())
    # if args.plt:
    #     loop.run_until_complete(logic.plot_playlists()) # DEPRECATED SINCE SPOTIFY CHANGED THEIR API ACCESS
    if args.hfull:
        loop.run_until_complete(logic.run_full_pipeline())

    # Close the database connection pool to allow the event loop to finish
    loop.run_until_complete(pool.close())
    logger.info("Database pool closed, exiting.")

if __name__ == "__main__":
    main()
