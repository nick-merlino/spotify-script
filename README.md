# Spotify Playlist Manager
 
Spotify Playlist Manager is a comprehensive tool for curating, updating, and optimizing your Spotify playlists. It synchronizes your Spotify account with a local SQLite database, applies intelligent filtering and sorting based on track attributes, and even uses machine learning to predict optimal playlist assignments. In addition, it provides rich visualizations to help you analyze audio feature trends across your playlists.

Unfortunately, Spotify has disabled a lot of their endpoint APIs and so a lot of functionality no longer works
https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api

TODO
- Add tqdm loading bars to functionality
- Get most popular version query exceeds maximum length of 25 characters
- Be able to lower ToProcess playlist minimum popularity and undo previous 'script_deleted' status
- confirm the resulting playlist is in order of popularity (it's not)

---
 
## Features
 
- **Playlist Synchronization:**  
  Automatically fetch and update your playlists and track metadata from Spotify into a local database.
 
- **Persistent & In-Memory Caching:**  
  Leverages both persistent caching (via diskcache) and in-memory caching to minimize redundant API calls and mitigate rate limiting.
 
- **Asynchronous Database Access:**  
  Uses asynchronous operations with aiosqlite and connection pooling for efficient, concurrent database operations.
 
- **Task Queue for Background Processing:**  
  Manages long-running tasks—such as machine learning training—with an asyncio-based task queue that includes built-in retry logic.
 
- **Machine Learning Predictions:**  
  Offloads heavy ML training to a separate process (via ProcessPoolExecutor) to predict optimal playlist assignments based on track audio features. The model is cached on disk and reused when possible.
 
- **Advanced Data Visualizations:**  
  Provides a suite of visualizations including a heatmap, radar chart, scatter plot with regression, and parallel coordinates plot to analyze audio feature trends across playlists.
 
- **Robust Error Handling & Logging:**  
  Implements structured logging with detailed error messages and advanced retry logic to ensure smooth operation.
 
- **Modular & Extensible Design:**  
  With clear separation between configuration, database access, API integration, task management, business logic, and plotting, the tool is easy to customize and extend.
 
---
 
## Why Use This Tool?
 
- **Automated Curation:**  
  Keep your playlists fresh and well-organized without manual intervention.
 
- **Quality Assurance:**  
  Automatically filter out duplicate or low-quality tracks to ensure you always enjoy the best versions of your favorite songs.
 
- **Data-Driven Insights:**  
  Visualize trends and patterns in your playlists to better understand your listening habits and optimize your music selection.
 
- **Customization & Extensibility:**  
  Easily adjust settings, add features, or integrate with other systems as your needs evolve.
 
---
 
## Prerequisites
 
- **Python 3.8+**
- Spotify Developer Account credentials:
   - `SPOTIFY_CLIENT_ID`
   - `SPOTIFY_CLIENT_SECRET`
   - `SPOTIFY_REDIRECT_URI` (default: `http://localhost:8888/callback`)
   - `SPOTIFY_USER_ID`
 
---
 
## Installation
 
1. **Clone the Repository:**
 
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
 
2. **Create and Activate a Virtual Environment:**
 
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
 
3. **Install Dependencies:**
 
   Create a `requirements.txt` file with the following packages (or install them manually):
 
   ```txt
  aiosqlite
  dash
  diskcache
  joblib
  matplotlib
  pandas
  python-dotenv
  scikit-learn
  spotipy
  streamlit
  sqlalchemy
  tenacity
  tqdm
  seaborn
   ```
   
   Then run:
 
   ```bash
   pip install -r requirements.txt
   ```
 
4. **Store Your Spotify Credentials Securely:**
 
   Create a file named `.env` in the project root and add your credentials:
 
   ```dotenv
   SPOTIFY_CLIENT_ID=your_client_id
   SPOTIFY_CLIENT_SECRET=your_client_secret
   SPOTIFY_REDIRECT_URI=http://localhost:8888/callback
   SPOTIFY_USER_ID=your_spotify_user_id
   ```
 
   **Important:** Ensure that your `.env` file is added to your `.gitignore` so that your credentials are not committed.
 
---
 
## Project Structure
 
- **config.py:**  
  Contains configuration variables, including Spotify credentials, database settings, and feature parameters.
 
- **database.py:**  
  Implements asynchronous database operations using aiosqlite with a connection pool, and provides functions for initializing and accessing the database.
 
- **spotify_api.py:**  
  Wraps Spotify API calls with caching and retry logic, and retrieves playlist data, audio features, and artist information.
 
- **task_queue.py:**  
  A lightweight, asyncio-based task queue that handles long-running background tasks with automatic retries.
 
- **business_logic.py:**  
  Contains core functionality for synchronizing playlists, updating track and artist data, filtering and sorting tracks, generating ML-based playlist predictions, and producing advanced visualizations of audio features.
 
- **main.py:**  
  The command-line interface (CLI) entry point that allows you to run individual functions or the entire pipeline.
 
---
 
## Usage
 
Run the full pipeline:
 
```bash
python main.py -hfull
```
 
Alternatively, run individual tasks with these flags:
 
- **-n:** Update playlist info from user playlists.
- **-f:** Find new music from categories and featured playlists.
- **-a:** Update artist cache.
- **-u:** Update audio features.
- **-b:** Ensure best versions of tracks are used.
- **-d:** Delete duds (low popularity tracks).
- **-t:** Move tracks between playlists based on energy.
- **-r:** Restore recently deleted songs.
- **-m:** Generate playlist predictions using ML.
- **-p:** Upload playlists to Spotify (sorted by popularity).
- **-plt:** Generate enhanced visualizations of playlist audio features.
 
Example (update playlist info, audio features, and plot visualizations):
 
```bash
python main.py -n -u -plt
```
 
---
 
## Notes
 
- **Spotify Authentication:**  
  When you run the tool for the first time, a browser window will open for Spotify authentication. Ensure your redirect URI is correctly set in your Spotify Developer Dashboard.
 
- **Caching:**  
  The tool uses both persistent caching (via diskcache) and in-memory caching to reduce redundant API calls. Cache lifetimes (TTLs) are configurable.
 
- **ML Model Caching:**  
  The machine learning model is saved to disk (`model.pkl`) and is reused if the training data remains unchanged.
 
- **Database:**  
  The local SQLite database (`spotify.db`) is automatically created. For larger-scale applications, consider migrating to a more robust database solution.
 
- **Dependencies:**  
  Ensure all required Python packages are installed as listed in `requirements.txt`.
 
---