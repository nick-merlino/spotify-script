# database.py
import aiosqlite
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple
from config import DATABASE_FILE

class ConnectionPool:
    def __init__(self, db_file: str, pool_size: int = 5) -> None:
        self.db_file = db_file
        self.pool_size = pool_size
        self._pool: asyncio.Queue = asyncio.Queue(maxsize=pool_size)

    async def init_pool(self) -> None:
        for _ in range(self.pool_size):
            conn = await aiosqlite.connect(self.db_file)
            await conn.execute("PRAGMA foreign_keys = ON;")
            await self._pool.put(conn)

    @asynccontextmanager
    async def get_connection(self):
        conn = await self._pool.get()
        try:
            yield conn
        finally:
            await self._pool.put(conn)

    async def close(self) -> None:
        while not self._pool.empty():
            conn = await self._pool.get()
            await conn.close()

pool = ConnectionPool(DATABASE_FILE, pool_size=5)

async def init_db() -> None:
    async with pool.get_connection() as conn:
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS tracks (
                track_id TEXT PRIMARY KEY,
                playlist_id TEXT,
                playlist_name TEXT,
                track_name TEXT,
                duration_ms REAL,
                popularity REAL,
                artists TEXT,
                release_year INTEGER,
                explicit INTEGER,
                danceability REAL,
                energy REAL,
                key REAL,
                speechiness REAL,
                loudness REAL,
                mode REAL,
                acousticness REAL,
                instrumentalness REAL,
                liveness REAL,
                valence REAL,
                tempo REAL,
                time_signature REAL,
                user_deleted INTEGER,
                script_deleted INTEGER,
                last_deleted_date TEXT
            );
            CREATE TABLE IF NOT EXISTS artists (
                artist_id TEXT PRIMARY KEY,
                name TEXT
            );
        """)
        await conn.commit()

async def insert_or_update_track(track: Dict[str, Any]) -> None:
    columns = ", ".join(track.keys())
    placeholders = ", ".join("?" for _ in track)
    update_clause = ", ".join(f"{col}=excluded.{col}" for col in track.keys())
    values = list(track.values())
    query = f"""
    INSERT INTO tracks ({columns})
    VALUES ({placeholders})
    ON CONFLICT(track_id) DO UPDATE SET {update_clause};
    """
    async with pool.get_connection() as conn:
        await conn.execute(query, values)
        await conn.commit()

async def get_all_tracks() -> List[Tuple[Any, ...]]:
    async with pool.get_connection() as conn:
        cursor = await conn.execute("SELECT * FROM tracks")
        rows = await cursor.fetchall()
        return rows

async def get_track_by_id(track_id: str) -> Optional[Tuple[Any, ...]]:
    async with pool.get_connection() as conn:
        cursor = await conn.execute("SELECT * FROM tracks WHERE track_id = ?", (track_id,))
        row = await cursor.fetchone()
        return row

async def insert_or_update_artist(artist_id: str, name: str) -> None:
    async with pool.get_connection() as conn:
        await conn.execute("""
            INSERT INTO artists (artist_id, name)
            VALUES (?, ?)
            ON CONFLICT(artist_id) DO UPDATE SET name=excluded.name;
        """, (artist_id, name))
        await conn.commit()

async def get_all_artists() -> Dict[str, str]:
    async with pool.get_connection() as conn:
        cursor = await conn.execute("SELECT * FROM artists")
        rows = await cursor.fetchall()
        return {row[0]: row[1] for row in rows}
