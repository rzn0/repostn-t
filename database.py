# database.py
import sqlite3
import os
import logging
import time
from imagehash import ImageHash, hex_to_hash
from typing import Optional, Dict, Any, List, Union, Set

DB_DIR = "db"
DB_NAME = "repost_hashes.db"
DB_PATH = os.path.join(DB_DIR, DB_NAME)
HASH_TABLE_NAME = "media_hashes"
CONFIG_TABLE_NAME = "guild_config"
WHITELIST_TABLE_NAME = "channel_whitelist"

log = logging.getLogger(__name__)

def _ensure_db_dir():
    """Ensures the database directory exists."""
    try:
        abs_db_dir = os.path.abspath(DB_DIR)
        os.makedirs(abs_db_dir, exist_ok=True)
    except OSError as e:
        log.error(f"Failed create/access DB dir '{abs_db_dir}': {e}", exc_info=True)
        raise

def _get_connection() -> Optional[sqlite3.Connection]:
    """Gets a connection to the SQLite database."""
    abs_db_path = os.path.abspath(DB_PATH)
    try:
        _ensure_db_dir()
        conn = sqlite3.connect(abs_db_path, timeout=10, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try: conn.execute("PRAGMA journal_mode=WAL;") # Improve concurrency
        except sqlite3.Error as e: log.warning(f"Could not set journal_mode=WAL: {e}")
        try: conn.execute("PRAGMA busy_timeout = 5000;") # Wait 5s if DB is locked
        except sqlite3.Error as e: log.warning(f"Could not set busy_timeout: {e}")
        try: conn.execute("PRAGMA foreign_keys = ON;")
        except sqlite3.Error as e: log.warning(f"Could not set foreign_keys=ON: {e}")
        log.debug(f"DB connection acquired: {abs_db_path}")
        return conn
    except Exception as e:
        log.error(f"Error connecting to DB {abs_db_path}: {e}", exc_info=True)
        return None

def _close_connection(conn: Optional[sqlite3.Connection]):
    """Closes the database connection."""
    if conn:
        try:
            conn.close()
            log.debug("DB connection closed.")
        except Exception as e:
            log.error(f"Error closing DB connection: {e}", exc_info=True)

def setup_database():
    """Creates the necessary tables and indexes if they don't exist."""
    log.info("--- Running Database Setup ---")
    conn = _get_connection()
    if not conn: raise ConnectionError("DB_SETUP: Failed to get DB connection.")
    try:
        log.info("DB_SETUP: Connection successful. Checking/Creating tables...")
        conn.execute("BEGIN IMMEDIATE;")
        cursor = conn.cursor()

        # Hash Table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {HASH_TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER NOT NULL,
                channel_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL UNIQUE,
                author_id INTEGER NOT NULL,
                hash_hex TEXT NOT NULL,
                media_url TEXT,
                timestamp INTEGER NOT NULL
            );
        """)
        log.info(f"DB_SETUP: Table '{HASH_TABLE_NAME}' ensured.")

        # Config Table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {CONFIG_TABLE_NAME} (
                guild_id INTEGER PRIMARY KEY,
                alert_channel_id INTEGER NULLABLE
            );
        """)
        log.info(f"DB_SETUP: Table '{CONFIG_TABLE_NAME}' ensured.")

        # Whitelist Table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {WHITELIST_TABLE_NAME} (
                guild_id INTEGER NOT NULL,
                channel_id INTEGER NOT NULL,
                PRIMARY KEY (guild_id, channel_id)
            );
        """)
        log.info(f"DB_SETUP: Table '{WHITELIST_TABLE_NAME}' ensured.")

        # Indexes
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_guild_hash ON {HASH_TABLE_NAME} (guild_id);")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_whitelist_guild ON {WHITELIST_TABLE_NAME} (guild_id);")
        log.info("DB_SETUP: Indexes ensured.")

        conn.commit()
        log.info("DB_SETUP: Transaction committed."); log.info("--- Database Setup Function Completed Successfully ---")
    except Exception as e:
        log.critical(f"--- Database Setup Function FAILED: {e} ---", exc_info=True)
        try: conn.rollback()
        except Exception: pass
        raise
    finally:
        _close_connection(conn)


def add_hash(guild_id: int, channel_id: int, message_id: int, author_id: int,
             hashes: Union[ImageHash, List[ImageHash]], media_url: str):
    """Adds a new media hash(es) to the database. Joins lists with ';'."""
    conn = _get_connection();
    if not conn: log.error(f"DB_ADD_HASH: Failed connection for MsgID {message_id}."); return

    hash_hex_string: str = ""
    if isinstance(hashes, list):
        valid_hashes = [str(h) for h in hashes if h is not None]
        if not valid_hashes:
            log.warning(f"DB_ADD_HASH: No valid hashes provided for MsgID {message_id}. Not adding.")
            _close_connection(conn); return
        hash_hex_string = ";".join(valid_hashes)
    elif isinstance(hashes, ImageHash):
        hash_hex_string = str(hashes)
    else:
        log.error(f"DB_ADD_HASH: Invalid hash type ({type(hashes)}) for MsgID {message_id}. Not adding.")
        _close_connection(conn); return

    timestamp = int(time.time())
    log.info(f"DB_ADD_HASH: Adding MsgID {message_id}, Guild {guild_id}, Hash(es) '{hash_hex_string[:20]}...'")
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"INSERT INTO {HASH_TABLE_NAME} (guild_id, channel_id, message_id, author_id, hash_hex, media_url, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (guild_id, channel_id, message_id, author_id, hash_hex_string, media_url, timestamp)
        )
        conn.commit()
        log.info(f"DB_ADD_HASH: Hash(es) added successfully for MsgID {message_id}.")
    except sqlite3.IntegrityError:
        log.warning(f"DB_ADD_HASH: Hash for MsgID {message_id} already exists. Ignoring duplicate.")
        try: conn.rollback()
        except Exception: pass
    except Exception as e:
        log.error(f"DB_ADD_HASH: Error adding hash for message {message_id}: {e}", exc_info=True)
        try: conn.rollback()
        except Exception: pass
    finally: _close_connection(conn)

def find_similar_hash(guild_id: int, current_hashes: Union[ImageHash, List[ImageHash]], threshold: int) -> Optional[Dict[str, Any]]:
    """Finds the earliest similar hash entry in the guild."""
    conn = _get_connection();
    if not conn: log.error(f"DB_FIND_HASH: Failed connection for Guild {guild_id}."); return None
    found_match = None

    target_hashes: List[ImageHash] = []
    if isinstance(current_hashes, list):
        target_hashes = [h for h in current_hashes if isinstance(h, ImageHash)]
    elif isinstance(current_hashes, ImageHash):
        target_hashes = [current_hashes]

    if not target_hashes:
        log.warning(f"DB_FIND_HASH: No valid current hashes provided for Guild {guild_id}. Cannot search.")
        _close_connection(conn); return None

    log.info(f"DB_FIND_HASH: Searching Guild {guild_id} for similar hash(es). Threshold: {threshold}")
    try:
        cursor = conn.cursor()
        query = f"SELECT hash_hex, message_id, channel_id, author_id, timestamp, media_url FROM {HASH_TABLE_NAME} WHERE guild_id = ? ORDER BY timestamp ASC"
        cursor.execute(query, (guild_id,))
        rows = cursor.fetchall()
        log.debug(f"DB_FIND_HASH: Found {len(rows)} potential rows for Guild {guild_id}.")

        if not rows: return None # No hashes stored for this guild yet

        for row in rows:
            db_hash_hex_string = row["hash_hex"]
            db_hashes_hex_list = [h for h in db_hash_hex_string.split(';') if h]

            for db_hash_hex in db_hashes_hex_list:
                try:
                    db_hash = hex_to_hash(db_hash_hex)
                    for target_hash in target_hashes:
                        difference = target_hash - db_hash
                        if difference <= threshold:
                            log.info(f"!!! DB_FIND_HASH: MATCH FOUND (Diff: {difference}) Target:{str(target_hash)} vs DB:{db_hash_hex} from MsgID: {row['message_id']} !!!")
                            original_link = f"https://discord.com/channels/{guild_id}/{row['channel_id']}/{row['message_id']}"
                            found_match = {
                                "message_id": row["message_id"], "channel_id": row["channel_id"], "author_id": row["author_id"],
                                "timestamp": row["timestamp"], "link": original_link, "similarity": difference,
                                "matched_db_hash": db_hash_hex, "matched_target_hash": str(target_hash),
                                "original_media_url": row["media_url"]
                            }
                            # Match found, return the earliest one due to ORDER BY
                            _close_connection(conn)
                            return found_match
                except ValueError: log.warning(f"DB_FIND_HASH: Invalid hash '{db_hash_hex}' in DB for msg {row['message_id']}. Skipping.")
                except Exception as e: log.error(f"DB_FIND_HASH: Error comparing hash (Target:{target_hash}/DB:{db_hash_hex}): {e}", exc_info=True)

    except Exception as e: log.error(f"DB_FIND_HASH: Unexpected error during search: {e}", exc_info=True)
    finally: _close_connection(conn)

    if not found_match: log.info("DB_FIND_HASH: Search finished. No match found.")
    return found_match


def set_alert_channel(guild_id: int, channel_id: Optional[int]):
    """Sets or clears the alert channel for a specific guild."""
    conn = _get_connection()
    if not conn: log.error(f"DB_SET_ALERT_CHANNEL: Failed connection for Guild {guild_id}."); return False
    log.info(f"DB_SET_ALERT_CHANNEL: Setting alert channel for Guild {guild_id} to {channel_id}")
    success = False
    try:
        cursor = conn.cursor()
        cursor.execute(f"INSERT OR REPLACE INTO {CONFIG_TABLE_NAME} (guild_id, alert_channel_id) VALUES (?, ?)", (guild_id, channel_id))
        conn.commit()
        success = True
    except Exception as e:
        log.error(f"DB_SET_ALERT_CHANNEL: Error setting alert channel for Guild {guild_id}: {e}", exc_info=True)
        try: conn.rollback()
        except Exception as rb_e: log.error(f"DB_SET_ALERT_CHANNEL: Rollback failed: {rb_e}")
    finally: _close_connection(conn)
    return success

def get_alert_channel(guild_id: int) -> Optional[int]:
    """Gets the configured alert channel ID for a guild. Returns None if not set."""
    conn = _get_connection()
    if not conn: log.error(f"DB_GET_ALERT_CHANNEL: Failed connection for Guild {guild_id}."); return None
    channel_id: Optional[int] = None
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT alert_channel_id FROM {CONFIG_TABLE_NAME} WHERE guild_id = ?", (guild_id,))
        row = cursor.fetchone()
        if row and row["alert_channel_id"] is not None:
            channel_id = int(row["alert_channel_id"])
    except Exception as e:
        log.error(f"DB_GET_ALERT_CHANNEL: Error getting alert channel for Guild {guild_id}: {e}", exc_info=True)
    finally: _close_connection(conn)
    log.debug(f"DB_GET_ALERT_CHANNEL: Found alert channel {channel_id} for Guild {guild_id}.")
    return channel_id


def add_whitelist_channel(guild_id: int, channel_id: int) -> bool:
    """Adds a channel to the whitelist. Returns True if added, False if already exists or error."""
    conn = _get_connection()
    if not conn: log.error(f"DB_WL_ADD: Failed connection for G:{guild_id}, C:{channel_id}."); return False
    success = False
    try:
        cursor = conn.cursor()
        cursor.execute(f"INSERT OR IGNORE INTO {WHITELIST_TABLE_NAME} (guild_id, channel_id) VALUES (?, ?)", (guild_id, channel_id))
        if cursor.rowcount > 0:
            conn.commit(); success = True
            log.info(f"DB_WL_ADD: Channel {channel_id} whitelisted for Guild {guild_id}.")
        else: log.warning(f"DB_WL_ADD: Channel {channel_id} already whitelisted for Guild {guild_id}.")
    except Exception as e:
        log.error(f"DB_WL_ADD: Error whitelisting G:{guild_id}, C:{channel_id}: {e}", exc_info=True)
        try: conn.rollback()
        except Exception as rb_e: log.error(f"DB_WL_ADD: Rollback failed: {rb_e}")
    finally: _close_connection(conn)
    return success

def remove_whitelist_channel(guild_id: int, channel_id: int) -> bool:
    """Removes a channel from the whitelist. Returns True if removed, False if not found or error."""
    conn = _get_connection()
    if not conn: log.error(f"DB_WL_REMOVE: Failed connection for G:{guild_id}, C:{channel_id}."); return False
    success = False
    try:
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {WHITELIST_TABLE_NAME} WHERE guild_id = ? AND channel_id = ?", (guild_id, channel_id))
        if cursor.rowcount > 0:
            conn.commit(); success = True
            log.info(f"DB_WL_REMOVE: Channel {channel_id} removed from whitelist for Guild {guild_id}.")
        else: log.warning(f"DB_WL_REMOVE: Channel {channel_id} not found in whitelist for Guild {guild_id}.")
    except Exception as e:
        log.error(f"DB_WL_REMOVE: Error removing G:{guild_id}, C:{channel_id} from whitelist: {e}", exc_info=True)
        try: conn.rollback()
        except Exception as rb_e: log.error(f"DB_WL_REMOVE: Rollback failed: {rb_e}")
    finally: _close_connection(conn)
    return success

def is_channel_whitelisted(guild_id: int, channel_id: int) -> bool:
    """Checks if a specific channel is whitelisted for a guild."""
    conn = _get_connection()
    if not conn: log.error(f"DB_WL_CHECK: Failed connection for G:{guild_id}, C:{channel_id}."); return False
    whitelisted = False
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT 1 FROM {WHITELIST_TABLE_NAME} WHERE guild_id = ? AND channel_id = ? LIMIT 1", (guild_id, channel_id))
        whitelisted = cursor.fetchone() is not None
    except Exception as e:
        log.error(f"DB_WL_CHECK: Error checking G:{guild_id}, C:{channel_id}: {e}", exc_info=True)
    finally: _close_connection(conn)
    #log.debug(f"DB_WL_CHECK: Channel {channel_id} Whitelisted = {whitelisted} in Guild {guild_id}.")
    return whitelisted