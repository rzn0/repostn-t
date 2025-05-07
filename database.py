# database.py
import sqlite3
import os
import logging
import time
import imagehash # Ensure this is imported
from imagehash import ImageHash, hex_to_hash
from typing import Optional, Dict, Any, List, Union, Set, Tuple

DB_DIR = "db"
DB_NAME = "repost_hashes.db"
DB_PATH = os.path.join(DB_DIR, DB_NAME)
HASH_TABLE_NAME = "media_hashes"
CONFIG_TABLE_NAME = "guild_config"
WHITELIST_TABLE_NAME = "channel_whitelist"

log = logging.getLogger(__name__)

def _ensure_db_dir():
    try:
        abs_db_dir = os.path.abspath(DB_DIR)
        os.makedirs(abs_db_dir, exist_ok=True)
    except OSError as e:
        log.error(f"Failed create/access DB dir '{abs_db_dir}': {e}", exc_info=True); raise

def _get_connection() -> Optional[sqlite3.Connection]:
    abs_db_path = os.path.abspath(DB_PATH)
    try:
        _ensure_db_dir()
        conn = sqlite3.connect(abs_db_path, timeout=10, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try: conn.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.Error as e: log.warning(f"Could not set journal_mode=WAL: {e}")
        try: conn.execute("PRAGMA busy_timeout = 5000;")
        except sqlite3.Error as e: log.warning(f"Could not set busy_timeout: {e}")
        try: conn.execute("PRAGMA foreign_keys = ON;")
        except sqlite3.Error as e: log.warning(f"Could not set foreign_keys=ON: {e}")
        log.debug(f"DB connection acquired: {abs_db_path}"); return conn
    except Exception as e: log.error(f"Error connecting to DB {abs_db_path}: {e}", exc_info=True); return None

def _close_connection(conn: Optional[sqlite3.Connection]):
    if conn:
        try: conn.close(); log.debug("DB connection closed.")
        except Exception as e: log.error(f"Error closing DB connection: {e}", exc_info=True)

def setup_database():
    log.info("--- Running Database Setup ---")
    conn = _get_connection()
    if not conn: raise ConnectionError("DB_SETUP: Failed to get DB connection.")
    try:
        log.info("DB_SETUP: Checking/Creating tables...")
        conn.execute("BEGIN IMMEDIATE;")
        cursor = conn.cursor()
        cursor.execute(f""" CREATE TABLE IF NOT EXISTS {HASH_TABLE_NAME} ( id INTEGER PRIMARY KEY AUTOINCREMENT, guild_id INTEGER NOT NULL, channel_id INTEGER NOT NULL, message_id INTEGER NOT NULL UNIQUE, author_id INTEGER NOT NULL, hash_hex TEXT NOT NULL, media_url TEXT, timestamp INTEGER NOT NULL ); """)
        cursor.execute(f""" CREATE TABLE IF NOT EXISTS {CONFIG_TABLE_NAME} ( guild_id INTEGER PRIMARY KEY, alert_channel_id INTEGER NULLABLE ); """)
        cursor.execute(f""" CREATE TABLE IF NOT EXISTS {WHITELIST_TABLE_NAME} ( guild_id INTEGER NOT NULL, channel_id INTEGER NOT NULL, PRIMARY KEY (guild_id, channel_id) ); """)
        log.info("DB_SETUP: Base tables ensured.")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_guild_hash ON {HASH_TABLE_NAME} (guild_id);")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_whitelist_guild ON {WHITELIST_TABLE_NAME} (guild_id);")
        log.info("DB_SETUP: Indexes ensured.")
        conn.commit(); log.info("DB_SETUP: Transaction committed."); log.info("--- Database Setup Function Completed Successfully ---")
    except Exception as e:
        log.critical(f"--- Database Setup Function FAILED: {e} ---", exc_info=True)
        if conn:
            try:
                conn.rollback()
                log.warning("DB_SETUP: Attempted rollback due to setup failure.")
            except Exception as rb_e:
                log.error(f"DB_SETUP: Rollback attempt failed: {rb_e}")
        raise
    finally:
        _close_connection(conn)
        log.debug("DB_SETUP: Connection closed in finally block.")


def add_hash(guild_id: int, channel_id: int, message_id: int, author_id: int,
             hashes: Union[ImageHash, List[ImageHash]], media_url: str):
    conn = _get_connection();
    if not conn: log.error(f"DB_ADD_HASH: Failed connection MsgID {message_id}."); return
    hash_hex_string: str = ""
    if isinstance(hashes, list):
        valid_hashes = [str(h) for h in hashes if h is not None]
        if not valid_hashes: _close_connection(conn); return
        hash_hex_string = ";".join(valid_hashes)
    elif isinstance(hashes, ImageHash): hash_hex_string = str(hashes)
    else: log.error(f"DB_ADD_HASH: Invalid hash type MsgID {message_id}."); _close_connection(conn); return
    timestamp = int(time.time())
    try:
        conn.execute("BEGIN;")
        cursor = conn.cursor()
        cursor.execute(f"INSERT INTO {HASH_TABLE_NAME} (guild_id, channel_id, message_id, author_id, hash_hex, media_url, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (guild_id, channel_id, message_id, author_id, hash_hex_string, media_url, timestamp))
        conn.commit(); log.info(f"DB_ADD_HASH: Added MsgID {message_id}.")
    except sqlite3.IntegrityError:
        log.warning(f"DB_ADD_HASH: Hash MsgID {message_id} already exists.")
    except Exception as e:
        log.error(f"DB_ADD_HASH: Error MsgID {message_id}: {e}", exc_info=True)
        if conn:
            try:
                conn.rollback()
                log.warning(f"DB_ADD_HASH: Rolled back transaction for MsgID {message_id} due to error.")
            except Exception as rb_e:
                 log.error(f"DB_ADD_HASH: Rollback failed: {rb_e}")
    finally: _close_connection(conn)

def find_similar_hash(guild_id: int, current_hashes: Union[ImageHash, List[ImageHash]], threshold: int) -> Optional[Dict[str, Any]]:
    conn = _get_connection();
    if not conn: log.error(f"DB_FIND_HASH: Failed connection Guild {guild_id}."); return None
    found_match = None
    target_hashes: List[ImageHash] = []
    if isinstance(current_hashes, list): target_hashes = [h for h in current_hashes if isinstance(h, ImageHash)]
    elif isinstance(current_hashes, ImageHash): target_hashes = [current_hashes]
    if not target_hashes: log.warning(f"DB_FIND_HASH: No valid hashes Guild {guild_id}."); _close_connection(conn); return None
    log.info(f"DB_FIND_HASH: Searching Guild {guild_id}. Threshold: {threshold}")
    try:
        cursor = conn.cursor()
        query = f"SELECT hash_hex, message_id, channel_id, author_id, timestamp, media_url FROM {HASH_TABLE_NAME} WHERE guild_id = ? ORDER BY timestamp ASC"
        cursor.execute(query, (guild_id,))
        rows = cursor.fetchall()
        if not rows: _close_connection(conn); return None
        for row in rows:
            db_hashes_hex_list = [h for h in row["hash_hex"].split(';') if h]
            for db_hash_hex in db_hashes_hex_list:
                target_hash = None
                try:
                    db_hash = hex_to_hash(db_hash_hex)
                    for target_hash in target_hashes:
                        difference = target_hash - db_hash
                        if difference <= threshold:
                            log.info(f"!!! DB_FIND_HASH: MATCH FOUND (Diff: {difference}) Target:{str(target_hash)} vs DB:{db_hash_hex} from MsgID: {row['message_id']} !!!")
                            original_link = f"https://discord.com/channels/{guild_id}/{row['channel_id']}/{row['message_id']}"
                            found_match = {"message_id": row["message_id"], "channel_id": row["channel_id"], "author_id": row["author_id"], "timestamp": row["timestamp"], "link": original_link, "similarity": difference, "matched_db_hash": db_hash_hex, "matched_target_hash": str(target_hash), "original_media_url": row["media_url"]}
                            _close_connection(conn); return found_match
                except ValueError: log.warning(f"DB_FIND_HASH: Invalid hash '{db_hash_hex}' in DB msg {row['message_id']}.")
                except Exception as e: log.error(f"DB_FIND_HASH: Error comparing DB hash '{db_hash_hex}' (MsgID {row['message_id']}): {e}", exc_info=True)
    except Exception as e: log.error(f"DB_FIND_HASH: Unexpected error during search: {e}", exc_info=True)
    finally: _close_connection(conn)
    if not found_match: log.info("DB_FIND_HASH: Search finished. No match found.")
    return found_match

def set_alert_channel(guild_id: int, channel_id: Optional[int]):
    conn = _get_connection()
    if not conn: log.error(f"DB_SET_ALERT_CHANNEL: Failed connection Guild {guild_id}."); return False
    success = False
    try:
        conn.execute("BEGIN IMMEDIATE;")
        cursor = conn.cursor()
        cursor.execute(f"INSERT OR REPLACE INTO {CONFIG_TABLE_NAME} (guild_id, alert_channel_id) VALUES (?, ?)", (guild_id, channel_id))
        conn.commit(); success = True
        log.info(f"DB_SET_ALERT_CHANNEL: Alert channel Guild {guild_id} set to {channel_id}.")
    except Exception as e:
        log.error(f"DB_SET_ALERT_CHANNEL: Error G:{guild_id}: {e}", exc_info=True)
        if conn:
            try:
                conn.rollback()
                log.warning(f"DB_SET_ALERT_CHANNEL: Rolled back G:{guild_id}.")
            except Exception as rb_e:
                log.error(f"DB_SET_ALERT_CHANNEL: Rollback failed: {rb_e}")
    finally: _close_connection(conn)
    return success

def get_alert_channel(guild_id: int) -> Optional[int]:
    conn = _get_connection()
    if not conn: log.error(f"DB_GET_ALERT_CHANNEL: Failed connection Guild {guild_id}."); return None
    channel_id: Optional[int] = None
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT alert_channel_id FROM {CONFIG_TABLE_NAME} WHERE guild_id = ?", (guild_id,))
        row = cursor.fetchone()
        if row and row["alert_channel_id"] is not None: channel_id = int(row["alert_channel_id"])
    except Exception as e: log.error(f"DB_GET_ALERT_CHANNEL: Error G:{guild_id}: {e}", exc_info=True)
    finally: _close_connection(conn)
    return channel_id

def add_whitelist_channel(guild_id: int, channel_id: int) -> bool:
    conn = _get_connection()
    if not conn: log.error(f"DB_WL_ADD: Failed connection G:{guild_id}, C:{channel_id}."); return False
    success = False
    try:
        conn.execute("BEGIN IMMEDIATE;")
        cursor = conn.cursor()
        cursor.execute(f"INSERT OR IGNORE INTO {WHITELIST_TABLE_NAME} (guild_id, channel_id) VALUES (?, ?)", (guild_id, channel_id))
        if cursor.rowcount > 0:
            conn.commit(); success = True; log.info(f"DB_WL_ADD: Whitelisted C:{channel_id} G:{guild_id}.")
        else:
            log.warning(f"DB_WL_ADD: Already whitelisted C:{channel_id} G:{guild_id}.")
            # No need to rollback if IGNORE worked as intended
    except Exception as e:
        log.error(f"DB_WL_ADD: Error G:{guild_id}, C:{channel_id}: {e}", exc_info=True)
        if conn: # Check connection before rollback attempt
            try:
                conn.rollback()
                log.warning(f"DB_WL_ADD: Rolled back G:{guild_id}, C:{channel_id}.")
            except Exception as rb_e:
                log.error(f"DB_WL_ADD: Rollback failed: {rb_e}")
    finally: _close_connection(conn)
    return success

def remove_whitelist_channel(guild_id: int, channel_id: int) -> bool:
    conn = _get_connection()
    if not conn: log.error(f"DB_WL_REMOVE: Failed connection G:{guild_id}, C:{channel_id}."); return False
    success = False
    try:
        conn.execute("BEGIN IMMEDIATE;")
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {WHITELIST_TABLE_NAME} WHERE guild_id = ? AND channel_id = ?", (guild_id, channel_id))
        if cursor.rowcount > 0:
            conn.commit(); success = True; log.info(f"DB_WL_REMOVE: Removed C:{channel_id} G:{guild_id}.")
        else:
            log.warning(f"DB_WL_REMOVE: Not found C:{channel_id} G:{guild_id}.")
            conn.rollback() # Rollback if nothing was deleted
    except Exception as e:
        log.error(f"DB_WL_REMOVE: Error G:{guild_id}, C:{channel_id}: {e}", exc_info=True)
        if conn:
             try:
                 conn.rollback()
                 log.warning(f"DB_WL_REMOVE: Rolled back G:{guild_id}, C:{channel_id}.")
             except Exception as rb_e:
                 log.error(f"DB_WL_REMOVE: Rollback failed: {rb_e}")
    finally: _close_connection(conn)
    return success

def is_channel_whitelisted(guild_id: int, channel_id: int) -> bool:
    conn = _get_connection()
    if not conn: log.error(f"DB_WL_CHECK: Failed connection G:{guild_id}, C:{channel_id}."); return False
    whitelisted = False
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT 1 FROM {WHITELIST_TABLE_NAME} WHERE guild_id = ? AND channel_id = ? LIMIT 1", (guild_id, channel_id))
        whitelisted = cursor.fetchone() is not None
    except Exception as e: log.error(f"DB_WL_CHECK: Error G:{guild_id}, C:{channel_id}: {e}", exc_info=True)
    finally: _close_connection(conn)
    return whitelisted

if __name__ == "__main__":
    print("Attempting to run database setup directly...")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] [%(name)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    main_log = logging.getLogger(__name__)
    main_log.info("Direct execution: Basic logging configured.")
    try:
        setup_database()
        print("\n--- Database setup script finished successfully. ---")
        print(f"DB file: {os.path.abspath(DB_PATH)}")
        print(f"Verify tables: sqlite3 {DB_PATH} \".schema\"")
    except Exception as e:
        print(f"\n--- Database setup script FAILED: {e} ---")
        main_log.exception("Error during direct database setup:")
        import sys; sys.exit(1)