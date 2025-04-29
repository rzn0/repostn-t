# database.py
import sqlite3
import os
import logging
import time
# Added List, Union, Optional to typing imports
from imagehash import ImageHash, hex_to_hash
from typing import Optional, Dict, Any, List, Union

DB_DIR = "db"
DB_NAME = "repost_hashes.db" # Keep the same DB file name
DB_PATH = os.path.join(DB_DIR, DB_NAME)
HASH_TABLE_NAME = "media_hashes"
CONFIG_TABLE_NAME = "guild_config" # New table for configuration

log = logging.getLogger(__name__)

def _ensure_db_dir():
    try:
        abs_db_dir = os.path.abspath(DB_DIR)
        log.debug(f"Ensuring DB dir exists: {abs_db_dir}")
        os.makedirs(abs_db_dir, exist_ok=True)
        if not os.access(abs_db_dir, os.W_OK):
             log.error(f"!!! Directory '{abs_db_dir}' NOT WRITABLE by user {os.getuid()} !!!")
        else:
            log.debug(f"Directory '{abs_db_dir}' exists and appears writable.")
    except OSError as e:
        log.error(f"Failed create/access DB dir '{abs_db_dir}': {e}", exc_info=True)
        raise

def _get_connection() -> Optional[sqlite3.Connection]:
    abs_db_path = os.path.abspath(DB_PATH)
    log.debug(f"Attempting connect to DB: {abs_db_path}")
    try: _ensure_db_dir()
    except Exception as e: log.error(f"Cannot get DB connection: directory setup failed: {e}"); return None
    try:
        conn = sqlite3.connect(abs_db_path, timeout=10, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try: conn.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.Error as e: log.warning(f"Could not set journal_mode=WAL: {e}")
        try: conn.execute("PRAGMA busy_timeout = 5000;")
        except sqlite3.Error as e: log.warning(f"Could not set busy_timeout: {e}")
        try: conn.execute("PRAGMA foreign_keys = ON;")
        except sqlite3.Error as e: log.warning(f"Could not set foreign_keys=ON: {e}")
        log.info(f"DB connection acquired: {abs_db_path}"); return conn
    except sqlite3.Error as e: log.error(f"DB connection error to {abs_db_path}: {e}", exc_info=True); return None
    except Exception as e: log.error(f"Unexpected error connecting to DB {abs_db_path}: {e}", exc_info=True); return None

def _close_connection(conn: Optional[sqlite3.Connection]):
    if conn:
        db_path = os.path.abspath(DB_PATH);
        try: conn.close(); log.debug(f"DB connection closed: {db_path}")
        except Exception as e: log.error(f"Error closing DB connection to {db_path}: {e}", exc_info=True)

def setup_database():
    """Creates the necessary tables and indexes if they don't exist."""
    log.info("--- Running Database Setup ---"); conn = None; abs_db_path = os.path.abspath(DB_PATH)
    log.info(f"Target database file: {abs_db_path}")
    try:
        conn = _get_connection()
        if not conn: raise ConnectionError("DB_SETUP: Failed to get DB connection.")
        log.info(f"DB_SETUP: Connection successful. Checking/Creating tables...")
        conn.execute("BEGIN IMMEDIATE;")
        cursor = conn.cursor()

        # --- Hash Table ---
        log.debug(f"DB_SETUP: Executing CREATE TABLE IF NOT EXISTS {HASH_TABLE_NAME}...")
        try:
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
            log.info(f"DB_SETUP: CREATE TABLE IF NOT EXISTS {HASH_TABLE_NAME} executed.")
        except sqlite3.Error as e_create: log.critical(f"DB_SETUP: !!! FAILED CREATE TABLE {HASH_TABLE_NAME}: {e_create} !!!", exc_info=True); raise
        log.debug(f"DB_SETUP: Verifying table existence: {HASH_TABLE_NAME}...");
        try:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (HASH_TABLE_NAME,))
            table_exists = cursor.fetchone()
            if table_exists: log.info(f"DB_SETUP: >>> Verification successful: Table '{HASH_TABLE_NAME}' exists. <<<")
            else: log.critical(f"DB_SETUP: !!! Verification FAILED: Table '{HASH_TABLE_NAME}' NOT FOUND! Check errors. !!!"); raise sqlite3.OperationalError(f"Table '{HASH_TABLE_NAME}' verification failed.")
        except sqlite3.Error as e_verify: log.critical(f"DB_SETUP: !!! Error during {HASH_TABLE_NAME} verification: {e_verify} !!!", exc_info=True); raise

        # --- Config Table (New) ---
        log.debug(f"DB_SETUP: Executing CREATE TABLE IF NOT EXISTS {CONFIG_TABLE_NAME}...")
        try:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {CONFIG_TABLE_NAME} (
                    guild_id INTEGER PRIMARY KEY,
                    alert_channel_id INTEGER NULLABLE
                );
            """)
            log.info(f"DB_SETUP: CREATE TABLE IF NOT EXISTS {CONFIG_TABLE_NAME} executed.")
        except sqlite3.Error as e_create_cfg: log.critical(f"DB_SETUP: !!! FAILED CREATE TABLE {CONFIG_TABLE_NAME}: {e_create_cfg} !!!", exc_info=True); raise
        log.debug(f"DB_SETUP: Verifying table existence: {CONFIG_TABLE_NAME}...");
        try:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (CONFIG_TABLE_NAME,))
            table_exists_cfg = cursor.fetchone()
            if table_exists_cfg: log.info(f"DB_SETUP: >>> Verification successful: Table '{CONFIG_TABLE_NAME}' exists. <<<")
            else: log.critical(f"DB_SETUP: !!! Verification FAILED: Table '{CONFIG_TABLE_NAME}' NOT FOUND! Check errors. !!!"); raise sqlite3.OperationalError(f"Table '{CONFIG_TABLE_NAME}' verification failed.")
        except sqlite3.Error as e_verify_cfg: log.critical(f"DB_SETUP: !!! Error during {CONFIG_TABLE_NAME} verification: {e_verify_cfg} !!!", exc_info=True); raise


        # --- Indexes ---
        log.debug("DB_SETUP: Ensuring indexes...");
        try:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_guild_hash ON {HASH_TABLE_NAME} (guild_id);") # Index on guild_id is useful
            log.debug(f"DB_SETUP: Index 'idx_guild_id' on {HASH_TABLE_NAME} ensured.")
            # No index needed on config table as guild_id is PK
        except sqlite3.Error as e_index: log.error(f"DB_SETUP: Error ensuring indexes: {e_index}", exc_info=True); # Non-critical if index fails

        log.debug("DB_SETUP: Attempting COMMIT..."); conn.commit(); log.info("DB_SETUP: Transaction committed."); log.info("--- Database Setup Function Completed Successfully ---")
    except Exception as e: log.critical(f"--- Database Setup Function FAILED: {e} ---", exc_info=True); raise
    finally:
        if conn: _close_connection(conn); log.debug("DB_SETUP: Connection closed in finally block.")
        else: log.warning("DB_SETUP: Connection was None in finally block.")

# Modified add_hash to accept a list of hashes or a single hash
def add_hash(guild_id: int, channel_id: int, message_id: int, author_id: int,
             hashes: Union[ImageHash, List[ImageHash]], media_url: str):
    """Adds a new media hash(es) to the database. Joins lists with ';'."""
    conn = _get_connection();
    if not conn: log.error(f"DB_ADD_HASH: Failed connection for MsgID {message_id}."); return

    # Convert single hash or list of hashes to a semicolon-delimited string
    hash_hex_string: str = ""
    if isinstance(hashes, list):
        valid_hashes = [str(h) for h in hashes if h is not None]
        if not valid_hashes:
            log.warning(f"DB_ADD_HASH: No valid hashes provided for MsgID {message_id}. Not adding.")
            _close_connection(conn); return
        hash_hex_string = ";".join(valid_hashes)
        log.debug(f"DB_ADD_HASH: Adding {len(valid_hashes)} hashes for MsgID {message_id} as '{hash_hex_string}'")
    elif isinstance(hashes, ImageHash):
        hash_hex_string = str(hashes)
        log.debug(f"DB_ADD_HASH: Adding single hash for MsgID {message_id} as '{hash_hex_string}'")
    else:
        log.error(f"DB_ADD_HASH: Invalid hash type ({type(hashes)}) for MsgID {message_id}. Not adding.")
        _close_connection(conn); return

    timestamp = int(time.time())
    log.info(f"DB_ADD_HASH: Attempting add for MsgID {message_id}, Guild {guild_id}, Hash(es) '{hash_hex_string[:20]}...'")
    try:
        conn.execute("BEGIN;"); cursor = conn.cursor()
        cursor.execute(
            # Insert the potentially semicolon-delimited string
            f"INSERT INTO {HASH_TABLE_NAME} (guild_id, channel_id, message_id, author_id, hash_hex, media_url, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (guild_id, channel_id, message_id, author_id, hash_hex_string, media_url, timestamp)
        )
        conn.commit(); log.info(f"DB_ADD_HASH: Hash(es) added successfully for MsgID {message_id}.")
    except sqlite3.IntegrityError:
        log.warning(f"DB_ADD_HASH: Hash for MsgID {message_id} already exists. Ignoring duplicate.")
        try: conn.rollback(); log.debug(f"DB_ADD_HASH: Rolled back for duplicate MsgID {message_id}.")
        except sqlite3.Error as rb_e: log.error(f"DB_ADD_HASH: Rollback failed after IntegrityError: {rb_e}")
    except sqlite3.OperationalError as oe:
        log.error(f"DB_ADD_HASH: SQLite OperationalError for MsgID {message_id}: {oe}", exc_info=True)
        try: conn.rollback(); log.warning(f"DB_ADD_HASH: Rolled back transaction for MsgID {message_id} due to OperationalError.")
        except sqlite3.Error as rb_e: log.error(f"DB_ADD_HASH: Rollback failed after OperationalError for MsgID {message_id}: {rb_e}")
    except sqlite3.Error as e:
        log.error(f"DB_ADD_HASH: Error adding hash for message {message_id}: {e}", exc_info=True)
        try: log.warning(f"DB_ADD_HASH: Rolling back transaction for MsgID {message_id} due to error."); conn.rollback()
        except sqlite3.Error as rb_e: log.error(f"DB_ADD_HASH: Rollback failed after error: {rb_e}")
    except Exception as e:
        log.error(f"DB_ADD_HASH: Unexpected non-SQLite error adding hash for MsgID {message_id}: {e}", exc_info=True)
        try: conn.rollback()
        except Exception as rb_e: log.error(f"DB_ADD_HASH: Rollback failed after unexpected error: {rb_e}")
    finally: _close_connection(conn)

# Modified find_similar_hash to handle multiple input hashes and stored delimited hashes
def find_similar_hash(guild_id: int, current_hashes: Union[ImageHash, List[ImageHash]], threshold: int) -> Optional[Dict[str, Any]]:
    """
    Finds the first stored media entry in the guild where *any* of its stored hashes
    are similar (within threshold) to *any* of the provided current_hashes.
    Prioritizes the earliest match by timestamp.
    """
    conn = _get_connection();
    if not conn: log.error(f"DB_FIND_HASH: Failed connection for Guild {guild_id}."); return None
    start_time = time.monotonic(); found_match = None; rows = []

    target_hashes: List[ImageHash] = []
    if isinstance(current_hashes, list):
        target_hashes = [h for h in current_hashes if isinstance(h, ImageHash)]
    elif isinstance(current_hashes, ImageHash):
        target_hashes = [current_hashes]

    if not target_hashes:
        log.warning(f"DB_FIND_HASH: No valid current hashes provided for Guild {guild_id}. Cannot search.")
        _close_connection(conn); return None

    current_hashes_str_repr = ";".join(str(h) for h in target_hashes)
    log.info(f"DB_FIND_HASH: Searching Guild {guild_id} for hash similar to {len(target_hashes)} provided hash(es) ('{current_hashes_str_repr[:20]}...') Threshold: {threshold}")

    try:
        cursor = conn.cursor()
        query = f"SELECT hash_hex, message_id, channel_id, author_id, timestamp, media_url FROM {HASH_TABLE_NAME} WHERE guild_id = ? ORDER BY timestamp ASC"
        log.debug(f"DB_FIND_HASH: Executing query: {query} with GuildID {guild_id}")
        cursor.execute(query, (guild_id,))
        rows = cursor.fetchall(); log.debug(f"DB_FIND_HASH: Query executed. Found {len(rows)} potential rows.")

        if not rows: log.info(f"DB_FIND_HASH: No previous hashes found in Guild {guild_id}."); return None

        log.info(f"DB_FIND_HASH: Comparing {len(target_hashes)} current hash(es) against {len(rows)} existing entries.")

        for i, row in enumerate(rows):
            db_hash_hex_string = row["hash_hex"]
            db_msg_id = row["message_id"]
            db_channel_id = row["channel_id"]
            db_author_id = row["author_id"]
            db_timestamp = row["timestamp"]
            db_media_url = row["media_url"] # Get media URL

            db_hashes_hex_list = [h for h in db_hash_hex_string.split(';') if h] # Filter empty strings

            log.debug(f"DB_FIND_HASH [{i+1}/{len(rows)}]: Checking MsgID:{db_msg_id} (Timestamp:{db_timestamp}) StoredHashes: {len(db_hashes_hex_list)}")

            match_found_for_row = False
            min_difference = float('inf')
            matched_db_hash_hex = None
            matched_target_hash_str = None

            for db_hash_hex in db_hashes_hex_list:
                try:
                    db_hash = hex_to_hash(db_hash_hex)
                    for target_hash in target_hashes:
                        difference = target_hash - db_hash # Hamming distance
                        log.debug(f"  Comparing Target:{str(target_hash)} vs DB:{db_hash_hex} -> Diff: {difference}")

                        if difference <= threshold:
                            log.info(f"!!! DB_FIND_HASH: MATCH FOUND (Diff: {difference} <= Threshold: {threshold}) between Target:{str(target_hash)} and DB:{db_hash_hex} from MsgID: {db_msg_id} !!!")
                            match_found_for_row = True
                            if difference < min_difference:
                                min_difference = difference
                                matched_db_hash_hex = db_hash_hex
                                matched_target_hash_str = str(target_hash)
                            break # Stop comparing other target hashes to this db_hash
                    if match_found_for_row:
                         break # Stop comparing this row's other db_hashes
                except ValueError: log.warning(f"DB_FIND_HASH: Invalid hash '{db_hash_hex}' in DB for msg {db_msg_id}. Skipping.")
                except TypeError as te: log.error(f"DB_FIND_HASH: TypeError comparing hashes (Target:{target_hash}/DB:{db_hash_hex}, MsgID:{db_msg_id}): {te}")
                except Exception as e: log.error(f"DB_FIND_HASH: Error comparing hash (Target:{target_hash}/DB:{db_hash_hex}, MsgID:{db_msg_id}): {e}", exc_info=True)

            if match_found_for_row:
                original_link = f"https://discord.com/channels/{guild_id}/{db_channel_id}/{db_msg_id}"
                found_match = {
                    "message_id": db_msg_id,
                    "channel_id": db_channel_id,
                    "author_id": db_author_id,
                    "timestamp": db_timestamp,
                    "link": original_link,
                    "similarity": min_difference,
                    "matched_db_hash": matched_db_hash_hex,
                    "matched_target_hash": matched_target_hash_str,
                    "original_media_url": db_media_url
                }
                break # Stop iterating through DB rows (ORDER BY timestamp ASC ensures this is the earliest)

    except sqlite3.OperationalError as oe: log.error(f"DB_FIND_HASH: SQLite OperationalError during search: {oe}", exc_info=True)
    except sqlite3.Error as e: log.error(f"DB_FIND_HASH: Database error during search: {e}", exc_info=True)
    except Exception as e: log.error(f"DB_FIND_HASH: Unexpected non-SQLite error during search: {e}", exc_info=True)
    finally: _close_connection(conn)

    end_time = time.monotonic(); processing_time = end_time - start_time
    if found_match:
        log.info(f"DB_FIND_HASH: Search finished. Earliest match found: MsgID {found_match['message_id']} (Similarity: {found_match['similarity']}). Time: {processing_time:.4f}s")
    else:
        log.info(f"DB_FIND_HASH: Search finished. No match found. Time: {processing_time:.4f}s")
    return found_match


# --- Functions for Guild Configuration --- THIS SECTION IS NEEDED ---

def set_alert_channel(guild_id: int, channel_id: Optional[int]):
    """Sets or clears the alert channel for a specific guild."""
    conn = _get_connection()
    if not conn:
        log.error(f"DB_SET_ALERT_CHANNEL: Failed connection for Guild {guild_id}.")
        return False

    log.info(f"DB_SET_ALERT_CHANNEL: Setting alert channel for Guild {guild_id} to {channel_id}")
    success = False
    try:
        conn.execute("BEGIN IMMEDIATE;")
        cursor = conn.cursor()
        # Use INSERT OR REPLACE to handle both new and existing guild entries easily
        cursor.execute(
            f"INSERT OR REPLACE INTO {CONFIG_TABLE_NAME} (guild_id, alert_channel_id) VALUES (?, ?)",
            (guild_id, channel_id) # channel_id can be None, which translates to NULL
        )
        conn.commit()
        log.info(f"DB_SET_ALERT_CHANNEL: Alert channel updated successfully for Guild {guild_id}.")
        success = True
    except sqlite3.OperationalError as oe:
        log.error(f"DB_SET_ALERT_CHANNEL: SQLite OperationalError for Guild {guild_id}: {oe}", exc_info=True)
        try: conn.rollback()
        except sqlite3.Error as rb_e: log.error(f"DB_SET_ALERT_CHANNEL: Rollback failed after OperationalError: {rb_e}")
    except sqlite3.Error as e:
        log.error(f"DB_SET_ALERT_CHANNEL: Error setting alert channel for Guild {guild_id}: {e}", exc_info=True)
        try: conn.rollback()
        except sqlite3.Error as rb_e: log.error(f"DB_SET_ALERT_CHANNEL: Rollback failed after error: {rb_e}")
    except Exception as e:
        log.error(f"DB_SET_ALERT_CHANNEL: Unexpected error setting alert channel for Guild {guild_id}: {e}", exc_info=True)
        try: conn.rollback()
        except Exception as rb_e: log.error(f"DB_SET_ALERT_CHANNEL: Rollback failed after unexpected error: {rb_e}")
    finally:
        _close_connection(conn)
    return success

def get_alert_channel(guild_id: int) -> Optional[int]:
    """Gets the configured alert channel ID for a guild. Returns None if not set."""
    conn = _get_connection()
    if not conn:
        log.error(f"DB_GET_ALERT_CHANNEL: Failed connection for Guild {guild_id}.")
        return None

    log.debug(f"DB_GET_ALERT_CHANNEL: Querying alert channel for Guild {guild_id}")
    channel_id: Optional[int] = None
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT alert_channel_id FROM {CONFIG_TABLE_NAME} WHERE guild_id = ?",
            (guild_id,)
        )
        row = cursor.fetchone()
        if row and row["alert_channel_id"] is not None:
            channel_id = int(row["alert_channel_id"])
            log.info(f"DB_GET_ALERT_CHANNEL: Found configured alert channel {channel_id} for Guild {guild_id}.")
        else:
            log.info(f"DB_GET_ALERT_CHANNEL: No alert channel configured for Guild {guild_id}.")
    except sqlite3.OperationalError as oe:
        log.error(f"DB_GET_ALERT_CHANNEL: SQLite OperationalError for Guild {guild_id}: {oe}", exc_info=True)
    except sqlite3.Error as e:
        log.error(f"DB_GET_ALERT_CHANNEL: Database error getting alert channel for Guild {guild_id}: {e}", exc_info=True)
    except Exception as e:
        log.error(f"DB_GET_ALERT_CHANNEL: Unexpected error getting alert channel for Guild {guild_id}: {e}", exc_info=True)
    finally:
        _close_connection(conn)
    return channel_id


# --- Code to run setup if script is executed directly ---
if __name__ == "__main__":
    print("Attempting to run database setup directly...")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-8s] [%(name)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    main_log = logging.getLogger(__name__)
    main_log.info("Direct execution: Basic logging configured.")
    try:
        setup_database()
        print("\n--- Database setup script finished successfully. ---")
        print(f"Database file should be located at: {os.path.abspath(DB_PATH)}")
        print(f"You can verify the tables using: sqlite3 {DB_PATH}")
        print(f"> .schema {HASH_TABLE_NAME}")
        print(f"> .schema {CONFIG_TABLE_NAME}") # Make sure this table exists
    except Exception as e:
        print(f"\n--- Database setup script FAILED: {e} ---")
        main_log.exception("Error during direct database setup:")
        print("Please check the logs above for details (especially permission errors or SQLite errors).")
        import sys
        sys.exit(1)
