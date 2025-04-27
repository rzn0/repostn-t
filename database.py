# database.py
import sqlite3
import os
import logging
import time
from imagehash import ImageHash, hex_to_hash
from typing import Optional, Dict, Any

DB_DIR = "db"
DB_NAME = "repost_hashes.db"
DB_PATH = os.path.join(DB_DIR, DB_NAME)
TABLE_NAME = "media_hashes"

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
        # isolation_level=None enables autocommit mode, manage transactions explicitly
        # check_same_thread=False needed if using executors heavily
        conn = sqlite3.connect(abs_db_path, timeout=10, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try: conn.execute("PRAGMA journal_mode=WAL;") # WAL for better concurrency
        except sqlite3.Error as e: log.warning(f"Could not set journal_mode=WAL: {e}")
        try: conn.execute("PRAGMA busy_timeout = 5000;") # Wait 5s if locked
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
    """Creates the necessary table and indexes if they don't exist."""
    log.info("--- Running Database Setup ---"); conn = None; abs_db_path = os.path.abspath(DB_PATH)
    log.info(f"Target database file: {abs_db_path}")
    try:
        conn = _get_connection()
        if not conn: raise ConnectionError("DB_SETUP: Failed to get DB connection.")
        log.info(f"DB_SETUP: Connection successful. Checking/Creating table '{TABLE_NAME}'...")
        conn.execute("BEGIN IMMEDIATE;")
        log.debug(f"DB_SETUP: Executing CREATE TABLE IF NOT EXISTS {TABLE_NAME}...")
        try:
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
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
            log.info(f"DB_SETUP: CREATE TABLE IF NOT EXISTS {TABLE_NAME} executed.")
        except sqlite3.Error as e_create: log.critical(f"DB_SETUP: !!! FAILED CREATE TABLE: {e_create} !!!", exc_info=True); raise
        log.debug("DB_SETUP: Verifying table existence...");
        try:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (TABLE_NAME,))
            table_exists = cursor.fetchone()
            if table_exists: log.info(f"DB_SETUP: >>> Verification successful: Table '{TABLE_NAME}' exists. <<<")
            else: log.critical(f"DB_SETUP: !!! Verification FAILED: Table '{TABLE_NAME}' NOT FOUND! Check errors. !!!"); raise sqlite3.OperationalError(f"Table '{TABLE_NAME}' verification failed.")
        except sqlite3.Error as e_verify: log.critical(f"DB_SETUP: !!! Error during table verification: {e_verify} !!!", exc_info=True); raise
        # --- Simplified: Index creation removed temporarily for debugging setup ---
        # log.info("DB_SETUP: Ensuring indexes...");
        # try:
        #     cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_guild_hash ON {TABLE_NAME} (guild_id);")
        #     log.debug(f"DB_SETUP: Index 'idx_guild_hash' ensured.")
        #     cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_timestamp ON {TABLE_NAME} (timestamp);")
        #     log.debug(f"DB_SETUP: Index 'idx_timestamp' ensured.")
        # except sqlite3.Error as e_index: log.error(f"DB_SETUP: Error ensuring indexes: {e_index}", exc_info=True); raise

        log.debug("DB_SETUP: Attempting COMMIT..."); conn.commit(); log.info("DB_SETUP: Transaction committed."); log.info("--- Database Setup Function Completed Successfully ---")
    except Exception as e: log.critical(f"--- Database Setup Function FAILED: {e} ---", exc_info=True); raise
    finally:
        if conn: _close_connection(conn); log.debug("DB_SETUP: Connection closed in finally block.")
        else: log.warning("DB_SETUP: Connection was None in finally block.")

def add_hash(guild_id: int, channel_id: int, message_id: int, author_id: int, hash_obj: ImageHash, media_url: str):
    """Adds a new media hash to the database."""
    conn = _get_connection();
    if not conn: log.error(f"DB_ADD: Failed connection for MsgID {message_id}."); return
    hash_hex = str(hash_obj); timestamp = int(time.time())
    log.info(f"DB_ADD: Attempting add for MsgID {message_id}, Guild {guild_id}, Hash {hash_hex}")
    try:
        conn.execute("BEGIN;"); cursor = conn.cursor()
        cursor.execute(
            f"INSERT INTO {TABLE_NAME} (guild_id, channel_id, message_id, author_id, hash_hex, media_url, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (guild_id, channel_id, message_id, author_id, hash_hex, media_url, timestamp)
        )
        conn.commit(); log.info(f"DB_ADD: Hash added successfully for MsgID {message_id}.")
    except sqlite3.IntegrityError:
        log.warning(f"DB_ADD: Hash for MsgID {message_id} already exists. Ignoring duplicate.")
        try: conn.rollback(); log.debug(f"DB_ADD: Rolled back for duplicate MsgID {message_id}.")
        except sqlite3.Error as rb_e: log.error(f"DB_ADD: Rollback failed after IntegrityError: {rb_e}")
    except sqlite3.OperationalError as oe:
        log.error(f"DB_ADD: SQLite OperationalError for MsgID {message_id}: {oe}", exc_info=True)
        try: conn.rollback(); log.warning(f"DB_ADD: Rolled back transaction for MsgID {message_id} due to OperationalError.")
        except sqlite3.Error as rb_e: log.error(f"DB_ADD: Rollback failed after OperationalError for MsgID {message_id}: {rb_e}")
    except sqlite3.Error as e:
        log.error(f"DB_ADD: Error adding hash for message {message_id}: {e}", exc_info=True)
        try: log.warning(f"DB_ADD: Rolling back transaction for MsgID {message_id} due to error."); conn.rollback()
        except sqlite3.Error as rb_e: log.error(f"DB_ADD: Rollback failed after error: {rb_e}")
    except Exception as e:
        log.error(f"DB_ADD: Unexpected non-SQLite error adding hash for MsgID {message_id}: {e}", exc_info=True)
        try: conn.rollback()
        except Exception as rb_e: log.error(f"DB_ADD: Rollback failed after unexpected error: {rb_e}")
    finally: _close_connection(conn)

def find_similar_hash(guild_id: int, current_hash: ImageHash, threshold: int) -> Optional[Dict[str, Any]]:
    """Finds the first similar hash in the guild based on Hamming distance."""
    conn = _get_connection();
    if not conn: log.error(f"DB_FIND: Failed connection for Guild {guild_id}."); return None
    start_time = time.monotonic(); found_match = None; rows = []; current_hash_str = str(current_hash)
    log.info(f"DB_FIND: Searching Guild {guild_id} for hash similar to {current_hash_str} (Threshold: {threshold})")
    try:
        cursor = conn.cursor()
        # ORDER BY timestamp ASC is crucial to find the *earliest* repost match
        query = f"SELECT hash_hex, message_id, channel_id, author_id, timestamp FROM {TABLE_NAME} WHERE guild_id = ? ORDER BY timestamp ASC"
        log.debug(f"DB_FIND: Executing query: {query} with GuildID {guild_id}")
        cursor.execute(query, (guild_id,))
        rows = cursor.fetchall(); log.debug(f"DB_FIND: Query executed. Found {len(rows)} potential rows.")
        if not rows: log.info(f"DB_FIND: No previous hashes found in Guild {guild_id}."); return None
        log.info(f"DB_FIND: Comparing hash {current_hash_str} against {len(rows)} existing hashes.")
        for i, row in enumerate(rows):
            db_hash_hex = row["hash_hex"]
            db_msg_id = row["message_id"]
            try:
                db_hash = hex_to_hash(db_hash_hex)
                difference = current_hash - db_hash # Hamming distance
                log.debug(f"DB_FIND [{i+1}/{len(rows)}]: Comparing Current:{current_hash_str} vs DB:{db_hash_hex} (MsgID:{db_msg_id}) -> Diff: {difference}")
                log.debug(f"DB_FIND [{i+1}/{len(rows)}]: Checking if diff ({difference}) <= threshold ({threshold})")
                if difference <= threshold:
                    similarity = difference; log.info(f"!!! DB_FIND: MATCH FOUND (Diff: {similarity} <= Threshold: {threshold}) !!! Original MsgID: {db_msg_id}")
                    original_link = f"https://discord.com/channels/{guild_id}/{row['channel_id']}/{db_msg_id}"
                    found_match = {"message_id": db_msg_id, "channel_id": row["channel_id"], "author_id": row["author_id"], "timestamp": row["timestamp"], "link": original_link, "similarity": similarity, "db_hash_hex": db_hash_hex}; break
            except ValueError: log.warning(f"DB_FIND: Invalid hash '{db_hash_hex}' in DB for msg {db_msg_id}. Skipping.")
            except TypeError as te: log.error(f"DB_FIND: TypeError comparing hashes (MsgID: {db_msg_id}): {te}", exc_info=True); continue
            except Exception as e: log.error(f"DB_FIND: Error comparing hash (MsgID: {db_msg_id}): {e}", exc_info=True); continue
    except sqlite3.OperationalError as oe: log.error(f"DB_FIND: SQLite OperationalError during search: {oe}", exc_info=True); pass
    except sqlite3.Error as e: log.error(f"DB_FIND: Database error during search: {e}", exc_info=True)
    except Exception as e: log.error(f"DB_FIND: Unexpected non-SQLite error during search: {e}", exc_info=True)
    finally: _close_connection(conn)
    end_time = time.monotonic(); processing_time = end_time - start_time
    if found_match: log.info(f"DB_FIND: Search finished. Found match: {found_match['message_id']}. Time: {processing_time:.4f}s")
    else: log.info(f"DB_FIND: Search finished. No match found. Time: {processing_time:.4f}s")
    return found_match


# --- Code to run setup if script is executed directly ---
if __name__ == "__main__":
    print("Attempting to run database setup directly...")
    # Setup basic logging ONLY when run directly for visibility
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-8s] [%(name)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    main_log = logging.getLogger(__name__) # Get logger for this block
    main_log.info("Direct execution: Basic logging configured.")
    try:
        setup_database()
        print("\n--- Database setup script finished successfully. ---")
        print(f"Database file should be located at: {os.path.abspath(DB_PATH)}")
        print("You can verify the table using: sqlite3 db/repost_hashes.db \".schema media_hashes\"")
    except Exception as e:
        print(f"\n--- Database setup script FAILED: {e} ---")
        main_log.exception("Error during direct database setup:")
        print("Please check the logs above for details (especially permission errors or SQLite errors).")
        sys.exit(1) # Exit with error code if setup fails when run directly
