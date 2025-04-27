# repostBot.py
import nextcord
from nextcord.ext import commands
import os
import io
import logging
import asyncio
import aiohttp
from PIL import Image, UnidentifiedImageError, ImageSequence
import imagehash
from dotenv import load_dotenv
import time
# Added List, Tuple to typing imports
from typing import Optional, Union, Dict, Any, List, Tuple
import sys
import signal
import tempfile
import numpy as np
import shutil

# --- OpenCV Import ---
_initial_log = logging.getLogger("InitialSetup")
try:
    import cv2
    OPENCV_AVAILABLE = True
    _initial_log.info("OpenCV (cv2) imported successfully. Video processing ENABLED.")
except ImportError:
    _initial_log.warning("OpenCV (cv2) library not found or failed to import. Video processing will be DISABLED. Install with 'pip install opencv-python'.")
    OPENCV_AVAILABLE = False
    cv2 = None
except Exception as cv2_import_err:
    _initial_log.warning(f"An error occurred during OpenCV (cv2) import: {cv2_import_err}. Video processing DISABLED.")
    OPENCV_AVAILABLE = False
    cv2 = None

# --- Local Imports ---
try: import database; _initial_log.debug("Database imported successfully.")
except Exception as e: print(f"CRITICAL DB Import Error: {e}"); sys.exit(1)

# --- Configuration Loading ---
load_dotenv()
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
try:
    HASH_SIZE = int(os.getenv("HASH_SIZE", "8")); SIMILARITY_THRESHOLD = int(os.getenv("SIMILARITY_THRESHOLD", "5"))
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "30")); LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
    BOT_COMMAND_PREFIX = os.getenv("BOT_PREFIX", "!")
    # New config for black frame detection
    BLACK_FRAME_THRESHOLD = int(os.getenv("BLACK_FRAME_THRESHOLD", "10")) # Lower = stricter black detection
except Exception as e:
    print(f"ERROR parsing .env: {e}"); HASH_SIZE=8; SIMILARITY_THRESHOLD=5; MAX_FILE_SIZE_MB=30; LOG_LEVEL_STR="INFO"; BOT_COMMAND_PREFIX="!"; BLACK_FRAME_THRESHOLD = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB*1024*1024
SUPPORTED_IMAGE_EXTENSIONS = ('.png','.jpg','.jpeg','.gif','.bmp','.webp')
SUPPORTED_VIDEO_EXTENSIONS = ('.mp4', '.webm', '.mov', '.avi', '.mkv', '.flv', '.wmv')
USER_AGENT = "DiscordBot RepostDetector (v1.14 - MultiFrameVideo)" # Version bump

# --- Logging Setup ---
LOG_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}; LOG_LEVEL = LOG_LEVELS.get(LOG_LEVEL_STR, logging.INFO)
LOG_FORMAT = '%(asctime)s [%(levelname)-8s] [%(name)s]: %(message)s'; LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
log_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT); log_handler = logging.StreamHandler(sys.stdout); log_handler.setFormatter(log_formatter)
logging.getLogger("aiohttp").setLevel(logging.WARNING); logging.getLogger("moviepy").setLevel(logging.CRITICAL + 1); logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("nextcord.gateway").setLevel(logging.INFO); logging.getLogger("nextcord.client").setLevel(logging.INFO)
root_logger = logging.getLogger(); root_logger.setLevel(LOG_LEVEL);
for handler in root_logger.handlers[:]: root_logger.removeHandler(handler);
root_logger.addHandler(log_handler); log = logging.getLogger(__name__)
def log_config(): # Add new config item
    config_log = logging.getLogger("ConfigLoader"); config_log.info("-" * 40); config_log.info(" Bot Configuration Loaded:"); config_log.info("-" * 40); config_log.info(f"  Log Level           : {LOG_LEVEL_STR} ({LOG_LEVEL})"); config_log.info(f"  Hash Size           : {HASH_SIZE}"); config_log.info(f"  Similarity Threshold: {SIMILARITY_THRESHOLD}"); config_log.info(f"  Max File Size (MB)  : {MAX_FILE_SIZE_MB}"); config_log.info(f"  Bot Prefix          : {BOT_COMMAND_PREFIX}"); config_log.info(f"  Black Frame Thresh  : {BLACK_FRAME_THRESHOLD}"); config_log.info(f"  Supported Images    : {' '.join(SUPPORTED_IMAGE_EXTENSIONS)}"); config_log.info(f"  Supported Videos    : {' '.join(SUPPORTED_VIDEO_EXTENSIONS)}"); config_log.info(f"  OpenCV Available    : {OPENCV_AVAILABLE}"); config_log.info("-" * 40)

# --- Bot Class Definition ---
class RepostBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        self._session_ready_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self._start_time = time.monotonic()
        log.info(f"RepostBot __init__: Instance ID: {id(self)}. Initializing _http_session to None.")

    @property
    async def http_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._http_session is None or self._http_session.closed:
                log.info("HTTP_SESSION_PROP: Creating new session...")
                try: self._http_session = aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}); log.info(f"HTTP_SESSION_PROP: New session CREATED. ID: {id(self._http_session)}"); self._session_ready_event.set()
                except Exception as e: log.critical(f"HTTP_SESSION_PROP: FAILED creation: {e}", exc_info=True); self._session_ready_event.clear(); raise RuntimeError("Failed HTTP session init") from e
        if not self._session_ready_event.is_set():
             log.warning("HTTP_SESSION_PROP: Waiting for ready event...");
             try: await asyncio.wait_for(self._session_ready_event.wait(), timeout=15.0); log.info("HTTP_SESSION_PROP: Ready event received.")
             except asyncio.TimeoutError: log.critical("HTTP_SESSION_PROP: Timed out waiting!"); raise RuntimeError("HTTP session init timed out.")
        if self._http_session is None or self._http_session.closed: raise RuntimeError("HTTP Session invalid after ready event!")
        return self._http_session

    async def setup_hook(self):
        hook_start_time = time.monotonic(); log.info("BOT: --- Running setup_hook ---")
        log.info("BOT: Scheduling HTTP session creation task...")
        self.loop.create_task(self._create_http_session(), name="CreateHttpSessionTask")
        log.info("BOT: HTTP session creation task scheduled.")
        db_setup_start_time = time.monotonic(); log.info("BOT: Verifying database setup (sync)...")
        try:
            # Run setup_database directly (it handles its own connection)
            database.setup_database(); db_duration = time.monotonic() - db_setup_start_time
            log.info(f"BOT: Database setup presumed complete (took {db_duration:.4f}s).")
            if db_duration > 1.0: log.warning(f"BOT: Database setup took > 1 second.")
        except Exception as e: log.critical(f"BOT: FATAL - DB setup failed: {e}", exc_info=True); raise ConnectionError("DB setup failed.") from e
        log.info("BOT: No Cogs to load."); hook_duration = time.monotonic() - hook_start_time
        log.info(f"BOT: --- Setup_hook finished successfully (total time: {hook_duration:.3f}s) ---")

    async def _create_http_session(self):
        log.info("SESSION_CREATE_TASK: Starting session creation...")
        async with self._session_lock:
            if self._http_session and not self._http_session.closed: log.warning("SESSION_CREATE_TASK: Session already exists/open."); self._session_ready_event.set(); return
            try:
                if self._http_session and self._http_session.closed: log.debug("SESSION_CREATE_TASK: Previous session closed."); self._http_session = None
                self._http_session = aiohttp.ClientSession(headers={"User-Agent": USER_AGENT})
                log.info(f"SESSION_CREATE_TASK: New session CREATED. ID: {id(self._http_session)}, Closed: {self._http_session.closed}")
                self._session_ready_event.set(); log.info("SESSION_CREATE_TASK: Session ready event SET.")
            except Exception as e: log.critical(f"SESSION_CREATE_TASK: FAILED create session: {e}", exc_info=True); self._http_session = None; self._session_ready_event.clear()

    async def close(self):
        if self._shutdown_event.is_set(): log.debug("BOT: Shutdown already in progress."); return
        log.warning("BOT: Close called..."); self._shutdown_event.set()
        if hasattr(self, '_http_session') and self._http_session and not self._http_session.closed: session_id_on_close = id(self._http_session); log.info(f"BOT: Closing aiohttp session... ID: {session_id_on_close}"); await self._http_session.close(); log.info(f"BOT: Aiohttp session closed. ID: {session_id_on_close}")
        elif hasattr(self, '_http_session') and self._http_session and self._http_session.closed: log.debug(f"BOT: Aiohttp session (ID: {id(self._http_session)}) already closed.")
        else: log.debug("BOT: Aiohttp session never created or is None.")
        log.info("BOT: Calling parent close method..."); await super().close(); log.info("BOT: Parent close finished.")
        shutdown_duration = time.monotonic() - self._start_time; log.info(f"BOT: Shutdown complete. Uptime: {shutdown_duration:.2f} seconds.")

    async def on_ready(self):
        ready_time = time.monotonic(); log.info("-" * 30); log.info(f'Logged in as: {self.user.name} ({self.user.id})'); log.info(f'Nextcord version: {nextcord.__version__}')
        latency_ms = self.latency * 1000 if self.latency is not None else "N/A"; log.info(f'Gateway Latency: {latency_ms} ms'); log.info(f'Connected to {len(self.guilds)} guilds.'); log.info(f'Bot ready after {ready_time - self._start_time:.2f} seconds.'); log.info("-" * 30)
        log.info(">>>> Repost Detector Bot is online and ready! <<<<")
        log.info(f"BOT: on_ready executing for Instance ID: {id(self)}.")
        if hasattr(self, '_http_session') and self._http_session: log.info(f"BOT: _http_session state in on_ready: ID: {id(self._http_session)}, Closed: {self._http_session.closed}, Is None: False")
        elif hasattr(self, '_http_session'): log.warning(f"BOT: _http_session attribute exists BUT IS NONE in on_ready!")
        else: log.warning("BOT: _http_session attribute MISSING in on_ready!")
        try: await self.change_presence(activity=nextcord.Activity(type=nextcord.ActivityType.watching, name="for reposts")); log.info("Presence set.")
        except Exception as e: log.error(f"Failed to set presence: {e}")

    async def on_disconnect(self): log.warning("BOT: Disconnected from Discord Gateway.")
    async def on_resumed(self): log.info("BOT: Session resumed successfully.")

    # --- Media Processing Helpers ---
    async def download_media(self, url: str) -> Optional[bytes]:
        try:
            session = await self.http_session # Get/Create session via property
            log.debug(f"DOWNLOADER: Using session ID: {id(session)}, Closed: {session.closed} for URL: {url}")
            if session.closed: log.error(f"DOWNLOADER: Session found closed before GET! URL: {url}"); return None
            log.debug(f"DOWNLOADER: Attempting download (Session OK): {url}")
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status != 200: log.warning(f"DOWNLOADER: HTTP {response.status} for {url}"); return None
                content_length = response.headers.get('Content-Length'); data = bytearray(); bytes_downloaded = 0
                if content_length:
                    try:
                        if int(content_length) > MAX_FILE_SIZE_BYTES: log.warning(f"DOWNLOADER: Exceeds size via header. Skipping: {url}"); return None
                    except ValueError: pass
                async for chunk in response.content.iter_chunked(1024 * 128): bytes_downloaded += len(chunk); data.extend(chunk);
                if bytes_downloaded > MAX_FILE_SIZE_BYTES: log.warning(f"DOWNLOADER: Exceeded max size during download. Aborting: {url}"); return None
                log.info(f"DOWNLOADER: Downloaded {bytes_downloaded / (1024*1024):.2f} MB from {url}"); return bytes(data)
        except RuntimeError as e: log.error(f"DOWNLOADER: Runtime error (session issue?) downloading {url}: {e}", exc_info=True); return None
        except asyncio.TimeoutError: log.error(f"DOWNLOADER: Timeout: {url}"); return None
        except aiohttp.ClientError as e: log.error(f"DOWNLOADER: ClientError {url}: {e}", exc_info=log.level <= logging.DEBUG); return None
        except Exception as e: log.error(f"DOWNLOADER: Unexpected error {url}: {e}", exc_info=True); return None

    async def get_image_phash(self, image_bytes: bytes) -> Optional[imagehash.ImageHash]:
        start_time = time.monotonic(); log.debug("HASHING: Starting image/GIF hashing...")
        try:
            loop = asyncio.get_running_loop(); img = None
            try: img = await loop.run_in_executor(None, Image.open, io.BytesIO(image_bytes)); log.debug(f"HASHING: Image opened...")
            except UnidentifiedImageError: log.warning("HASHING: Could not identify image format."); return None
            except Exception as e_open: log.error(f"HASHING: Error during Image.open: {e_open}", exc_info=True); return None
            if img is None: log.error("HASHING: Image object None after open."); return None
            try:
                first_frame = img; target_frame = first_frame
                if getattr(img, "is_animated", False) and getattr(img, "n_frames", 1) > 1:
                    log.debug("HASHING: Animated GIF. Getting first frame.")
                    try: iterator = ImageSequence.Iterator(img); first_frame = next(iterator); log.debug("HASHING: Got first frame."); target_frame = await loop.run_in_executor(None, first_frame.copy)
                    except StopIteration: log.warning("HASHING: StopIteration on GIF. Using base."); target_frame = await loop.run_in_executor(None, img.copy)
                    except Exception as e_frame: log.error(f"HASHING: Error getting GIF frame: {e_frame}"); target_frame = await loop.run_in_executor(None, img.copy) # Fallback
                # Convert before hashing
                if target_frame.mode not in ('L', 'RGB'): log.debug(f"HASHING: Converting mode {target_frame.mode} to RGB"); target_frame = await loop.run_in_executor(None, target_frame.convert, 'RGB')
                elif target_frame.mode == 'RGBA': log.debug(f"HASHING: Converting RGBA to RGB"); target_frame = await loop.run_in_executor(None, target_frame.convert, 'RGB') # Convert RGBA too

                log.debug(f"HASHING: Calculating pHash..."); hash_val = await loop.run_in_executor(None, imagehash.phash, target_frame, HASH_SIZE)
                processing_time = time.monotonic() - start_time; log.info(f"HASHING: Hashed IMAGE/GIF OK. Hash: {hash_val}, Time: {processing_time:.4f}s"); return hash_val
            except Exception as e_process: log.error(f"HASHING: Error hashing image post-open: {e_process}", exc_info=True)
            finally:
                 if img: # Ensure image is closed
                     try: img.close()
                     except Exception as e_close: log.warning(f"HASHING: Error closing PIL image: {e_close}")
        except Exception as e: log.error(f"HASHING: Unexpected outer error hashing image: {e}", exc_info=True)
        processing_time = time.monotonic() - start_time; log.warning(f"HASHING: Image/GIF hashing FAILED after {processing_time:.4f}s."); return None

    # --- New Multi-Frame OpenCV Video Hashing ---
    def _is_frame_black(self, frame: Optional[np.ndarray], threshold: int) -> bool:
        """Checks if a frame is likely black or very dark."""
        if frame is None: return False # Treat unreadable frames as not black
        # Check if frame has dimensions before trying to convert color
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            log.warning("FRAME_CHECK: Frame has zero dimension, cannot process.")
            return False
        try:
            # Convert to grayscale and calculate mean pixel intensity
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray)
            is_black = mean_intensity < threshold
            log.debug(f"FRAME_CHECK: Mean Intensity={mean_intensity:.2f} (Threshold={threshold}) -> Is Black? {is_black}")
            return is_black
        except cv2.error as cv_err:
            log.error(f"FRAME_CHECK: OpenCV error during black frame check: {cv_err}", exc_info=True)
            return False # Treat errors as non-black to avoid skipping valid content
        except Exception as e:
            log.error(f"FRAME_CHECK: Unexpected error during black frame check: {e}", exc_info=True)
            return False

    def _blocking_video_multi_hash_cv2(self, video_bytes: bytes, hash_size_local: int, black_thresh: int) -> Optional[List[imagehash.ImageHash]]:
        """
        Hashes first non-black, middle, and last frames of a video using OpenCV.
        Returns a list of unique hashes, or None on failure.
        """
        start_time_blocking = time.monotonic(); tmp_filepath, cap = None, None; hashes: List[imagehash.ImageHash] = []
        sync_log = logging.getLogger(f"{__name__}._blocking_video_multi_hash_cv2")
        sync_log.info("Starting blocking CV2 multi-frame video processing.")
        if not OPENCV_AVAILABLE or cv2 is None: sync_log.error("OpenCV unavailable."); return None

        try:
            # Write to temp file
            with tempfile.NamedTemporaryFile(prefix="repostbot_vid_cv2_", suffix=".tmpvid", delete=False) as tmp_file:
                tmp_file.write(video_bytes)
                tmp_filepath = tmp_file.name
            sync_log.debug(f"Video bytes written to temp file: {tmp_filepath}")

            # Open video capture
            sync_log.debug(f"Opening video capture: {tmp_filepath}")
            cap = cv2.VideoCapture(tmp_filepath)
            if not cap.isOpened(): sync_log.error(f"OpenCV failed to open: {tmp_filepath}"); return None
            sync_log.debug("Video capture opened.")

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sync_log.info(f"Video properties: Frame Count={frame_count}")

            if frame_count <= 0:
                sync_log.error(f"Video has no frames or invalid frame count: {tmp_filepath}"); return None

            # --- Determine Frame Indices ---
            first_frame_idx = 0
            middle_frame_idx = max(0, frame_count // 2) # Integer division
            last_frame_idx = max(0, frame_count - 1)

            sync_log.debug(f"Target frame indices: Initial First={first_frame_idx}, Middle={middle_frame_idx}, Last={last_frame_idx}")

            # --- Find First Non-Black Frame (starting from index 0) ---
            sync_log.debug(f"Scanning for first non-black frame (threshold: {black_thresh})...")
            actual_first_idx = -1
            scanned_frames = 0
            max_scan_frames = min(frame_count, max(10, frame_count // 10)) # Scan first 10% or 10 frames, whichever is larger (up to total frames)
            for idx in range(max_scan_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                scanned_frames += 1
                if not ret: # Removed 'or frame is None' check as ret=False implies frame is bad
                    sync_log.warning(f"Could not read frame {idx} while seeking non-black frame."); continue

                if not self._is_frame_black(frame, black_thresh):
                    sync_log.info(f"Found first non-black frame at index {idx}.")
                    actual_first_idx = idx
                    break
                else:
                     sync_log.debug(f"Frame {idx} is black/dark, skipping.")
            else: # Loop finished without break
                 sync_log.warning(f"No non-black frame found within first {scanned_frames} frames. Using frame 0 as 'first'.")
                 actual_first_idx = 0 # Default to 0 if all initial frames are black/unreadable

            # --- Hash Calculation Helper ---
            processed_hashes = set() # Use a set to store string representations to ensure uniqueness
            def get_hash_for_frame(target_idx: int, frame_name: str) -> None:
                nonlocal hashes, processed_hashes # Allow modification
                # Avoid processing the same frame index multiple times if first=middle or middle=last etc.
                if target_idx in [h_info[0] for h_info in processed_hashes]: # Check if index was already processed
                    sync_log.debug(f"Skipping hash for {frame_name} frame index {target_idx} as it was already processed.")
                    return

                pil_image = None # Define outside try
                sync_log.debug(f"Attempting to hash {frame_name} frame at index {target_idx}...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ret, frame = cap.read()
                if not ret: # Check if frame read was successful
                    sync_log.error(f"OpenCV failed to read {frame_name} frame (index {target_idx}) from {tmp_filepath}")
                    return

                try:
                    # Check frame validity again after read
                    if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                        sync_log.error(f"Invalid frame obtained for {frame_name} (index {target_idx}). Shape: {frame.shape if frame is not None else 'None'}")
                        return

                    sync_log.debug(f"Frame {frame_name} (idx {target_idx}) obtained (shape: {frame.shape}). Converting BGR to RGB.")
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    sync_log.debug("Converting frame to PIL Image...")
                    pil_image = Image.fromarray(rgb_frame)
                    sync_log.debug(f"Calculating pHash (hash_size={hash_size_local})...")
                    hash_val = imagehash.phash(pil_image, hash_size=hash_size_local)
                    hash_str = str(hash_val)

                    # Add the (index, hash_string) tuple to processed_hashes set
                    processed_hashes.add((target_idx, hash_str))
                    # Only add the ImageHash object to the final list if the string isn't already there
                    if hash_str not in [h_info[1] for h_info in processed_hashes if h_info[0] != target_idx]: # Check only hash strings of *other* indices
                       sync_log.info(f"Successfully hashed {frame_name} frame (idx {target_idx}). Hash: {hash_str}")
                       hashes.append(hash_val)
                    else:
                        sync_log.info(f"Hash {hash_str} for {frame_name} frame (idx {target_idx}) is a duplicate of another frame's hash. Storing index but not adding hash object again.")

                except cv2.error as cv_err:
                    sync_log.error(f"OpenCV error hashing {frame_name} frame (index {target_idx}): {cv_err}", exc_info=True)
                except Exception as e_hash:
                    sync_log.error(f"Error hashing {frame_name} frame (index {target_idx}): {e_hash}", exc_info=True)
                finally:
                    if pil_image: # Ensure PIL image is closed
                        try: pil_image.close()
                        except Exception as e_close: sync_log.warning(f"Error closing PIL image for frame {target_idx}: {e_close}")


            # --- Get Hashes for Selected Frames ---
            get_hash_for_frame(actual_first_idx, "First (non-black)")
            get_hash_for_frame(middle_frame_idx, "Middle")
            get_hash_for_frame(last_frame_idx, "Last")


        except Exception as e:
            sync_log.error(f"Error during video hashing process ({tmp_filepath}): {e}", exc_info=True)
            hashes = [] # Clear hashes on major error
        finally:
            # Release capture
            if cap is not None: # Check existence before isOpened
                try:
                    if cap.isOpened():
                        sync_log.debug("Releasing OpenCV video capture...")
                        cap.release()
                        sync_log.debug("OpenCV video capture released.")
                    else:
                        sync_log.debug("OpenCV capture was not opened or already released.")
                except Exception as e_rel:
                    sync_log.error(f"Error releasing OpenCV capture: {e_rel}")
            # Delete temp file
            if tmp_filepath and os.path.exists(tmp_filepath):
                sync_log.debug(f"Deleting temp file: {tmp_filepath}")
                try:
                    os.remove(tmp_filepath)
                    sync_log.debug("Temp file deleted.")
                except OSError as ed:
                    sync_log.error(f"Error deleting temp file {tmp_filepath}: {ed}")

        processing_time_blocking = time.monotonic() - start_time_blocking
        # Ensure we only return unique ImageHash objects
        unique_hashes_final = list(dict.fromkeys(hashes)) # Preserve order while making unique
        if unique_hashes_final:
            sync_log.info(f"CV2 Video multi-hashing finished. Generated {len(unique_hashes_final)} unique hash(es). Time: {processing_time_blocking:.4f}s.")
            return unique_hashes_final
        else:
            sync_log.warning(f"CV2 Video multi-hashing FAILED or produced no unique hashes after {processing_time_blocking:.4f}s.")
            return None

    async def get_video_multi_frame_phashes(self, video_bytes: bytes) -> Optional[List[imagehash.ImageHash]]:
        """Async wrapper for multi-frame video hashing using OpenCV."""
        start_time = time.monotonic(); log.info("HASHING: Starting video multi-frame hashing (async wrapper using OpenCV)...")
        if not OPENCV_AVAILABLE: log.error("HASHING: OpenCV unavailable."); return None
        try:
            loop = asyncio.get_running_loop(); log.debug("HASHING: Scheduling blocking CV2 video multi-hash...")
            # Pass HASH_SIZE and BLACK_FRAME_THRESHOLD from config
            hash_list = await loop.run_in_executor(None, self._blocking_video_multi_hash_cv2, video_bytes, HASH_SIZE, BLACK_FRAME_THRESHOLD)
            processing_time = time.monotonic() - start_time
            if hash_list:
                hashes_str = "; ".join(str(h) for h in hash_list)
                log.info(f"HASHING: OpenCV video multi-hashing OK (async). Hashes: [{hashes_str}], Count: {len(hash_list)}, Time: {processing_time:.4f}s");
                return hash_list
            else:
                log.warning(f"HASHING: OpenCV video multi-hashing FAILED (async) or no hashes generated after {processing_time:.4f}s.");
                return None
        except Exception as e: log.error(f"HASHING: Error in async CV2 video multi-hash wrapper: {e}", exc_info=True); return None

    async def handle_repost(self, repost_message: nextcord.Message, original_post_info: dict):
        task_id = f"MsgID {repost_message.id} (Repost Handler)"; log.info(f"HANDLE_REPOST [{task_id}]: Starting actions...")
        if not repost_message or not repost_message.guild or not repost_message.channel: log.warning(f"HANDLE_REPOST [{task_id}]: Context lost."); return
        try: channel_perms = repost_message.channel.permissions_for(repost_message.guild.me); can_send, can_delete = channel_perms.send_messages, channel_perms.manage_messages; log.info(f"HANDLE_REPOST [{task_id}]: Perms Check - Send:{can_send}, Delete:{can_delete}")
        except Exception as e: log.error(f"HANDLE_REPOST [{task_id}]: Perm check failed: {e}"); return

        original_author_name = f"User ID: {original_post_info['author_id']}";
        try:
            original_author = repost_message.guild.get_member(original_post_info['author_id']) or await self.fetch_user(original_post_info['author_id']);
            if original_author: original_author_name = getattr(original_author, 'display_name', original_author.name)
        except nextcord.NotFound: log.warning(f"HANDLE_REPOST [{task_id}]: Original author {original_post_info['author_id']} not found.")
        except Exception as e_fetch: log.warning(f"HANDLE_REPOST [{task_id}]: Could not fetch original author {original_post_info['author_id']}: {e_fetch}")

        original_timestamp = int(original_post_info['timestamp']); reply_sent = False
        similarity_score = original_post_info.get('similarity', 'N/A') # Get similarity score if available

        if can_send:
            log.debug(f"HANDLE_REPOST [{task_id}]: Attempting reply...");
            try:
                reply_content = (
                    f"⚠️ **Repost Alert & Removed!** {repost_message.author.mention}, this looks very similar (Similarity: {similarity_score})...\n"
                    f"Original by **{original_author_name}** on <t:{original_timestamp}:f> (<t:{original_timestamp}:R>)\n"
                    f"🔗 Original post: {original_post_info['link']}"
                )
                await repost_message.reply(reply_content, mention_author=True); log.info(f"HANDLE_REPOST [{task_id}]: Reply sent."); reply_sent = True
            except Exception as e: log.error(f"HANDLE_REPOST [{task_id}]: Error replying: {e}", exc_info=log.level<=logging.DEBUG)
        else: log.warning(f"HANDLE_REPOST [{task_id}]: Skipping reply (no permission).")

        if can_delete:
            log.debug(f"HANDLE_REPOST [{task_id}]: Attempting delete...");
            try: await repost_message.delete(); log.info(f"HANDLE_REPOST [{task_id}]: Delete successful.")
            except nextcord.NotFound: log.warning(f"HANDLE_REPOST [{task_id}]: Message {repost_message.id} not found for deletion (already deleted?).")
            except nextcord.Forbidden: log.error(f"HANDLE_REPOST [{task_id}]: Missing permission to delete message {repost_message.id}.")
            except Exception as e: log.error(f"HANDLE_REPOST [{task_id}]: Error deleting: {e}", exc_info=log.level<=logging.DEBUG)
        else:
            log.warning(f"HANDLE_REPOST [{task_id}]: Skipping delete (no permission).")
            if reply_sent and can_send:
                 try: await repost_message.channel.send(f"ℹ️ Unable to delete repost by {repost_message.author.mention}. Missing 'Manage Messages'.", delete_after=30)
                 except Exception: pass
        log.info(f"HANDLE_REPOST [{task_id}]: Finished actions.")

    async def process_media(self, message: nextcord.Message, media_url: str, source_description: str, media_type: str):
        task_id = f"MsgID {message.id} ({media_type} via {source_description.split(' ')[0]})"; log.info(f"PROCESS_MEDIA [{task_id}]: START. URL: {media_url}")
        if not message.guild: log.warning(f"PROCESS_MEDIA [{task_id}]: No guild context."); return

        # Re-fetch message to ensure it still exists and get current state
        try: message = await message.channel.fetch_message(message.id);
        except (nextcord.NotFound, nextcord.Forbidden) as e: log.warning(f"PROCESS_MEDIA [{task_id}]: Msg not found/inaccessible ({type(e).__name__}). Aborting."); return
        except Exception as e: log.error(f"PROCESS_MEDIA [{task_id}]: Error re-fetching msg: {e}. Aborting."); return
        if not message or not message.guild: log.warning(f"PROCESS_MEDIA [{task_id}]: Msg deleted/context lost after re-fetch. Aborting."); return

        guild_id = message.guild.id

        media_bytes = await self.download_media(media_url)
        if not media_bytes: log.warning(f"PROCESS_MEDIA [{task_id}]: Download failed."); return

        # current_hashes can now be a single hash or a list
        current_hashes: Optional[Union[ImageHash, List[ImageHash]]] = None
        log.debug(f"PROCESS_MEDIA [{task_id}]: Hashing as {media_type}...")

        if media_type == 'image':
            current_hashes = await self.get_image_phash(media_bytes)
        elif media_type == 'video':
            if OPENCV_AVAILABLE:
                current_hashes = await self.get_video_multi_frame_phashes(media_bytes)
            else:
                log.warning(f"PROCESS_MEDIA [{task_id}]: Skipping video, OpenCV unavailable."); return
        else:
            log.error(f"PROCESS_MEDIA [{task_id}]: Invalid media_type '{media_type}'."); return

        # Check if hashing was successful (result is not None and not an empty list)
        if not current_hashes:
            log.warning(f"PROCESS_MEDIA [{task_id}]: Hashing failed or produced no valid hashes."); return

        # Log the hash(es) generated
        if isinstance(current_hashes, list):
             hashes_str = "; ".join(str(h) for h in current_hashes)
             log.info(f"PROCESS_MEDIA [{task_id}]: Hashing successful. Hashes: [{hashes_str}] (Count: {len(current_hashes)})")
        else: # Single ImageHash
             log.info(f"PROCESS_MEDIA [{task_id}]: Hashing successful. Hash: {str(current_hashes)}")

        try: # Database Interaction & Action
            log.debug(f"PROCESS_MEDIA [{task_id}]: Checking DB..."); threshold_used = SIMILARITY_THRESHOLD
            log.debug(f"PROCESS_MEDIA [{task_id}]: Using Threshold: {threshold_used}")

            # Pass the current hash or list of hashes to the updated find_similar_hash
            existing_post = database.find_similar_hash(guild_id, current_hashes, threshold_used)

            if existing_post: log.debug(f"PROCESS_MEDIA [{task_id}]: DB Match Found! Original MsgID: {existing_post['message_id']} (Similarity: {existing_post.get('similarity', 'N/A')})")
            else: log.debug(f"PROCESS_MEDIA [{task_id}]: DB No Match Found.")

            if existing_post:
                original_msg_id, current_msg_id = existing_post["message_id"], message.id
                log.debug(f"PROCESS_MEDIA [{task_id}]: Comparing Found MsgID ({original_msg_id}) vs Current ({current_msg_id})")

                if original_msg_id == current_msg_id:
                    log.info(f"PROCESS_MEDIA [{task_id}]: Match is the current message itself (likely race condition or re-scan). Ensuring hash is present.");
                    database.add_hash(guild_id, message.channel.id, message.id, message.author.id, current_hashes, media_url)
                else:
                    log.info(f"!!! PROCESS_MEDIA [{task_id}]: REPOST DETECTED !!! Similar to {original_msg_id}");
                    log.debug(f"PROCESS_MEDIA [{task_id}]: Calling handle_repost...");
                    await self.handle_repost(message, existing_post)
            else:
                log.info(f"PROCESS_MEDIA [{task_id}]: No match found. Adding new hash(es).");
                database.add_hash(guild_id, message.channel.id, message.id, message.author.id, current_hashes, media_url)

        except Exception as e: log.error(f"PROCESS_MEDIA [{task_id}]: Error during DB/Action phase: {e}", exc_info=True)
        log.info(f"PROCESS_MEDIA [{task_id}]: END")


    async def on_message(self, message: nextcord.Message):
        if not message.author.bot: log.debug(f"### ON_MESSAGE RECEIVED EVENT (BOT CLASS): MsgID {message.id}, Author {message.author.id} ###")
        if message.author.bot: return;
        if not message.guild: return;
        if not message.attachments and not message.embeds: return;
        if message.content.startswith(BOT_COMMAND_PREFIX): return # Ignore potential commands handled elsewhere
        if message.type not in (nextcord.MessageType.default, nextcord.MessageType.reply): return;

        msg_id = message.id; log.debug(f"ON_MESSAGE [{msg_id}]: Passed initial filters.")

        # Permission Check
        if isinstance(message.channel, (nextcord.TextChannel, nextcord.Thread, nextcord.VoiceChannel)):
            log.debug(f"ON_MESSAGE [{msg_id}]: Checking permissions in Ch:{message.channel.id} '{message.channel.name}'...")
            try:
                perms = message.channel.permissions_for(message.guild.me); missing = []
                if not perms.read_message_history: missing.append("Read History")
                if not perms.send_messages: missing.append("Send")
                if not perms.manage_messages: missing.append("Manage")
                if missing: log.warning(f"ON_MESSAGE [{msg_id}]: !!! Missing permissions ({', '.join(missing)}) in Ch:{message.channel.id}. Skipping. !!!"); return
                else: log.info(f"ON_MESSAGE [{msg_id}]: Bot has required permissions in Ch:{message.channel.id}.")
            except Exception as e: log.error(f"ON_MESSAGE [{msg_id}]: Error checking permissions: {e}"); return
        else: log.debug(f"ON_MESSAGE [{msg_id}]: Skipping unsupported channel type: {message.channel.type}"); return

        # Media Identification & Task Creation
        log.debug(f"ON_MESSAGE [{msg_id}]: Identifying media items...")
        processed_urls = set()
        tasks_to_create = []

        # --- Attachment Processing ---
        if message.attachments: log.debug(f"ON_MESSAGE [{msg_id}]: Processing {len(message.attachments)} attachments...")
        for attachment in message.attachments:
            if attachment.url in processed_urls: continue
            if not (0 < attachment.size <= MAX_FILE_SIZE_BYTES):
                log.debug(f"ON_MESSAGE [{msg_id}]: Attachment '{attachment.filename}' size {attachment.size} out of range (0-{MAX_FILE_SIZE_BYTES}). Skipping.")
                continue

            file_ext = os.path.splitext(attachment.filename)[1].lower(); media_type = None
            if file_ext in SUPPORTED_IMAGE_EXTENSIONS:
                 media_type = 'image'
            elif file_ext in SUPPORTED_VIDEO_EXTENSIONS:
                if OPENCV_AVAILABLE:
                     media_type = 'video'
                else:
                     log.warning(f"ON_MESSAGE [{msg_id}]: Skipping video attachment '{attachment.filename}', OpenCV unavailable."); continue

            if media_type:
                log.info(f"ON_MESSAGE [{msg_id}]: Found {media_type} attachment: '{attachment.filename}'. Queueing task.");
                # Queue task instead of creating immediately
                tasks_to_create.append(
                    self.process_media(message, attachment.url, f"attachment '{attachment.filename}'", media_type)
                )
                processed_urls.add(attachment.url)

        # --- Embed Processing ---
        if message.embeds: log.debug(f"ON_MESSAGE [{msg_id}]: Processing {len(message.embeds)} embeds...")
        for i, embed in enumerate(message.embeds):
            media_url, embed_type_desc, potential_media_type = None, "unknown", None

            if embed.video and embed.video.url:
                 media_url, embed_type_desc, potential_media_type = embed.video.url, f"{embed.type} embed (video URL)", 'video'
                 log.debug(f"ON_MESSAGE [{msg_id}]: Embed #{i+1} has explicit video URL.")
            elif embed.image and embed.image.url:
                 media_url, embed_type_desc, potential_media_type = embed.image.url, f"image embed", 'image'
                 log.debug(f"ON_MESSAGE [{msg_id}]: Embed #{i+1} has image URL.")
            elif embed.thumbnail and embed.thumbnail.url:
                 media_url, embed_type_desc, potential_media_type = embed.thumbnail.url, f"thumbnail embed", 'image'
                 log.debug(f"ON_MESSAGE [{msg_id}]: Embed #{i+1} has thumbnail URL.")

            if media_url and media_url not in processed_urls:
                log.debug(f"ON_MESSAGE [{msg_id}]: Found potential media URL in embed #{i+1} ({embed_type_desc}): {media_url}")
                final_media_type = None
                try:
                    parsed_url_path = media_url.split('?')[0]
                    file_ext = os.path.splitext(parsed_url_path)[1].lower()

                    if file_ext in SUPPORTED_IMAGE_EXTENSIONS:
                        final_media_type = 'image'
                    elif file_ext in SUPPORTED_VIDEO_EXTENSIONS:
                        if OPENCV_AVAILABLE:
                             final_media_type = 'video'
                        else:
                             log.warning(f"ON_MESSAGE [{msg_id}]: Skipping video embed '{media_url}', OpenCV unavailable."); continue
                    elif not file_ext and potential_media_type == 'image': # Trust image hint if no extension
                        final_media_type = 'image'
                        log.debug(f"ON_MESSAGE [{msg_id}]: No extension on embed URL '{media_url}', but assuming image based on embed type.")
                    elif not file_ext and potential_media_type == 'video': # Trust video hint if no extension
                        if OPENCV_AVAILABLE:
                             final_media_type = 'video'
                             log.debug(f"ON_MESSAGE [{msg_id}]: No extension on embed URL '{media_url}', but assuming video based on embed type.")
                        else:
                             log.warning(f"ON_MESSAGE [{msg_id}]: Skipping video embed '{media_url}' (no ext, type hint), OpenCV unavailable."); continue

                except Exception as url_e:
                    log.warning(f"ON_MESSAGE [{msg_id}]: Failed to parse embed URL '{media_url}': {url_e}"); continue

                if final_media_type:
                    log.info(f"ON_MESSAGE [{msg_id}]: Found {final_media_type} embed #{i+1}. Queueing task.");
                    # Queue task
                    tasks_to_create.append(
                        self.process_media(message, media_url, embed_type_desc, final_media_type)
                    )
                    processed_urls.add(media_url)
                else:
                    log.debug(f"ON_MESSAGE [{msg_id}]: Could not determine valid media type for embed URL: {media_url}")


        # --- Create and Run Tasks ---
        if tasks_to_create:
             log.info(f"ON_MESSAGE [{msg_id}]: Creating {len(tasks_to_create)} processing tasks...")
             # Use asyncio.gather to run them concurrently and wait for completion (optional, but good practice)
             try:
                 await asyncio.gather(*tasks_to_create, return_exceptions=True)
                 log.info(f"ON_MESSAGE [{msg_id}]: Finished processing {len(tasks_to_create)} media items.")
             except Exception as gather_err:
                 log.error(f"ON_MESSAGE [{msg_id}]: Error during asyncio.gather for media processing: {gather_err}", exc_info=True)
        else:
             log.debug(f"ON_MESSAGE [{msg_id}]: No processable media found in message.")

        log.debug(f"ON_MESSAGE [{msg_id}]: Finished scanning message.")


# --- Graceful Shutdown Signal Handling ---
async def shutdown_signal_handler(bot_instance: RepostBot, signal_type: signal.Signals):
    log.warning(f"SIGNAL: Received {signal_type.name}. Initiating shutdown...");
    if not bot_instance.is_closed() and not bot_instance._shutdown_event.is_set():
        # Use create_task to avoid blocking the signal handler itself
        asyncio.create_task(bot_instance.close(), name="SignalShutdownTask")

# --- Main Execution Block (`if __name__ == "__main__":`) ---
if __name__ == "__main__":
    log.info(f"--- Starting Repost Detector Bot ({USER_AGENT}) ---") # Use variable
    if not BOT_TOKEN: log.critical("BOT: FATAL - DISCORD_BOT_TOKEN not set!"); sys.exit(1)
    try: log_config()
    except Exception as cfg_log_err: log.error(f"Failed log config: {cfg_log_err}")

    intents = nextcord.Intents.default()
    intents.message_content=True # Explicitly needed
    intents.messages=True        # Needed for on_message
    intents.guilds=True          # Needed for guild context, members etc.

    bot = RepostBot(command_prefix=BOT_COMMAND_PREFIX, intents=intents)

    # Ensure event loop exists
    try: loop = asyncio.get_event_loop()
    except RuntimeError: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)

    # Setup signal handlers
    graceful_exit_signals = (signal.SIGINT, signal.SIGTERM)
    if os.name == 'nt': graceful_exit_signals = (signal.SIGINT,) # SIGTERM not available on Windows console apps typically
    try:
        for sig in graceful_exit_signals:
             # Use lambda to pass bot instance and signal type
             loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown_signal_handler(bot, s)))
             log.info(f"Registered signal handler for {sig.name}")
    except NotImplementedError: # Happens in environments like IDLE or some non-Unix systems
        log.warning("Signal handlers not supported in this environment.")
    except Exception as e: log.error(f"Error setting up signal handlers: {e}", exc_info=True)

    # Main bot execution loop
    try:
        log.info("BOT: Starting bot execution using bot.run()...")
        bot.run(BOT_TOKEN)
    except nextcord.LoginFailure:
        log.critical("BOT: Login failed. Check your BOT_TOKEN.")
    except Exception as e:
        log.critical(f"BOT: Bot run failed: {e}", exc_info=True)
    finally:
        log.info("BOT: Main execution scope finished or interrupted.")
        # --- Final Cleanup ---
        # Check if bot needs closing (might already be closed by signal handler)
        if not bot.is_closed() and not bot._shutdown_event.is_set():
             log.warning("BOT: Bot not closed properly in finally block. Forcing final close...")
             try:
                 # Run the close coroutine synchronously if loop isn't running
                 if loop.is_running():
                     loop.run_until_complete(asyncio.wait_for(bot.close(), timeout=10.0))
                 else:
                     # If loop is closed or stopped, try running directly (less ideal)
                     async def force_close(): await bot.close()
                     asyncio.run(force_close()) # Runs a new event loop temporarily if needed
             except asyncio.TimeoutError:
                 log.error("BOT: Final forced close timed out.")
             except Exception as fe:
                 log.error(f"Error during final forced close: {fe}", exc_info=True)

        # Clean up remaining tasks if loop is still accessible and running
        if not loop.is_closed():
            tasks = [t for t in asyncio.all_tasks(loop=loop) if not t.done()]
            if tasks:
                log.info(f"BOT: Cancelling {len(tasks)} outstanding tasks...");
                for task in tasks: task.cancel()
                try:
                    # Wait for tasks to cancel
                    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                except RuntimeError as e_runtime: # Handle case where loop is closed during gather
                    log.error(f"Error gathering cancelled tasks (loop likely closed): {e_runtime}")
                except Exception as e_gather:
                    log.error(f"Error while gathering cancelled tasks: {e_gather}", exc_info=True)

            if loop.is_running():
                 log.info("BOT: Stopping asyncio event loop...")
                 loop.stop() # Stops the loop if run_forever was used (not typical with bot.run)
            log.info("BOT: Closing asyncio event loop...")
            loop.close()
            log.info("BOT: Loop closed.")
        else:
             log.info("BOT: Event loop already closed.")

        log.info(f"--- Repost Detector Bot ({USER_AGENT}) process finished ---")
