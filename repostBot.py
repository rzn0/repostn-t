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
# Added List, Tuple, Optional to typing imports
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
    OPENCV_AVAILABLE = False; cv2 = None
except Exception as cv2_import_err:
    _initial_log.warning(f"An error occurred during OpenCV (cv2) import: {cv2_import_err}. Video processing DISABLED.")
    OPENCV_AVAILABLE = False; cv2 = None

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
    BLACK_FRAME_THRESHOLD = int(os.getenv("BLACK_FRAME_THRESHOLD", "10"))
except Exception as e:
    print(f"ERROR parsing .env: {e}"); HASH_SIZE=8; SIMILARITY_THRESHOLD=5; MAX_FILE_SIZE_MB=30; LOG_LEVEL_STR="INFO"; BOT_COMMAND_PREFIX="!"; BLACK_FRAME_THRESHOLD = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB*1024*1024
SUPPORTED_IMAGE_EXTENSIONS = ('.png','.jpg','.jpeg','.gif','.bmp','.webp')
SUPPORTED_VIDEO_EXTENSIONS = ('.mp4', '.webm', '.mov', '.avi', '.mkv', '.flv', '.wmv')
USER_AGENT = "DiscordBot RepostDetector (v1.19 - InstanceCommandSyntax)" # Version bump

# --- Logging Setup ---
LOG_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}; LOG_LEVEL = LOG_LEVELS.get(LOG_LEVEL_STR, logging.INFO)
LOG_FORMAT = '%(asctime)s [%(levelname)-8s] [%(name)s]: %(message)s'; LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
log_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT); log_handler = logging.StreamHandler(sys.stdout); log_handler.setFormatter(log_formatter)
logging.getLogger("aiohttp").setLevel(logging.WARNING); logging.getLogger("moviepy").setLevel(logging.CRITICAL + 1); logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("nextcord.gateway").setLevel(logging.INFO); logging.getLogger("nextcord.client").setLevel(logging.INFO); logging.getLogger("nextcord.ext.commands").setLevel(logging.INFO)
root_logger = logging.getLogger(); root_logger.setLevel(LOG_LEVEL);
for handler in root_logger.handlers[:]: root_logger.removeHandler(handler);
root_logger.addHandler(log_handler); log = logging.getLogger(__name__)
def log_config():
    config_log = logging.getLogger("ConfigLoader"); config_log.info("-" * 40); config_log.info(" Bot Configuration Loaded:"); config_log.info("-" * 40); config_log.info(f"  Log Level           : {LOG_LEVEL_STR} ({LOG_LEVEL})"); config_log.info(f"  Hash Size           : {HASH_SIZE}"); config_log.info(f"  Similarity Threshold: {SIMILARITY_THRESHOLD}"); config_log.info(f"  Max File Size (MB)  : {MAX_FILE_SIZE_MB}"); config_log.info(f"  Bot Prefix          : {BOT_COMMAND_PREFIX}"); config_log.info(f"  Black Frame Thresh  : {BLACK_FRAME_THRESHOLD}"); config_log.info(f"  Supported Images    : {' '.join(SUPPORTED_IMAGE_EXTENSIONS)}"); config_log.info(f"  Supported Videos    : {' '.join(SUPPORTED_VIDEO_EXTENSIONS)}"); config_log.info(f"  OpenCV Available    : {OPENCV_AVAILABLE}"); config_log.info("-" * 40)

# --- Bot Intents ---
intents = nextcord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True

# --- Bot Instance Creation ---
# We create the bot instance *before* defining commands with @bot.command
bot = commands.Bot(command_prefix=BOT_COMMAND_PREFIX, intents=intents)

# --- Global Bot State / Helpers ---
# Store http session globally or pass bot instance around carefully.
# Passing bot instance is cleaner.
# We'll keep the aiohttp session management within the class for now,
# but access it via the 'bot' instance in helper functions.

# --- Bot Class Definition (Stripped Down) ---
class RepostBotState(commands.Bot): # Renamed slightly to avoid conflict if needed, though reusing bot instance is fine.
    # Keep methods that benefit from 'self' or manage internal state like http session.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # Pass args/kwargs correctly
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        self._session_ready_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self._start_time = time.monotonic()
        log.info(f"RepostBotState __init__: Instance ID: {id(self)}. Initializing _http_session to None.")

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
        # Only manage things tightly coupled with the bot instance lifecycle here
        log.info("BOT: Scheduling HTTP session creation task...")
        self.loop.create_task(self._create_http_session(), name="CreateHttpSessionTask")
        log.info("BOT: HTTP session creation task scheduled.")
        # DB setup moved to on_ready
        hook_duration = time.monotonic() - hook_start_time
        log.info(f"BOT: --- Setup_hook finished (minimal) (total time: {hook_duration:.3f}s) ---")

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

    # Override close to handle http session
    async def close(self):
        if self._shutdown_event.is_set(): log.debug("BOT: Shutdown already in progress."); return
        log.warning("BOT: Close called..."); self._shutdown_event.set()
        if hasattr(self, '_http_session') and self._http_session and not self._http_session.closed: session_id_on_close = id(self._http_session); log.info(f"BOT: Closing aiohttp session... ID: {session_id_on_close}"); await self._http_session.close(); log.info(f"BOT: Aiohttp session closed. ID: {session_id_on_close}")
        elif hasattr(self, '_http_session') and self._http_session and self._http_session.closed: log.debug(f"BOT: Aiohttp session (ID: {id(self._http_session)}) already closed.")
        else: log.debug("BOT: Aiohttp session never created or is None.")
        log.info("BOT: Calling parent close method..."); await super().close(); log.info("BOT: Parent close finished.")
        shutdown_duration = time.monotonic() - self._start_time; log.info(f"BOT: Shutdown complete. Uptime: {shutdown_duration:.2f} seconds.")

# Re-assign the 'bot' variable to an instance of our customized class
# This keeps the @bot.command() syntax working while allowing us to manage
# the http session cleanly within the class.
bot = RepostBotState(command_prefix=BOT_COMMAND_PREFIX, intents=intents)


# --- Event Handlers (using @bot.event) ---

@bot.event
async def on_ready():
    ready_time_start = time.monotonic()
    log.info(f"BOT: --- on_ready started ---")

    # --- DB SETUP HERE ---
    db_setup_start_time = time.monotonic(); log.info("BOT: Verifying database setup (sync) in on_ready...")
    try:
        database.setup_database(); db_duration = time.monotonic() - db_setup_start_time
        log.info(f"BOT: Database setup complete in on_ready (took {db_duration:.4f}s).")
        if db_duration > 1.0: log.warning(f"BOT: Database setup took > 1 second.")
    except Exception as e:
        log.critical(f"BOT: FATAL - DB setup failed in on_ready: {e}", exc_info=True)
        # Consider closing the bot if DB setup fails critically
        # await bot.close()
        # return

    # Log basic ready info
    log.info("-" * 30); log.info(f'Logged in as: {bot.user.name} ({bot.user.id})'); log.info(f'Nextcord version: {nextcord.__version__}')
    latency_ms = bot.latency * 1000 if bot.latency is not None else "N/A"; log.info(f'Gateway Latency: {latency_ms} ms'); log.info(f'Connected to {len(bot.guilds)} guilds.');

    # Check registered commands
    registered_commands = list(bot.all_commands.keys())
    log.info(f"BOT: Registered commands checked in on_ready: {registered_commands}")
    if 'setalertchannel' not in registered_commands:
         log.error("BOT: !!! CRITICAL: 'setalertchannel' command STILL NOT registered after on_ready. !!!")

    # Check HTTP session status from the bot instance
    if hasattr(bot, '_http_session') and bot._http_session: log.info(f"BOT: _http_session state in on_ready: ID: {id(bot._http_session)}, Closed: {bot._http_session.closed}, Is None: False")
    elif hasattr(bot, '_http_session'): log.warning(f"BOT: _http_session attribute exists BUT IS NONE in on_ready!")
    else: log.warning("BOT: _http_session attribute MISSING in on_ready!") # Should not happen with RepostBotState class

    try: await bot.change_presence(activity=nextcord.Activity(type=nextcord.ActivityType.watching, name="for reposts")); log.info("Presence set.")
    except Exception as e: log.error(f"Failed to set presence: {e}")

    ready_duration = time.monotonic() - ready_time_start
    log.info(f"BOT: --- on_ready finished (took {ready_duration:.3f}s) ---")
    log.info(">>>> Repost Detector Bot is online and ready! <<<<")


@bot.event
async def on_disconnect():
    log.warning("BOT: Disconnected from Discord Gateway.")

@bot.event
async def on_resumed():
    log.info("BOT: Session resumed successfully.")


# --- Helper Functions (Now standalone, potentially taking 'bot' instance if needed) ---

async def download_media(bot_instance: commands.Bot, url: str) -> Optional[bytes]:
    # Access http_session via the bot instance passed
    try:
        # Ensure the session property is accessed correctly from the instance
        session = await bot_instance.http_session # Access property via instance
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

async def get_image_phash(image_bytes: bytes) -> Optional[imagehash.ImageHash]:
    # This function doesn't need the bot instance
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
            if target_frame.mode not in ('L', 'RGB'): log.debug(f"HASHING: Converting mode {target_frame.mode} to RGB"); target_frame = await loop.run_in_executor(None, target_frame.convert, 'RGB')
            elif target_frame.mode == 'RGBA': log.debug(f"HASHING: Converting RGBA to RGB"); target_frame = await loop.run_in_executor(None, target_frame.convert, 'RGB')

            log.debug(f"HASHING: Calculating pHash..."); hash_val = await loop.run_in_executor(None, imagehash.phash, target_frame, HASH_SIZE)
            processing_time = time.monotonic() - start_time; log.info(f"HASHING: Hashed IMAGE/GIF OK. Hash: {hash_val}, Time: {processing_time:.4f}s"); return hash_val
        except Exception as e_process: log.error(f"HASHING: Error hashing image post-open: {e_process}", exc_info=True)
        finally:
             if img:
                 try: img.close()
                 except Exception as e_close: log.warning(f"HASHING: Error closing PIL image: {e_close}")
    except Exception as e: log.error(f"HASHING: Unexpected outer error hashing image: {e}", exc_info=True)
    processing_time = time.monotonic() - start_time; log.warning(f"HASHING: Image/GIF hashing FAILED after {processing_time:.4f}s."); return None

# --- Video Hashing Helpers (Standalone if cv2 is imported globally) ---
def _is_frame_black(frame: Optional[np.ndarray], threshold: int) -> bool:
    if frame is None: return False
    if frame.shape[0] == 0 or frame.shape[1] == 0: log.warning("FRAME_CHECK: Frame has zero dimension."); return False
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        is_black = mean_intensity < threshold
        log.debug(f"FRAME_CHECK: Mean Intensity={mean_intensity:.2f} (Threshold={threshold}) -> Is Black? {is_black}")
        return is_black
    except cv2.error as cv_err: log.error(f"FRAME_CHECK: OpenCV error during black frame check: {cv_err}", exc_info=True); return False
    except Exception as e: log.error(f"FRAME_CHECK: Unexpected error during black frame check: {e}", exc_info=True); return False

def _blocking_video_multi_hash_cv2(video_bytes: bytes, hash_size_local: int, black_thresh: int) -> Optional[List[imagehash.ImageHash]]:
    start_time_blocking = time.monotonic(); tmp_filepath, cap = None, None; hashes: List[imagehash.ImageHash] = []
    sync_log = logging.getLogger(f"{__name__}._blocking_video_multi_hash_cv2")
    sync_log.info("Starting blocking CV2 multi-frame video processing.")
    if not OPENCV_AVAILABLE or cv2 is None: sync_log.error("OpenCV unavailable."); return None
    try:
        with tempfile.NamedTemporaryFile(prefix="repostbot_vid_cv2_", suffix=".tmpvid", delete=False) as tmp_file:
            tmp_file.write(video_bytes); tmp_filepath = tmp_file.name
        sync_log.debug(f"Video bytes written to temp file: {tmp_filepath}")
        sync_log.debug(f"Opening video capture: {tmp_filepath}")
        cap = cv2.VideoCapture(tmp_filepath)
        if not cap.isOpened(): sync_log.error(f"OpenCV failed to open: {tmp_filepath}"); return None
        sync_log.debug("Video capture opened.")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sync_log.info(f"Video properties: Frame Count={frame_count}")
        if frame_count <= 0: sync_log.error(f"Video has no frames: {tmp_filepath}"); return None

        first_frame_idx = 0; middle_frame_idx = max(0, frame_count // 2); last_frame_idx = max(0, frame_count - 1)
        sync_log.debug(f"Target indices: First={first_frame_idx}, Middle={middle_frame_idx}, Last={last_frame_idx}")
        sync_log.debug(f"Scanning for first non-black frame (threshold: {black_thresh})...")
        actual_first_idx = -1; scanned_frames = 0; max_scan_frames = min(frame_count, max(10, frame_count // 10))
        for idx in range(max_scan_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ret, frame = cap.read(); scanned_frames += 1
            if not ret: sync_log.warning(f"Could not read frame {idx} while seeking non-black."); continue
            if not _is_frame_black(frame, black_thresh): sync_log.info(f"Found first non-black frame at index {idx}."); actual_first_idx = idx; break
            else: sync_log.debug(f"Frame {idx} is black/dark, skipping.")
        else: sync_log.warning(f"No non-black frame found in first {scanned_frames} frames. Using frame 0."); actual_first_idx = 0

        processed_hashes = set()
        def get_hash_for_frame(target_idx: int, frame_name: str) -> None:
            nonlocal hashes, processed_hashes
            if target_idx in [h_info[0] for h_info in processed_hashes]: sync_log.debug(f"Skipping {frame_name} frame index {target_idx} (already processed)."); return
            pil_image = None
            sync_log.debug(f"Attempting hash for {frame_name} frame at index {target_idx}...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx); ret, frame = cap.read()
            if not ret: sync_log.error(f"OpenCV failed to read {frame_name} frame (index {target_idx}) from {tmp_filepath}"); return
            try:
                if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0: sync_log.error(f"Invalid frame for {frame_name} (index {target_idx}). Shape: {frame.shape if frame is not None else 'None'}"); return
                sync_log.debug(f"Frame {frame_name} (idx {target_idx}) obtained (shape: {frame.shape}). Converting BGR->RGB."); rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sync_log.debug("Converting to PIL Image..."); pil_image = Image.fromarray(rgb_frame)
                sync_log.debug(f"Calculating pHash (hash_size={hash_size_local})..."); hash_val = imagehash.phash(pil_image, hash_size=hash_size_local); hash_str = str(hash_val)
                processed_hashes.add((target_idx, hash_str))
                if hash_str not in [h_info[1] for h_info in processed_hashes if h_info[0] != target_idx]: sync_log.info(f"Successfully hashed {frame_name} frame (idx {target_idx}). Hash: {hash_str}"); hashes.append(hash_val)
                else: sync_log.info(f"Hash {hash_str} for {frame_name} frame (idx {target_idx}) is duplicate. Not adding hash object again.")
            except cv2.error as cv_err: sync_log.error(f"OpenCV error hashing {frame_name} frame (index {target_idx}): {cv_err}", exc_info=True)
            except Exception as e_hash: sync_log.error(f"Error hashing {frame_name} frame (index {target_idx}): {e_hash}", exc_info=True)
            finally:
                if pil_image:
                    try: pil_image.close()
                    except Exception as e_close: sync_log.warning(f"Error closing PIL image for frame {target_idx}: {e_close}")

        get_hash_for_frame(actual_first_idx, "First (non-black)")
        get_hash_for_frame(middle_frame_idx, "Middle")
        get_hash_for_frame(last_frame_idx, "Last")
    except Exception as e: sync_log.error(f"Error during video hashing ({tmp_filepath}): {e}", exc_info=True); hashes = []
    finally:
        if cap is not None:
            try:
                if cap.isOpened(): sync_log.debug("Releasing OpenCV capture..."); cap.release(); sync_log.debug("Capture released.")
                else: sync_log.debug("OpenCV capture already released/not opened.")
            except Exception as e_rel: sync_log.error(f"Error releasing OpenCV capture: {e_rel}")
        if tmp_filepath and os.path.exists(tmp_filepath):
            sync_log.debug(f"Deleting temp file: {tmp_filepath}")
            try: os.remove(tmp_filepath); sync_log.debug("Temp file deleted.")
            except OSError as ed: sync_log.error(f"Error deleting temp file {tmp_filepath}: {ed}")

    processing_time_blocking = time.monotonic() - start_time_blocking
    unique_hashes_final = list(dict.fromkeys(hashes))
    if unique_hashes_final: sync_log.info(f"CV2 Video multi-hashing finished. Generated {len(unique_hashes_final)} unique hash(es). Time: {processing_time_blocking:.4f}s."); return unique_hashes_final
    else: sync_log.warning(f"CV2 Video multi-hashing FAILED or no unique hashes after {processing_time_blocking:.4f}s."); return None

async def get_video_multi_frame_phashes(video_bytes: bytes) -> Optional[List[imagehash.ImageHash]]:
    # This doesn't need the bot instance
    start_time = time.monotonic(); log.info("HASHING: Starting video multi-frame hashing (async wrapper using OpenCV)...")
    if not OPENCV_AVAILABLE: log.error("HASHING: OpenCV unavailable."); return None
    try:
        loop = asyncio.get_running_loop(); log.debug("HASHING: Scheduling blocking CV2 video multi-hash...")
        # Pass HASH_SIZE and BLACK_FRAME_THRESHOLD from global scope
        hash_list = await loop.run_in_executor(None, _blocking_video_multi_hash_cv2, video_bytes, HASH_SIZE, BLACK_FRAME_THRESHOLD)
        processing_time = time.monotonic() - start_time
        if hash_list:
            hashes_str = "; ".join(str(h) for h in hash_list)
            log.info(f"HASHING: OpenCV video multi-hashing OK (async). Hashes: [{hashes_str}], Count: {len(hash_list)}, Time: {processing_time:.4f}s");
            return hash_list
        else:
            log.warning(f"HASHING: OpenCV video multi-hashing FAILED (async) or no hashes generated after {processing_time:.4f}s.");
            return None
    except Exception as e: log.error(f"HASHING: Error in async CV2 video multi-hash wrapper: {e}", exc_info=True); return None


# --- Repost Handling (Standalone, requires bot instance) ---
async def handle_repost(bot_instance: commands.Bot, repost_message: nextcord.Message, original_post_info: dict):
    """Handles sending alert and deleting repost, but only if setup is complete."""
    task_id = f"MsgID {repost_message.id} (Repost Handler)"; log.info(f"HANDLE_REPOST [{task_id}]: Starting actions...")
    if not repost_message or not repost_message.guild or not repost_message.channel:
        log.warning(f"HANDLE_REPOST [{task_id}]: Context lost (message/guild/channel missing)."); return

    guild_id = repost_message.guild.id
    repost_channel = repost_message.channel
    alert_channel_id = None # Initialize

    # --- Check if setup is complete for this guild ---
    try:
        alert_channel_id = database.get_alert_channel(guild_id)
        if alert_channel_id is None:
            log.warning(f"HANDLE_REPOST [{task_id}]: Guild {guild_id} has not completed setup. Repost detected but action halted.")
            try:
                perms_repost_channel = repost_channel.permissions_for(repost_message.guild.me)
                if perms_repost_channel.send_messages:
                    # Use bot_instance.command_prefix here
                    setup_msg = (
                        f"⚠️ **Repost Bot Setup Required!**\n"
                        f"A repost by {repost_message.author.mention} was detected, but I need setup!\n"
                        f"An administrator with 'Manage Server' permission must run the command `{bot_instance.command_prefix}setalertchannel` **in the specific text channel** where you want repost alerts to be sent."
                    )
                    await repost_channel.send(setup_msg, delete_after=120)
                else:
                    log.error(f"HANDLE_REPOST [{task_id}]: Cannot send setup reminder in Ch:{repost_channel.id}, missing Send Messages permission.")
            except Exception as e_setup_msg:
                log.error(f"HANDLE_REPOST [{task_id}]: Failed to send setup reminder message: {e_setup_msg}")
            return # Stop processing this repost
    except Exception as e_db_get:
        log.error(f"HANDLE_REPOST [{task_id}]: Error retrieving alert channel config from DB: {e_db_get}. Halting repost processing.", exc_info=True)
        return # Stop if DB error occurs during setup check

    # --- If Setup Complete, Proceed ---
    log.info(f"HANDLE_REPOST [{task_id}]: Guild {guild_id} setup complete (Alert Channel ID: {alert_channel_id}). Proceeding.")
    alert_target_channel: Optional[nextcord.abc.GuildChannel] = None
    alert_channel_validated = False
    try:
        # Use bot_instance to fetch channel
        fetched_channel = await bot_instance.fetch_channel(alert_channel_id)
        if isinstance(fetched_channel, (nextcord.TextChannel, nextcord.Thread, nextcord.VoiceChannel)):
            if fetched_channel.guild.id == guild_id:
               perms_in_alert_channel = fetched_channel.permissions_for(repost_message.guild.me)
               if perms_in_alert_channel.send_messages:
                   alert_target_channel = fetched_channel
                   alert_channel_validated = True
                   log.info(f"HANDLE_REPOST [{task_id}]: Successfully validated configured alert channel #{fetched_channel.name} ({fetched_channel.id}).")
               else: log.warning(f"HANDLE_REPOST [{task_id}]: Configured alert channel {alert_channel_id} found, but bot lacks 'Send Messages' there. Alerting may fail.")
            else: log.warning(f"HANDLE_REPOST [{task_id}]: Configured alert channel {alert_channel_id} belongs to a different guild (!?). Alerting may fail.")
        else: log.warning(f"HANDLE_REPOST [{task_id}]: Configured alert channel ID {alert_channel_id} is not a valid Text/Thread/Voice channel type ({type(fetched_channel)}). Alerting may fail.")
    except nextcord.NotFound: log.warning(f"HANDLE_REPOST [{task_id}]: Configured alert channel ID {alert_channel_id} not found (deleted?). Alerting may fail.")
    except nextcord.Forbidden: log.error(f"HANDLE_REPOST [{task_id}]: Bot lacks permissions to fetch/view configured alert channel {alert_channel_id}. Alerting may fail.")
    except Exception as e_fetch_alert: log.error(f"HANDLE_REPOST [{task_id}]: Unexpected error fetching alert channel {alert_channel_id}: {e_fetch_alert}. Alerting may fail.", exc_info=True)

    if not alert_target_channel:
        log.warning(f"HANDLE_REPOST [{task_id}]: Failed to validate configured alert channel {alert_channel_id}. Falling back to sending alert in original channel {repost_channel.id} if possible.")
        alert_target_channel = repost_channel
        try: alert_channel_validated = repost_channel.permissions_for(repost_message.guild.me).send_messages
        except Exception: alert_channel_validated = False

    can_send_alert = alert_channel_validated
    can_delete_repost = False
    try:
        perms_repost_channel = repost_channel.permissions_for(repost_message.guild.me)
        can_delete_repost = perms_repost_channel.manage_messages
        log.info(f"HANDLE_REPOST [{task_id}]: Perm Check (Alert Target: #{alert_target_channel.name if alert_target_channel else 'N/A'}): Send={can_send_alert}")
        log.info(f"HANDLE_REPOST [{task_id}]: Perm Check (Repost Channel: #{repost_channel.name}): Manage={can_delete_repost}")
    except Exception as e: log.error(f"HANDLE_REPOST [{task_id}]: Permission check failed: {e}"); return

    original_author_name = f"User ID: {original_post_info['author_id']}";
    try:
        # Use bot_instance to fetch user if needed
        original_author = repost_message.guild.get_member(original_post_info['author_id']) or await bot_instance.fetch_user(original_post_info['author_id']);
        if original_author: original_author_name = getattr(original_author, 'display_name', original_author.name)
    except nextcord.NotFound: log.warning(f"HANDLE_REPOST [{task_id}]: Original author {original_post_info['author_id']} not found.")
    except Exception as e_fetch: log.warning(f"HANDLE_REPOST [{task_id}]: Could not fetch original author {original_post_info['author_id']}: {e_fetch}")

    original_timestamp = int(original_post_info['timestamp']);
    similarity_score = original_post_info.get('similarity', 'N/A')
    repost_alert_prefix = "⚠️ **Repost Alert & Removed!**" if can_delete_repost else "⚠️ **Repost Alert!** (Deletion failed/skipped)"
    alert_content = (
        f"{repost_alert_prefix} {repost_message.author.mention}, this looks very similar (Similarity: {similarity_score})...\n"
        f"Original by **{original_author_name}** on <t:{original_timestamp}:f> (<t:{original_timestamp}:R>)\n"
        f"🔗 Original post: {original_post_info['link']}"
    )
    if alert_target_channel and alert_target_channel.id != repost_channel.id:
        alert_content += f"\n*(Detected in {repost_channel.mention})*"

    alert_sent = False
    if can_send_alert and alert_target_channel:
        log.debug(f"HANDLE_REPOST [{task_id}]: Attempting to send alert to #{alert_target_channel.name} ({alert_target_channel.id})...");
        try:
            await alert_target_channel.send(alert_content, allowed_mentions=nextcord.AllowedMentions(users=True));
            log.info(f"HANDLE_REPOST [{task_id}]: Alert sent successfully to #{alert_target_channel.name}."); alert_sent = True
        except Exception as e:
             log.error(f"HANDLE_REPOST [{task_id}]: Error sending alert to #{alert_target_channel.name}: {e}", exc_info=log.level<=logging.DEBUG)
    else:
        log.warning(f"HANDLE_REPOST [{task_id}]: Skipping alert (no permission/validation failed for target channel).")

    delete_attempted = False
    delete_successful = False
    if can_delete_repost:
        delete_attempted = True
        log.debug(f"HANDLE_REPOST [{task_id}]: Attempting delete in #{repost_channel.name}...");
        try:
            await repost_message.delete(); delete_successful = True;
            log.info(f"HANDLE_REPOST [{task_id}]: Repost message deleted successfully from #{repost_channel.name}.")
        except nextcord.NotFound: log.warning(f"HANDLE_REPOST [{task_id}]: Message {repost_message.id} not found for deletion (already deleted?).")
        except nextcord.Forbidden: log.error(f"HANDLE_REPOST [{task_id}]: Missing 'Manage Messages' permission to delete message {repost_message.id} in #{repost_channel.name}.")
        except Exception as e: log.error(f"HANDLE_REPOST [{task_id}]: Error deleting message {repost_message.id}: {e}", exc_info=log.level<=logging.DEBUG)
    else:
        log.warning(f"HANDLE_REPOST [{task_id}]: Skipping delete (no permission in #{repost_channel.name}).")
        if alert_sent and can_send_alert and alert_target_channel:
             try: await alert_target_channel.send(f"ℹ️ Note: Unable to delete the repost by {repost_message.author.mention} in {repost_channel.mention}. Missing 'Manage Messages' permission there.", delete_after=60, allowed_mentions=nextcord.AllowedMentions.none())
             except Exception: pass

    log.info(f"HANDLE_REPOST [{task_id}]: Finished actions. Alert Sent: {alert_sent}, Delete Attempted: {delete_attempted}, Delete Succeeded: {delete_successful}")

# --- Media Processing (Standalone, requires bot instance) ---
async def process_media(bot_instance: commands.Bot, message: nextcord.Message, media_url: str, source_description: str, media_type: str):
    task_id = f"MsgID {message.id} ({media_type} via {source_description.split(' ')[0]})"; log.info(f"PROCESS_MEDIA [{task_id}]: START. URL: {media_url}")
    if not message.guild: log.warning(f"PROCESS_MEDIA [{task_id}]: No guild context."); return

    try: message = await message.channel.fetch_message(message.id);
    except (nextcord.NotFound, nextcord.Forbidden) as e: log.warning(f"PROCESS_MEDIA [{task_id}]: Msg not found/inaccessible ({type(e).__name__}). Aborting."); return
    except Exception as e: log.error(f"PROCESS_MEDIA [{task_id}]: Error re-fetching msg: {e}. Aborting."); return
    if not message or not message.guild: log.warning(f"PROCESS_MEDIA [{task_id}]: Msg deleted/context lost after re-fetch. Aborting."); return

    guild_id = message.guild.id

    # Pass bot_instance to download_media
    media_bytes = await download_media(bot_instance, media_url)
    if not media_bytes: log.warning(f"PROCESS_MEDIA [{task_id}]: Download failed."); return

    current_hashes: Optional[Union[ImageHash, List[ImageHash]]] = None
    log.debug(f"PROCESS_MEDIA [{task_id}]: Hashing as {media_type}...")

    # These helpers don't need bot instance
    if media_type == 'image': current_hashes = await get_image_phash(media_bytes)
    elif media_type == 'video':
        if OPENCV_AVAILABLE: current_hashes = await get_video_multi_frame_phashes(media_bytes)
        else: log.warning(f"PROCESS_MEDIA [{task_id}]: Skipping video, OpenCV unavailable."); return
    else: log.error(f"PROCESS_MEDIA [{task_id}]: Invalid media_type '{media_type}'."); return

    if not current_hashes: log.warning(f"PROCESS_MEDIA [{task_id}]: Hashing failed or produced no valid hashes."); return

    if isinstance(current_hashes, list): hashes_str = "; ".join(str(h) for h in current_hashes); log.info(f"PROCESS_MEDIA [{task_id}]: Hashing successful. Hashes: [{hashes_str}] (Count: {len(current_hashes)})")
    else: log.info(f"PROCESS_MEDIA [{task_id}]: Hashing successful. Hash: {str(current_hashes)}")

    try: # Database Interaction & Action
        log.debug(f"PROCESS_MEDIA [{task_id}]: Checking DB..."); threshold_used = SIMILARITY_THRESHOLD
        log.debug(f"PROCESS_MEDIA [{task_id}]: Using Threshold: {threshold_used}")

        existing_post = database.find_similar_hash(guild_id, current_hashes, threshold_used)

        if existing_post: log.debug(f"PROCESS_MEDIA [{task_id}]: DB Match Found! Original MsgID: {existing_post['message_id']} (Similarity: {existing_post.get('similarity', 'N/A')})")
        else: log.debug(f"PROCESS_MEDIA [{task_id}]: DB No Match Found.")

        if existing_post:
            original_msg_id, current_msg_id = existing_post["message_id"], message.id
            log.debug(f"PROCESS_MEDIA [{task_id}]: Comparing Found MsgID ({original_msg_id}) vs Current ({current_msg_id})")

            if original_msg_id == current_msg_id:
                log.info(f"PROCESS_MEDIA [{task_id}]: Match is current message. Ensuring hash present.");
                database.add_hash(guild_id, message.channel.id, message.id, message.author.id, current_hashes, media_url)
            else:
                log.info(f"!!! PROCESS_MEDIA [{task_id}]: REPOST DETECTED !!! Similar to {original_msg_id}");
                log.debug(f"PROCESS_MEDIA [{task_id}]: Calling handle_repost (will check setup)...");
                # Pass bot_instance to handle_repost
                await handle_repost(bot_instance, message, existing_post)
        else:
            log.info(f"PROCESS_MEDIA [{task_id}]: No match found. Adding new hash(es).");
            database.add_hash(guild_id, message.channel.id, message.id, message.author.id, current_hashes, media_url)

    except Exception as e: log.error(f"PROCESS_MEDIA [{task_id}]: Error during DB/Action phase: {e}", exc_info=True)
    log.info(f"PROCESS_MEDIA [{task_id}]: END")


# --- on_message (using @bot.event) ---
@bot.event
async def on_message(message: nextcord.Message):
    # Use the global 'bot' instance here
    if message.author.bot: return;
    if not message.guild: return;

    # --- Check for command prefix FIRST ---
    if message.content.startswith(bot.command_prefix):
        log.debug(f"ON_MESSAGE [{message.id}]: Message starts with prefix '{bot.command_prefix}'. Passing to command handler.")
        await bot.process_commands(message) # Use bot instance
        return

    # --- If not a command, proceed with media detection ---
    if not message.attachments and not message.embeds: return;
    if message.type not in (nextcord.MessageType.default, nextcord.MessageType.reply): return;

    msg_id = message.id; log.debug(f"ON_MESSAGE [{msg_id}]: Passed initial filters (not a command, has media).")

    # Basic permission check in the origin channel
    if isinstance(message.channel, (nextcord.TextChannel, nextcord.Thread, nextcord.VoiceChannel)):
        log.debug(f"ON_MESSAGE [{msg_id}]: Checking permissions in Ch:{message.channel.id} '{message.channel.name}'...")
        try:
            perms = message.channel.permissions_for(message.guild.me); missing = []
            if not perms.read_message_history: missing.append("Read History")
            if not perms.send_messages: missing.append("Send (Origin)")
            if not perms.manage_messages: missing.append("Manage (Origin)")

            if missing: log.warning(f"ON_MESSAGE [{msg_id}]: !!! Missing baseline permissions ({', '.join(missing)}) in Ch:{message.channel.id}. Skipping media check. !!!"); return
            else: log.info(f"ON_MESSAGE [{msg_id}]: Bot has baseline permissions in Ch:{message.channel.id}.")
        except Exception as e: log.error(f"ON_MESSAGE [{msg_id}]: Error checking permissions: {e}"); return
    else: log.debug(f"ON_MESSAGE [{msg_id}]: Skipping unsupported channel type: {message.channel.type}"); return

    # --- Media Identification & Task Creation ---
    log.debug(f"ON_MESSAGE [{msg_id}]: Identifying media items...")
    processed_urls = set()
    tasks_to_create = []

    if message.attachments: log.debug(f"ON_MESSAGE [{msg_id}]: Processing {len(message.attachments)} attachments...")
    for attachment in message.attachments:
        if attachment.url in processed_urls: continue
        if not (0 < attachment.size <= MAX_FILE_SIZE_BYTES): log.debug(f"ON_MESSAGE [{msg_id}]: Attachment '{attachment.filename}' size {attachment.size} out of range. Skipping."); continue
        file_ext = os.path.splitext(attachment.filename)[1].lower(); media_type = None
        if file_ext in SUPPORTED_IMAGE_EXTENSIONS: media_type = 'image'
        elif file_ext in SUPPORTED_VIDEO_EXTENSIONS:
            if OPENCV_AVAILABLE: media_type = 'video'
            else: log.warning(f"ON_MESSAGE [{msg_id}]: Skipping video attachment '{attachment.filename}', OpenCV unavailable."); continue
        if media_type:
            log.info(f"ON_MESSAGE [{msg_id}]: Found {media_type} attachment: '{attachment.filename}'. Queueing task.");
            # Pass the global 'bot' instance to process_media
            tasks_to_create.append(process_media(bot, message, attachment.url, f"attachment '{attachment.filename}'", media_type))
            processed_urls.add(attachment.url)

    if message.embeds: log.debug(f"ON_MESSAGE [{msg_id}]: Processing {len(message.embeds)} embeds...")
    for i, embed in enumerate(message.embeds):
        media_url, embed_type_desc, potential_media_type = None, "unknown", None
        if embed.video and embed.video.url: media_url, embed_type_desc, potential_media_type = embed.video.url, f"{embed.type} embed (video URL)", 'video'; log.debug(f"ON_MESSAGE [{msg_id}]: Embed #{i+1} has video URL.")
        elif embed.image and embed.image.url: media_url, embed_type_desc, potential_media_type = embed.image.url, f"image embed", 'image'; log.debug(f"ON_MESSAGE [{msg_id}]: Embed #{i+1} has image URL.")
        elif embed.thumbnail and embed.thumbnail.url: media_url, embed_type_desc, potential_media_type = embed.thumbnail.url, f"thumbnail embed", 'image'; log.debug(f"ON_MESSAGE [{msg_id}]: Embed #{i+1} has thumbnail URL.")

        if media_url and media_url not in processed_urls:
            log.debug(f"ON_MESSAGE [{msg_id}]: Found potential media URL in embed #{i+1} ({embed_type_desc}): {media_url}")
            final_media_type = None
            try:
                parsed_url_path = media_url.split('?')[0]; file_ext = os.path.splitext(parsed_url_path)[1].lower()
                if file_ext in SUPPORTED_IMAGE_EXTENSIONS: final_media_type = 'image'
                elif file_ext in SUPPORTED_VIDEO_EXTENSIONS:
                    if OPENCV_AVAILABLE: final_media_type = 'video'
                    else: log.warning(f"ON_MESSAGE [{msg_id}]: Skipping video embed '{media_url}', OpenCV unavailable."); continue
                elif not file_ext and potential_media_type == 'image': final_media_type = 'image'; log.debug(f"ON_MESSAGE [{msg_id}]: Assuming image embed (no ext): {media_url}")
                elif not file_ext and potential_media_type == 'video':
                    if OPENCV_AVAILABLE: final_media_type = 'video'; log.debug(f"ON_MESSAGE [{msg_id}]: Assuming video embed (no ext): {media_url}")
                    else: log.warning(f"ON_MESSAGE [{msg_id}]: Skipping video embed (no ext, type hint), OpenCV unavailable."); continue
            except Exception as url_e: log.warning(f"ON_MESSAGE [{msg_id}]: Failed to parse embed URL '{media_url}': {url_e}"); continue

            if final_media_type:
                log.info(f"ON_MESSAGE [{msg_id}]: Found {final_media_type} embed #{i+1}. Queueing task.");
                # Pass the global 'bot' instance to process_media
                tasks_to_create.append(process_media(bot, message, media_url, embed_type_desc, final_media_type))
                processed_urls.add(media_url)
            else: log.debug(f"ON_MESSAGE [{msg_id}]: Could not determine valid media type for embed URL: {media_url}")

    # --- Create and Run Tasks ---
    if tasks_to_create:
         log.info(f"ON_MESSAGE [{msg_id}]: Creating {len(tasks_to_create)} processing tasks...")
         try:
             results = await asyncio.gather(*tasks_to_create, return_exceptions=True)
             for i, result in enumerate(results):
                 if isinstance(result, Exception):
                     log.error(f"ON_MESSAGE [{msg_id}]: Task {i+1} failed: {result}", exc_info=False)
             log.info(f"ON_MESSAGE [{msg_id}]: Finished processing {len(tasks_to_create)} media items.")
         except Exception as gather_err:
             log.error(f"ON_MESSAGE [{msg_id}]: Unexpected error during asyncio.gather for media processing: {gather_err}", exc_info=True)
    else:
         log.debug(f"ON_MESSAGE [{msg_id}]: No processable media found in message.")
    log.debug(f"ON_MESSAGE [{msg_id}]: Finished scanning message.")


# --- Setup Command (using @bot.command) ---
@bot.command(
    name="setalertchannel",
    help="REQUIRED: Run this command in the channel where you want repost alerts sent. Requires 'Manage Server' permission.",
    aliases=['setalerts']
)
@commands.guild_only()
@commands.has_permissions(manage_guild=True)
async def set_alert_channel_command(ctx: commands.Context):
    """Sets the alert channel to the channel where the command is invoked."""
    guild_id = ctx.guild.id
    channel = ctx.channel

    log.info(f"CMD_SETALERT [{ctx.message.id}]: Received command from {ctx.author} ({ctx.author.id}) in Guild {guild_id}. Targeting current channel: #{channel.name} ({channel.id})")

    if not isinstance(channel, nextcord.TextChannel):
         log.warning(f"CMD_SETALERT [{ctx.message.id}]: Command used in non-text channel type: {type(channel)}. Aborting.")
         await ctx.reply(f"⚠️ This command must be run in a standard text channel.", delete_after=30)
         return

    try:
        perms = channel.permissions_for(ctx.guild.me)
        missing_perms = []
        if not perms.send_messages: missing_perms.append("Send Messages")
        if not perms.read_message_history: missing_perms.append("Read Message History")

        if missing_perms:
            log.warning(f"CMD_SETALERT [{ctx.message.id}]: Bot lacks permissions ({', '.join(missing_perms)}) in target channel {channel.id}.")
            await ctx.reply(f"⚠️ I need the following permissions in {channel.mention} to send alerts there:\n- `{'`, `'.join(missing_perms)}`\nPlease grant them and run the command again.")
            return
    except Exception as e:
        log.error(f"CMD_SETALERT [{ctx.message.id}]: Error checking permissions for channel {channel.id}: {e}", exc_info=True)
        await ctx.reply("❌ An error occurred while checking permissions for this channel.")
        return

    # Use database module directly
    success = database.set_alert_channel(guild_id, channel.id)
    if success:
        log.info(f"CMD_SETALERT [{ctx.message.id}]: Successfully set alert channel for Guild {guild_id} to {channel.id}.")
        await ctx.reply(f"✅ Repost alerts for this server will now be sent to {channel.mention}. The bot is now active.")
    else:
        log.error(f"CMD_SETALERT [{ctx.message.id}]: Database operation failed for setting alert channel {channel.id}.")
        await ctx.reply("❌ An error occurred while saving this channel setting.")

@set_alert_channel_command.error
async def set_alert_channel_error(ctx: commands.Context, error: commands.CommandError):
    """Error handler for the setalertchannel command."""
    log_prefix = f"CMD_SETALERT_ERR [{ctx.message.id}]:"
    if isinstance(error, commands.MissingPermissions):
        log.warning(f"{log_prefix} User {ctx.author} lacks 'Manage Server' permission.")
        await ctx.reply("🚫 You need the 'Manage Server' permission to use this command.")
    elif isinstance(error, commands.NoPrivateMessage):
         log.warning(f"{log_prefix} Command invoked in DMs.")
         await ctx.reply("🚫 This command can only be used within a server text channel.")
    elif isinstance(error, commands.CommandInvokeError):
        log.error(f"{log_prefix} An error occurred during command execution: {error.original}", exc_info=True)
        await ctx.reply("❌ An internal error occurred while trying to run this command.")
    else:
        log.error(f"{log_prefix} An unexpected error occurred: {error}", exc_info=True)
        await ctx.reply("❌ An unexpected error occurred.")


# --- Graceful Shutdown Signal Handling ---
async def shutdown_signal_handler(bot_instance: commands.Bot, signal_type: signal.Signals): # Takes bot instance
    # Need to access the _shutdown_event on the instance
    if not isinstance(bot_instance, RepostBotState) or bot_instance._shutdown_event.is_set():
        log.debug("SIGNAL: Shutdown already in progress or invalid bot type.")
        return
    log.warning(f"SIGNAL: Received {signal_type.name}. Initiating shutdown...");
    # bot_instance.close() will handle setting the event and cleanup
    if not bot_instance.is_closed():
        asyncio.create_task(bot_instance.close(), name="SignalShutdownTask")


# --- Main Execution Block (`if __name__ == "__main__":`) ---
if __name__ == "__main__":
    log.info(f"--- Starting Repost Detector Bot ({USER_AGENT}) ---")
    if not BOT_TOKEN: log.critical("BOT: FATAL - DISCORD_BOT_TOKEN not set!"); sys.exit(1)
    try: log_config()
    except Exception as cfg_log_err: log.error(f"Failed log config: {cfg_log_err}")

    # 'bot' instance is already created above
    try: loop = asyncio.get_event_loop()
    except RuntimeError: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)

    graceful_exit_signals = (signal.SIGINT, signal.SIGTERM)
    if os.name == 'nt': graceful_exit_signals = (signal.SIGINT,)
    try:
        for sig in graceful_exit_signals:
             # Pass the global 'bot' instance to the handler
             loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown_signal_handler(bot, s)))
             log.info(f"Registered signal handler for {sig.name}")
    except NotImplementedError: log.warning("Signal handlers not supported in this environment.")
    except Exception as e: log.error(f"Error setting up signal handlers: {e}", exc_info=True)

    try:
        log.info("BOT: Starting bot execution using bot.run()...")
        # setup_hook is called implicitly by bot.run() on the 'bot' instance
        # on_ready event handler defined above will be triggered
        bot.run(BOT_TOKEN)
    except nextcord.LoginFailure: log.critical("BOT: Login failed. Check your BOT_TOKEN.")
    except Exception as e: log.critical(f"BOT: Bot run failed: {e}", exc_info=True)
    finally:
        log.info("BOT: Main execution scope finished or interrupted.")
        # Final cleanup using the global 'bot' instance
        # Need to check instance state correctly
        if isinstance(bot, RepostBotState) and not bot.is_closed() and not bot._shutdown_event.is_set():
             log.warning("BOT: Bot not closed properly in finally block. Forcing final close...")
             try:
                 if loop.is_running(): loop.run_until_complete(asyncio.wait_for(bot.close(), timeout=10.0))
                 else:
                     async def force_close(): await bot.close()
                     asyncio.run(force_close())
             except asyncio.TimeoutError: log.error("BOT: Final forced close timed out.")
             except RuntimeError as e: log.error(f"BOT: Error during final close (loop state): {e}")
             except Exception as fe: log.error(f"Error during final forced close: {fe}", exc_info=True)

        if not loop.is_closed():
            tasks = [t for t in asyncio.all_tasks(loop=loop) if not t.done()]
            if tasks:
                log.info(f"BOT: Cancelling {len(tasks)} outstanding tasks...");
                for task in tasks: task.cancel()
                try: loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                except RuntimeError as e_runtime: log.error(f"Error gathering cancelled tasks (loop closed?): {e_runtime}")
                except Exception as e_gather: log.error(f"Error while gathering cancelled tasks: {e_gather}", exc_info=True)

            if loop.is_running(): log.info("BOT: Stopping asyncio event loop..."); loop.stop()
            log.info("BOT: Closing asyncio event loop..."); loop.close(); log.info("BOT: Loop closed.")
        else: log.info("BOT: Event loop already closed.")

        log.info(f"--- Repost Detector Bot ({USER_AGENT}) process finished ---")
