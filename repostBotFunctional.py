# bot.py
import nextcord
from nextcord.ext import commands
import os
import io
import logging
import asyncio
import aiohttp # For async web requests
from PIL import Image, UnidentifiedImageError, ImageSequence # Pillow for image processing
import imagehash # For perceptual hashing
from dotenv import load_dotenv
import time
from typing import Optional, Union, Dict, Any
import sys
import signal
import tempfile
import numpy as np # Keep as imagehash might use it indirectly
import shutil # For shutil.which

# --- MoviePy Import Removed ---
MOVIEPY_AVAILABLE = False

# --- Local Imports ---
try: import database; logging.debug("Database imported.")
except Exception as e: print(f"CRITICAL DB Import Error: {e}"); sys.exit(1)

# --- Configuration Loading ---
load_dotenv()
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
try: # Simplified config loading
    HASH_SIZE = int(os.getenv("HASH_SIZE", "8")); SIMILARITY_THRESHOLD = int(os.getenv("SIMILARITY_THRESHOLD", "5"))
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "30")); LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
    BOT_COMMAND_PREFIX = os.getenv("BOT_PREFIX", "!")
except Exception as e: print(f"ERROR parsing .env: {e}"); HASH_SIZE=8; SIMILARITY_THRESHOLD=5; MAX_FILE_SIZE_MB=30; LOG_LEVEL_STR="INFO"; BOT_COMMAND_PREFIX="!"
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB*1024*1024
SUPPORTED_IMAGE_EXTENSIONS = ('.png','.jpg','.jpeg','.gif','.bmp','.webp')
USER_AGENT = "DiscordBot RepostDetector (v1.11a-SyntaxSweep)" # Version bump

# --- Logging Setup ---
LOG_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}; LOG_LEVEL = LOG_LEVELS.get(LOG_LEVEL_STR, logging.INFO)
LOG_FORMAT = '%(asctime)s [%(levelname)-8s] [%(name)s]: %(message)s'; LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
log_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT); log_handler = logging.StreamHandler(sys.stdout); log_handler.setFormatter(log_formatter)
logging.getLogger("aiohttp").setLevel(logging.WARNING); logging.getLogger("PIL").setLevel(logging.INFO);
logging.getLogger("nextcord.gateway").setLevel(logging.INFO); logging.getLogger("nextcord.client").setLevel(logging.INFO)
root_logger = logging.getLogger(); root_logger.setLevel(LOG_LEVEL);
for handler in root_logger.handlers[:]: root_logger.removeHandler(handler);
root_logger.addHandler(log_handler); log = logging.getLogger(__name__)
def log_config(): # Definition remains same
    config_log = logging.getLogger("ConfigLoader"); config_log.info("-" * 40); config_log.info(" Bot Configuration Loaded:"); config_log.info("-" * 40); config_log.info(f"  Log Level           : {LOG_LEVEL_STR} ({LOG_LEVEL})"); config_log.info(f"  Hash Size           : {HASH_SIZE}"); config_log.info(f"  Similarity Threshold: {SIMILARITY_THRESHOLD}"); config_log.info(f"  Max File Size (MB)  : {MAX_FILE_SIZE_MB}"); config_log.info(f"  Bot Prefix          : {BOT_COMMAND_PREFIX}"); config_log.info(f"  Supported Images    : {' '.join(SUPPORTED_IMAGE_EXTENSIONS)}"); config_log.info(f"  MoviePy Available   : {MOVIEPY_AVAILABLE}"); config_log.info("-" * 40)

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

    # --- Async Property for HTTP Session ---
    @property
    async def http_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._http_session is None or self._http_session.closed:
                log.info("HTTP_SESSION_PROP: Creating new session...")
                try:
                    self._http_session = aiohttp.ClientSession(headers={"User-Agent": USER_AGENT})
                    log.info(f"HTTP_SESSION_PROP: New session CREATED. ID: {id(self._http_session)}")
                    self._session_ready_event.set()
                except Exception as e: log.critical(f"HTTP_SESSION_PROP: FAILED creation: {e}", exc_info=True); self._session_ready_event.clear(); raise RuntimeError("Failed HTTP session init") from e
        if not self._session_ready_event.is_set():
             log.warning("HTTP_SESSION_PROP: Waiting for ready event...");
             try: await asyncio.wait_for(self._session_ready_event.wait(), timeout=15.0); log.info("HTTP_SESSION_PROP: Ready event received.")
             except asyncio.TimeoutError: log.critical("HTTP_SESSION_PROP: Timed out waiting!"); raise RuntimeError("HTTP session init timed out.")
        if self._http_session is None or self._http_session.closed: raise RuntimeError("HTTP Session invalid after ready event!")
        return self._http_session

    # --- Setup Hook ---
    async def setup_hook(self):
        hook_start_time = time.monotonic(); log.info("BOT: --- Running setup_hook ---")
        log.info("BOT: Scheduling HTTP session creation task...")
        self.loop.create_task(self._create_http_session(), name="CreateHttpSessionTask")
        log.info("BOT: HTTP session creation task scheduled.")
        db_setup_start_time = time.monotonic(); log.info("BOT: Verifying database setup (sync)...")
        try:
            database.setup_database(); db_duration = time.monotonic() - db_setup_start_time
            log.info(f"BOT: Database setup presumed complete (took {db_duration:.4f}s).")
            if db_duration > 1.0: log.warning(f"BOT: Database setup took > 1 second.")
        except Exception as e: log.critical(f"BOT: FATAL - DB setup failed: {e}", exc_info=True); raise ConnectionError("DB setup failed.") from e
        log.info("BOT: No Cogs to load."); hook_duration = time.monotonic() - hook_start_time
        log.info(f"BOT: --- Setup_hook finished successfully (total time: {hook_duration:.3f}s) ---")

    # --- NEW Method to Create Session ---
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

    # --- Graceful Shutdown ---
    async def close(self):
        if self._shutdown_event.is_set(): log.debug("BOT: Shutdown already in progress."); return
        log.warning("BOT: Close called..."); self._shutdown_event.set()
        if hasattr(self, '_http_session') and self._http_session and not self._http_session.closed:
            session_id_on_close = id(self._http_session); log.info(f"BOT: Closing aiohttp session... ID: {session_id_on_close}")
            await self._http_session.close(); log.info(f"BOT: Aiohttp session closed. ID: {session_id_on_close}")
        elif hasattr(self, '_http_session') and self._http_session and self._http_session.closed: log.debug(f"BOT: Aiohttp session (ID: {id(self._http_session)}) already closed.")
        else: log.debug("BOT: Aiohttp session never created or is None.")
        log.info("BOT: Calling parent close method..."); await super().close(); log.info("BOT: Parent close finished.")
        shutdown_duration = time.monotonic() - self._start_time; log.info(f"BOT: Shutdown complete. Uptime: {shutdown_duration:.2f} seconds.")

    # --- Standard Bot Events ---
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
    # on_error handled implicitly

    # --- Media Processing Helpers (as Methods) ---
    async def download_media(self, url: str) -> Optional[bytes]:
        try:
            session = await self.http_session # Get/Create session via property
            log.debug(f"DOWNLOADER: Using session ID: {id(session)}, Closed: {session.closed} for URL: {url}")
            if session.closed:
                log.error(f"DOWNLOADER: Session found closed before GET! URL: {url}")
                return None

            log.debug(f"DOWNLOADER: Attempting download (Session OK): {url}")
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status != 200:
                    log.warning(f"DOWNLOADER: HTTP {response.status} for {url}")
                    return None

                content_length = response.headers.get('Content-Length')
                data = bytearray()
                bytes_downloaded = 0

                if content_length:
                    try:
                        size = int(content_length)
                        if size > MAX_FILE_SIZE_BYTES:
                            log.warning(f"DOWNLOADER: Exceeds size via header ({size / (1024*1024):.2f}MB). Skipping: {url}")
                            return None
                    except ValueError:
                        pass # Proceed with chunk check if header invalid

                async for chunk in response.content.iter_chunked(1024 * 128):
                    bytes_downloaded += len(chunk)
                    if bytes_downloaded > MAX_FILE_SIZE_BYTES:
                        log.warning(f"DOWNLOADER: Exceeded max size during download. Aborting: {url}")
                        return None
                    data.extend(chunk)

                log.info(f"DOWNLOADER: Downloaded {bytes_downloaded / (1024*1024):.2f} MB from {url}")
                return bytes(data) # Return successful download
        except RuntimeError as e: log.error(f"DOWNLOADER: Runtime error (session issue?) downloading {url}: {e}", exc_info=True); return None
        except asyncio.TimeoutError: log.error(f"DOWNLOADER: Timeout: {url}"); return None
        except aiohttp.ClientError as e: log.error(f"DOWNLOADER: ClientError {url}: {e}", exc_info=log.level <= logging.DEBUG); return None
        except Exception as e: log.error(f"DOWNLOADER: Unexpected error {url}: {e}", exc_info=True); return None

    async def get_image_phash(self, image_bytes: bytes) -> Optional[imagehash.ImageHash]:
        start_time = time.monotonic(); log.debug("HASHING: Starting image/GIF hashing...")
        try:
            loop = asyncio.get_running_loop()
            try: img = await loop.run_in_executor(None, Image.open, io.BytesIO(image_bytes)); log.debug(f"HASHING: Image opened...")
            except Exception as e_open: log.error(f"HASHING: Error during Image.open: {e_open}", exc_info=True); return None

            first_frame = img; target_frame = first_frame
            if getattr(img, "is_animated", False) and getattr(img, "n_frames", 1) > 1:
                log.debug("HASHING: Animated GIF. Getting first frame.")
                try:
                    iterator = ImageSequence.Iterator(img)
                    first_frame = next(iterator) # Get first frame object
                    log.debug("HASHING: Successfully obtained first frame of GIF.")
                    target_frame = await loop.run_in_executor(None, first_frame.copy)
                except StopIteration:
                     log.warning("HASHING: Could not get first frame from GIF iterator (StopIteration). Using base image.")
                     target_frame = await loop.run_in_executor(None, img.copy) # Copy base image
                except Exception as e_frame:
                     log.error(f"HASHING: Error getting first GIF frame: {e_frame}", exc_info=True)
                     target_frame = await loop.run_in_executor(None, img.copy) # Fallback

            log.debug(f"HASHING: Current target frame mode: {target_frame.mode}")
            if target_frame.mode not in ('L', 'RGB'): log.debug(f"HASHING: Converting mode {target_frame.mode} to RGB"); target_frame = await loop.run_in_executor(None, target_frame.convert, 'RGB')
            elif target_frame.mode == 'RGBA': log.debug(f"HASHING: Converting RGBA to RGB"); target_frame = await loop.run_in_executor(None, target_frame.convert, 'RGB')

            log.debug(f"HASHING: Calculating pHash...");
            hash_val = await loop.run_in_executor(None, imagehash.phash, target_frame, HASH_SIZE)

            processing_time = time.monotonic() - start_time; log.info(f"HASHING: Hashed IMAGE/GIF OK. Hash: {hash_val}, Time: {processing_time:.4f}s"); return hash_val
        except Exception as e: log.error(f"HASHING: Unexpected error during image hashing: {e}", exc_info=True)
        processing_time = time.monotonic() - start_time; log.warning(f"HASHING: Image/GIF hashing FAILED after {processing_time:.4f}s."); return None

    # Video methods removed

    async def handle_repost(self, repost_message: nextcord.Message, original_post_info: dict):
        task_id = f"MsgID {repost_message.id} (Repost Handler)"; log.info(f"HANDLE_REPOST [{task_id}]: Starting actions...")
        if not repost_message or not repost_message.guild or not repost_message.channel: log.warning(f"HANDLE_REPOST [{task_id}]: Context lost. Aborting."); return
        try: channel_perms = repost_message.channel.permissions_for(repost_message.guild.me); can_send, can_delete = channel_perms.send_messages, channel_perms.manage_messages; log.info(f"HANDLE_REPOST [{task_id}]: Perms Check - Send:{can_send}, Delete:{can_delete}")
        except Exception as e: log.error(f"HANDLE_REPOST [{task_id}]: Perm check failed. Aborting. Error: {e}"); return
        original_author_name = f"User ID: {original_post_info['author_id']}";
        try: original_author = repost_message.guild.get_member(original_post_info['author_id']) or await self.fetch_user(original_post_info['author_id']);
        except Exception: log.warning(f"HANDLE_REPOST [{task_id}]: Could not fetch original author")
        if 'original_author' in locals() and original_author: original_author_name = getattr(original_author, 'display_name', original_author.name)
        original_timestamp = int(original_post_info['timestamp']); reply_sent = False
        if can_send:
            log.debug(f"HANDLE_REPOST [{task_id}]: Attempting reply...");
            try: reply_content = (f"⚠️ **Repost Alert & Removed!** {repost_message.author.mention}, this looks very similar...\nOriginal by **{original_author_name}** on <t:{original_timestamp}:f> (<t:{original_timestamp}:R>)\n🔗 Original post: {original_post_info['link']}"); await repost_message.reply(reply_content, mention_author=True); log.info(f"HANDLE_REPOST [{task_id}]: Reply sent."); reply_sent = True
            except Exception as e: log.error(f"HANDLE_REPOST [{task_id}]: Error replying: {e}", exc_info=log.level<=logging.DEBUG)
        else: log.warning(f"HANDLE_REPOST [{task_id}]: Skipping reply (no permission).")
        if can_delete:
            log.debug(f"HANDLE_REPOST [{task_id}]: Attempting delete...");
            try: await repost_message.delete(); log.info(f"HANDLE_REPOST [{task_id}]: Delete successful.")
            except Exception as e: log.error(f"HANDLE_REPOST [{task_id}]: Error deleting: {e}", exc_info=log.level<=logging.DEBUG)
        else:
            log.warning(f"HANDLE_REPOST [{task_id}]: Skipping delete (no permission).")
            # --- Start of Corrected Block ---
            if reply_sent and can_send:
                 # Properly indented try/except
                 try:
                     await repost_message.channel.send(f"ℹ️ Unable to delete repost by {repost_message.author.mention}. Missing 'Manage Messages'.", delete_after=30)
                 except Exception:
                     pass # Ignore errors sending secondary message
            # --- End of Corrected Block ---
        log.info(f"HANDLE_REPOST [{task_id}]: Finished actions.")

    async def process_media(self, message: nextcord.Message, media_url: str, source_description: str, media_type: str):
        task_id = f"MsgID {message.id} ({media_type} via {source_description.split(' ')[0]})"; log.info(f"PROCESS_MEDIA [{task_id}]: START. URL: {media_url}")
        if not message.guild: log.warning(f"PROCESS_MEDIA [{task_id}]: No guild context. Skipping."); return
        try: message = await message.channel.fetch_message(message.id);
        except (nextcord.NotFound, nextcord.Forbidden) as e: log.warning(f"PROCESS_MEDIA [{task_id}]: Msg not found/inaccessible ({type(e).__name__}). Aborting."); return
        except Exception as e: log.error(f"PROCESS_MEDIA [{task_id}]: Error re-fetching msg: {e}", exc_info=True); return
        if not message or not message.guild: log.warning(f"PROCESS_MEDIA [{task_id}]: Msg deleted/context lost after fetch. Aborting."); return
        guild_id = message.guild.id
        media_bytes = await self.download_media(media_url) # Use lazy session getter
        if not media_bytes: log.warning(f"PROCESS_MEDIA [{task_id}]: Download failed. Aborting."); return
        current_hash: Optional[imagehash.ImageHash] = None; log.debug(f"PROCESS_MEDIA [{task_id}]: Hashing as {media_type}...")
        if media_type == 'image': current_hash = await self.get_image_phash(media_bytes) # Only image hashing
        else: log.error(f"PROCESS_MEDIA [{task_id}]: Invalid media_type '{media_type}'."); return
        if not current_hash: log.warning(f"PROCESS_MEDIA [{task_id}]: Hashing failed. Aborting."); return
        log.info(f"PROCESS_MEDIA [{task_id}]: Hashing successful. Hash: {current_hash}")
        try: # Database Interaction & Action
            log.debug(f"PROCESS_MEDIA [{task_id}]: Checking DB..."); threshold_used = SIMILARITY_THRESHOLD
            log.debug(f"PROCESS_MEDIA [{task_id}]: Using Threshold: {threshold_used}")
            existing_post = database.find_similar_hash(guild_id, current_hash, threshold_used)
            if existing_post: log.debug(f"PROCESS_MEDIA [{task_id}]: DB Check Found Match! Original MsgID: {existing_post['message_id']}")
            else: log.debug(f"PROCESS_MEDIA [{task_id}]: DB Check Found No Match.")
            if existing_post:
                original_msg_id, current_msg_id = existing_post["message_id"], message.id
                log.debug(f"PROCESS_MEDIA [{task_id}]: Comparing Found MsgID ({original_msg_id}) vs Current ({current_msg_id})")
                if original_msg_id == current_msg_id: log.info(f"PROCESS_MEDIA [{task_id}]: Match is current message. Not repost."); database.add_hash(guild_id, message.channel.id, message.id, message.author.id, current_hash, media_url)
                else: log.info(f"!!! PROCESS_MEDIA [{task_id}]: REPOST DETECTED !!! Similar to {original_msg_id}"); log.debug(f"PROCESS_MEDIA [{task_id}]: Calling handle_repost..."); await self.handle_repost(message, existing_post)
            else: log.info(f"PROCESS_MEDIA [{task_id}]: No match found. Adding new hash."); database.add_hash(guild_id, message.channel.id, message.id, message.author.id, current_hash, media_url)
        except Exception as e: log.error(f"PROCESS_MEDIA [{task_id}]: Error during DB/Action phase: {e}", exc_info=True)
        log.info(f"PROCESS_MEDIA [{task_id}]: END")


    # --- Main Event Listener (No Decorator) ---
    async def on_message(self, message: nextcord.Message):
        if not message.author.bot: log.debug(f"### ON_MESSAGE RECEIVED EVENT (BOT CLASS - Direct Override): MsgID {message.id}, Author {message.author.id} ###")
        if message.author.bot: return;
        if not message.guild: return;
        if not message.attachments and not message.embeds: return;
        if message.type not in (nextcord.MessageType.default, nextcord.MessageType.reply): return;

        msg_id = message.id; log.debug(f"ON_MESSAGE [{msg_id}]: Passed initial filters.")
        # Removed explicit session check here - rely on lazy getter in download_media

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
        if message.attachments: log.debug(f"ON_MESSAGE [{msg_id}]: Processing {len(message.attachments)} attachments...")
        for attachment in message.attachments:
            if attachment.url in processed_urls: continue
            if not (0 < attachment.size <= MAX_FILE_SIZE_BYTES): continue
            file_ext = os.path.splitext(attachment.filename)[1].lower()
            if file_ext in SUPPORTED_IMAGE_EXTENSIONS: # Only image/gif
                media_type = 'image'
                log.info(f"ON_MESSAGE [{msg_id}]: Found {media_type} attachment: '{attachment.filename}'. Scheduling task.")
                asyncio.create_task(self.process_media(message, attachment.url, f"attachment '{attachment.filename}'", media_type), name=f"process_media_{msg_id}_{attachment.id}")
                processed_urls.add(attachment.url)
        if message.embeds: log.debug(f"ON_MESSAGE [{msg_id}]: Processing {len(message.embeds)} embeds...")
        for i, embed in enumerate(message.embeds):
            media_url, embed_type_desc, potential_media_type = None, "unknown", None
            if embed.image and embed.image.url: media_url, embed_type_desc, potential_media_type = embed.image.url, "image embed", 'image'
            elif embed.thumbnail and embed.thumbnail.url: media_url, embed_type_desc, potential_media_type = embed.thumbnail.url, "thumbnail embed", 'image'
            if media_url and media_url not in processed_urls:
                log.debug(f"ON_MESSAGE [{msg_id}]: Found potential media URL in embed #{i+1} ({embed_type_desc}): {media_url}")
                try: parsed_url, file_ext = media_url.split('?')[0], os.path.splitext(media_url.split('?')[0])[1].lower()
                except Exception as url_e: log.warning(f"ON_MESSAGE [{msg_id}]: Failed to parse embed URL '{media_url}': {url_e}"); continue
                final_media_type = None
                if file_ext in SUPPORTED_IMAGE_EXTENSIONS: final_media_type = 'image'
                elif not file_ext and potential_media_type == 'image': final_media_type = 'image' # Trust hint if image
                if final_media_type == 'image': # Only schedule if image
                    log.info(f"ON_MESSAGE [{msg_id}]: Found {final_media_type} embed #{i+1} ({embed_type_desc}). Scheduling task.")
                    asyncio.create_task(self.process_media(message, media_url, embed_type_desc, final_media_type), name=f"process_media_{msg_id}_embed{i}")
                    processed_urls.add(media_url)
        log.debug(f"ON_MESSAGE [{msg_id}]: Finished scanning message for media.")


# --- Graceful Shutdown Signal Handling ---
async def shutdown_signal_handler(bot_instance: RepostBot, signal_type: signal.Signals):
    log.warning(f"SIGNAL: Received OS signal {signal_type.name}. Initiating graceful shutdown...")
    if not bot_instance.is_closed() and not bot_instance._shutdown_event.is_set(): asyncio.create_task(bot_instance.close(), name="SignalShutdownTask")
    else: log.debug(f"SIGNAL: Bot close already initiated or bot is closed. Signal {signal_type.name} ignored.")

# --- Main Execution Block (`if __name__ == "__main__":`) ---
if __name__ == "__main__":
    log.info("--- Starting Repost Detector Bot (v1.10c - Lazy Session - Syntax Fix 3) ---")
    if not BOT_TOKEN: log.critical("BOT: FATAL - DISCORD_BOT_TOKEN not set!"); sys.exit(1)
    try: log_config()
    except Exception as cfg_log_err: log.error(f"Failed to log configuration: {cfg_log_err}")
    intents = nextcord.Intents.default(); intents.message_content = True; intents.messages = True; intents.guilds = True
    bot = RepostBot(command_prefix=BOT_COMMAND_PREFIX, intents=intents)
    try: loop = asyncio.get_event_loop()
    except RuntimeError: log.info("No running event loop, creating new one."); loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    graceful_exit_signals = (signal.SIGINT, signal.SIGTERM)
    if os.name == 'nt': graceful_exit_signals = (signal.SIGINT,)
    try:
        for sig in graceful_exit_signals: loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown_signal_handler(bot, s))); log.info(f"Registered signal handler for {sig.name}")
    except NotImplementedError: log.warning("Signal handlers not fully supported on this platform.")
    except Exception as e: log.error(f"Error setting up signal handlers: {e}", exc_info=True)
    try:
        log.info("BOT: Starting bot execution using bot.run()...")
        bot.run(BOT_TOKEN)
    except nextcord.LoginFailure: log.critical("BOT: Login Failed - Invalid Discord Bot Token.")
    except nextcord.PrivilegedIntentsRequired: log.critical("BOT: Privileged Intents Required - Check Message Content Intent.")
    except ConnectionError as db_err: log.critical(f"BOT: Startup aborted due to database error: {db_err}")
    except RuntimeError as rt_err: log.critical(f"BOT: Runtime error: {rt_err}", exc_info=True)
    except KeyboardInterrupt: log.info("BOT: KeyboardInterrupt caught.")
    except Exception as e: log.critical(f"BOT: Unexpected fatal error in main run block: {e}", exc_info=True)
    finally:
        log.info("BOT: Main execution scope finished or interrupted.")
        if not bot.is_closed(): log.warning("BOT: Bot was not fully closed. Forcing final cleanup...");
        if not loop.is_closed():
            if loop.is_running():
                try: loop.run_until_complete(asyncio.wait_for(bot.close(), timeout=5.0))
                except Exception as fe: log.error(f"Error during final forced close: {fe}")
            tasks = [t for t in asyncio.all_tasks(loop=loop) if not t.done()]
            if tasks: log.info(f"BOT: Cancelling {len(tasks)} remaining tasks...");[task.cancel() for task in tasks]; loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            log.info("BOT: Closing asyncio event loop..."); loop.close(); log.info("BOT: Loop closed.")
        log.info("--- Repost Detector Bot process finished ---")
