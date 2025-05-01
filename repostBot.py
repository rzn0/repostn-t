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
from typing import Optional, Union, Dict, Any, List, Tuple, Set
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
    _initial_log.info("OpenCV imported successfully. Video processing ENABLED.")
except ImportError:
    _initial_log.warning("OpenCV library not found. Video processing DISABLED.")
    OPENCV_AVAILABLE = False; cv2 = None
except Exception as cv2_import_err:
    _initial_log.warning(f"OpenCV import error: {cv2_import_err}. Video processing DISABLED.")
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
USER_AGENT = "DiscordBot RepostDetector (v1.21 - DBWhitelist)"

# --- Logging Setup ---
LOG_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}; LOG_LEVEL = LOG_LEVELS.get(LOG_LEVEL_STR, logging.INFO)
LOG_FORMAT = '%(asctime)s [%(levelname)-8s] [%(name)s]: %(message)s'; LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
log_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT); log_handler = logging.StreamHandler(sys.stdout); log_handler.setFormatter(log_formatter)
logging.getLogger("aiohttp").setLevel(logging.WARNING); logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("nextcord").setLevel(logging.INFO)
root_logger = logging.getLogger(); root_logger.setLevel(LOG_LEVEL);
for handler in root_logger.handlers[:]: root_logger.removeHandler(handler);
root_logger.addHandler(log_handler); log = logging.getLogger(__name__)

def log_config():
    """Logs the loaded configuration."""
    config_log = logging.getLogger("ConfigLoader"); config_log.info("-" * 40); config_log.info(" Bot Configuration Loaded:"); config_log.info("-" * 40); config_log.info(f"  Log Level           : {LOG_LEVEL_STR} ({LOG_LEVEL})"); config_log.info(f"  Hash Size           : {HASH_SIZE}"); config_log.info(f"  Similarity Threshold: {SIMILARITY_THRESHOLD}"); config_log.info(f"  Max File Size (MB)  : {MAX_FILE_SIZE_MB}"); config_log.info(f"  Bot Prefix          : {BOT_COMMAND_PREFIX}"); config_log.info(f"  Black Frame Thresh  : {BLACK_FRAME_THRESHOLD}"); config_log.info(f"  Supported Images    : {' '.join(SUPPORTED_IMAGE_EXTENSIONS)}"); config_log.info(f"  Supported Videos    : {' '.join(SUPPORTED_VIDEO_EXTENSIONS)}"); config_log.info(f"  OpenCV Available    : {OPENCV_AVAILABLE}")
    config_log.info("-" * 40)

# --- Bot Intents ---
intents = nextcord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True

# --- Bot Class Definition (for state/session management) ---
class RepostBotState(commands.Bot):
    """Custom Bot class to manage state like the HTTP session."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        self._session_ready_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self._start_time = time.monotonic()
        log.info(f"RepostBotState __init__: Instance ID: {id(self)}.")

    @property
    async def http_session(self) -> aiohttp.ClientSession:
        """Provides the active aiohttp ClientSession, creating one if needed."""
        async with self._session_lock:
            if self._http_session is None or self._http_session.closed:
                log.info("HTTP_SESSION_PROP: Creating new session...")
                try:
                    self._http_session = aiohttp.ClientSession(headers={"User-Agent": USER_AGENT})
                    log.info(f"HTTP_SESSION_PROP: New session CREATED. ID: {id(self._http_session)}")
                    self._session_ready_event.set()
                except Exception as e:
                    log.critical(f"HTTP_SESSION_PROP: FAILED creation: {e}", exc_info=True)
                    self._session_ready_event.clear()
                    raise RuntimeError("Failed HTTP session init") from e
        if not self._session_ready_event.is_set():
             log.warning("HTTP_SESSION_PROP: Waiting for ready event...")
             try: await asyncio.wait_for(self._session_ready_event.wait(), timeout=15.0)
             except asyncio.TimeoutError: raise RuntimeError("HTTP session init timed out.")
        if self._http_session is None or self._http_session.closed: raise RuntimeError("HTTP Session invalid after ready event!")
        return self._http_session

    async def setup_hook(self):
        """Called after login but before gateway connection."""
        hook_start_time = time.monotonic()
        log.info("BOT: --- Running setup_hook ---")
        self.loop.create_task(self._create_http_session(), name="CreateHttpSessionTask")
        hook_duration = time.monotonic() - hook_start_time
        log.info(f"BOT: --- Setup_hook finished (total time: {hook_duration:.3f}s) ---")

    async def _create_http_session(self):
        """Internal helper to create the session."""
        async with self._session_lock:
            if self._http_session and not self._http_session.closed: return # Already exists
            try:
                if self._http_session and self._http_session.closed: self._http_session = None # Reset if closed
                self._http_session = aiohttp.ClientSession(headers={"User-Agent": USER_AGENT})
                self._session_ready_event.set()
                log.info(f"SESSION_CREATE_TASK: New session CREATED. ID: {id(self._http_session)}")
            except Exception as e:
                log.critical(f"SESSION_CREATE_TASK: FAILED create session: {e}", exc_info=True)
                self._http_session = None; self._session_ready_event.clear()

    async def close(self):
        """Closes the bot and its resources gracefully."""
        if self._shutdown_event.is_set(): return
        self._shutdown_event.set()
        log.warning("BOT: Close called...")
        if hasattr(self, '_http_session') and self._http_session and not self._http_session.closed:
            await self._http_session.close()
            log.info("BOT: Aiohttp session closed.")
        await super().close()
        shutdown_duration = time.monotonic() - self._start_time
        log.info(f"BOT: Shutdown complete. Uptime: {shutdown_duration:.2f} seconds.")

# --- Bot Instance Creation (using custom class) ---
bot = RepostBotState(command_prefix=BOT_COMMAND_PREFIX, intents=intents)

# --- Event Handlers (using @bot.event) ---
@bot.event
async def on_ready():
    """Called when the bot is fully connected and ready."""
    log.info(f"BOT: --- on_ready started ---")
    try:
        database.setup_database() # Setup DB on ready
    except Exception as e:
        log.critical(f"BOT: FATAL - DB setup failed in on_ready: {e}", exc_info=True)
        # Consider closing if DB fails: await bot.close()

    log.info("-" * 30); log.info(f'Logged in as: {bot.user.name} ({bot.user.id})'); log.info(f'Nextcord version: {nextcord.__version__}'); log.info(f'Connected to {len(bot.guilds)} guilds.'); log.info(f"BOT: Registered commands: {list(bot.all_commands.keys())}"); log.info("-" * 30)
    try:
        await bot.change_presence(activity=nextcord.Activity(type=nextcord.ActivityType.watching, name="for reposts"))
    except Exception as e: log.error(f"Failed to set presence: {e}")
    log.info(">>>> Repost Detector Bot is online and ready! <<<<")

@bot.event
async def on_disconnect(): log.warning("BOT: Disconnected from Discord Gateway.")
@bot.event
async def on_resumed(): log.info("BOT: Session resumed successfully.")


# --- Helper Functions ---
async def download_media(bot_instance: commands.Bot, url: str) -> Optional[bytes]:
    """Downloads media from a URL, respecting size limits."""
    try:
        session = await bot_instance.http_session
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
            if response.status != 200: log.warning(f"DOWNLOADER: HTTP {response.status} for {url}"); return None
            content_length = response.headers.get('Content-Length'); data = bytearray(); bytes_downloaded = 0
            if content_length and int(content_length) > MAX_FILE_SIZE_BYTES: log.warning(f"DOWNLOADER: Exceeds size via header: {url}"); return None
            async for chunk in response.content.iter_chunked(1024 * 128):
                bytes_downloaded += len(chunk); data.extend(chunk);
                if bytes_downloaded > MAX_FILE_SIZE_BYTES: log.warning(f"DOWNLOADER: Exceeded max size during download: {url}"); return None
            log.info(f"DOWNLOADER: Downloaded {bytes_downloaded / (1024*1024):.2f} MB from {url}"); return bytes(data)
    except Exception as e: log.error(f"DOWNLOADER: Error downloading {url}: {e}", exc_info=True); return None

async def get_image_phash(image_bytes: bytes) -> Optional[imagehash.ImageHash]:
    """Calculates the pHash for an image or the first frame of a GIF."""
    try:
        loop = asyncio.get_running_loop()
        img = await loop.run_in_executor(None, Image.open, io.BytesIO(image_bytes))
        target_frame = img
        if getattr(img, "is_animated", False) and getattr(img, "n_frames", 1) > 1:
            iterator = ImageSequence.Iterator(img)
            target_frame = await loop.run_in_executor(None, next(iterator).copy) # Get first frame
        if target_frame.mode not in ('L', 'RGB'): target_frame = await loop.run_in_executor(None, target_frame.convert, 'RGB')
        hash_val = await loop.run_in_executor(None, imagehash.phash, target_frame, HASH_SIZE)
        img.close()
        log.info(f"HASHING: Hashed IMAGE/GIF OK. Hash: {hash_val}"); return hash_val
    except Exception as e: log.error(f"HASHING: Error hashing image: {e}", exc_info=True); return None

def _is_frame_black(frame: Optional[np.ndarray], threshold: int) -> bool:
    """Checks if an OpenCV frame is likely black or very dark."""
    if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0: return False
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) < threshold
    except Exception as e: log.error(f"FRAME_CHECK: Error: {e}", exc_info=True); return False

def _blocking_video_multi_hash_cv2(video_bytes: bytes, hash_size_local: int, black_thresh: int) -> Optional[List[imagehash.ImageHash]]:
    """Extracts and hashes keyframes from video bytes using OpenCV."""
    if not OPENCV_AVAILABLE or cv2 is None: log.error("CV2 Hash: OpenCV unavailable."); return None
    tmp_filepath, cap = None, None; hashes: List[imagehash.ImageHash] = []
    try:
        with tempfile.NamedTemporaryFile(prefix="repvid_", suffix=".tmp", delete=False) as tmp_file:
            tmp_file.write(video_bytes); tmp_filepath = tmp_file.name
        cap = cv2.VideoCapture(tmp_filepath)
        if not cap.isOpened(): log.error(f"CV2 Hash: Failed to open video: {tmp_filepath}"); return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0: log.error(f"CV2 Hash: Video has no frames: {tmp_filepath}"); return None

        first_idx, mid_idx, last_idx = 0, max(0, frame_count // 2), max(0, frame_count - 1)
        actual_first_idx = 0 # Default
        max_scan = min(frame_count, max(10, frame_count // 10))
        for idx in range(max_scan): # Find first non-black frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ret, frame = cap.read()
            if ret and not _is_frame_black(frame, black_thresh): actual_first_idx = idx; break

        processed_indices = set()
        indices_to_hash = [actual_first_idx, mid_idx, last_idx]
        for target_idx in indices_to_hash:
            if target_idx in processed_indices: continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx); ret, frame = cap.read()
            if ret and frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    hash_val = imagehash.phash(pil_image, hash_size=hash_size_local)
                    hashes.append(hash_val)
                    pil_image.close()
                    processed_indices.add(target_idx)
                except Exception as e_hash: log.error(f"CV2 Hash: Error hashing frame {target_idx}: {e_hash}", exc_info=True)
            else: log.warning(f"CV2 Hash: Failed to read frame {target_idx}")
    except Exception as e: log.error(f"CV2 Hash: Error during video processing: {e}", exc_info=True); hashes = []
    finally:
        if cap: cap.release()
        if tmp_filepath and os.path.exists(tmp_filepath):
            try: os.remove(tmp_filepath)
            except OSError as ed: log.error(f"CV2 Hash: Error deleting temp file {tmp_filepath}: {ed}")
    unique_hashes = list(dict.fromkeys(hashes)) # Remove duplicates while preserving order
    log.info(f"CV2 Hash: Video hashing generated {len(unique_hashes)} unique hashes.")
    return unique_hashes if unique_hashes else None

async def get_video_multi_frame_phashes(video_bytes: bytes) -> Optional[List[imagehash.ImageHash]]:
    """Async wrapper for multi-frame video hashing."""
    if not OPENCV_AVAILABLE: log.error("HASHING: OpenCV unavailable for video."); return None
    try:
        loop = asyncio.get_running_loop()
        hash_list = await loop.run_in_executor(None, _blocking_video_multi_hash_cv2, video_bytes, HASH_SIZE, BLACK_FRAME_THRESHOLD)
        return hash_list
    except Exception as e: log.error(f"HASHING: Error in async video hash wrapper: {e}", exc_info=True); return None

async def handle_repost(bot_instance: commands.Bot, repost_message: nextcord.Message, original_post_info: dict):
    """Handles sending alert and deleting repost, but only if setup is complete."""
    if not repost_message or not repost_message.guild or not repost_message.channel: return
    guild_id = repost_message.guild.id; repost_channel = repost_message.channel
    alert_channel_id = database.get_alert_channel(guild_id)

    # --- Setup Check ---
    if alert_channel_id is None:
        log.warning(f"HANDLE_REPOST: Guild {guild_id} setup incomplete. Halting action.")
        try:
            if repost_channel.permissions_for(repost_message.guild.me).send_messages:
                setup_msg = (f"⚠️ **Repost Bot Setup Required!**\nA repost by {repost_message.author.mention} was detected. An admin must run `{bot_instance.command_prefix}setalertchannel` in the desired alert channel.")
                await repost_channel.send(setup_msg, delete_after=120)
        except Exception as e_setup_msg: log.error(f"HANDLE_REPOST: Failed send setup reminder: {e_setup_msg}")
        return

    # --- Proceed with handling ---
    alert_target_channel: Optional[nextcord.abc.GuildChannel] = None
    try: alert_target_channel = await bot_instance.fetch_channel(alert_channel_id)
    except (nextcord.NotFound, nextcord.Forbidden, Exception) as e: log.warning(f"HANDLE_REPOST: Failed validate alert channel {alert_channel_id}: {e}. Falling back.")
    if not isinstance(alert_target_channel, (nextcord.TextChannel, nextcord.Thread, nextcord.VoiceChannel)): alert_target_channel = repost_channel # Fallback

    can_send_alert = alert_target_channel.permissions_for(repost_message.guild.me).send_messages
    can_delete_repost = repost_channel.permissions_for(repost_message.guild.me).manage_messages

    original_author_name = f"User ID: {original_post_info['author_id']}"
    try:
        original_author = repost_message.guild.get_member(original_post_info['author_id']) or await bot_instance.fetch_user(original_post_info['author_id'])
        if original_author: original_author_name = getattr(original_author, 'display_name', original_author.name)
    except Exception: pass # Ignore fetch errors

    repost_alert_prefix = "⚠️ **Repost Alert & Removed!**" if can_delete_repost else "⚠️ **Repost Alert!**"
    alert_content = (f"{repost_alert_prefix} {repost_message.author.mention}, this looks very similar (Similarity: {original_post_info.get('similarity', 'N/A')})...\n"
                     f"Original by **{original_author_name}** on <t:{int(original_post_info['timestamp'])}:f> (<t:{int(original_post_info['timestamp'])}:R>)\n"
                     f"🔗 Original post: {original_post_info['link']}")
    if alert_target_channel.id != repost_channel.id: alert_content += f"\n*(Detected in {repost_channel.mention})*"

    if can_send_alert:
        try: await alert_target_channel.send(alert_content, allowed_mentions=nextcord.AllowedMentions(users=True))
        except Exception as e: log.error(f"HANDLE_REPOST: Error sending alert to #{alert_target_channel.name}: {e}")
    else: log.warning(f"HANDLE_REPOST: Skipping alert (no permission in target channel).")

    if can_delete_repost:
        try: await repost_message.delete()
        except Exception as e: log.error(f"HANDLE_REPOST: Error deleting repost {repost_message.id}: {e}")
    else: log.warning(f"HANDLE_REPOST: Skipping delete (no permission).")


async def process_media(bot_instance: commands.Bot, message: nextcord.Message, media_url: str, source_description: str, media_type: str):
    """Downloads, hashes, checks DB, and triggers action for a single media item."""
    task_id = f"MsgID {message.id} ({media_type} via {source_description.split(' ')[0]})"; log.info(f">>> PROCESS_MEDIA [{task_id}]: START")
    if not message.guild: return
    try: message = await message.channel.fetch_message(message.id) # Re-fetch for consistency
    except Exception: log.warning(f"PROCESS_MEDIA [{task_id}]: Msg deleted/inaccessible."); return
    if not message or not message.guild: return

    media_bytes = await download_media(bot_instance, media_url)
    if not media_bytes: log.warning(f"PROCESS_MEDIA [{task_id}]: Download failed."); return

    current_hashes: Optional[Union[ImageHash, List[ImageHash]]] = None
    if media_type == 'image': current_hashes = await get_image_phash(media_bytes)
    elif media_type == 'video': current_hashes = await get_video_multi_frame_phashes(media_bytes)
    if not current_hashes: log.warning(f"PROCESS_MEDIA [{task_id}]: Hashing failed."); return

    try:
        existing_post = database.find_similar_hash(message.guild.id, current_hashes, SIMILARITY_THRESHOLD)
        if existing_post and existing_post["message_id"] != message.id:
            log.info(f"!!! PROCESS_MEDIA [{task_id}]: REPOST DETECTED !!! Similar to {existing_post['message_id']}");
            await handle_repost(bot_instance, message, existing_post)
        elif not existing_post:
            log.info(f"PROCESS_MEDIA [{task_id}]: No match found. Adding new hash(es).");
            database.add_hash(message.guild.id, message.channel.id, message.id, message.author.id, current_hashes, media_url)
    except Exception as e: log.error(f"PROCESS_MEDIA [{task_id}]: Error during DB/Action phase: {e}", exc_info=True)
    log.info(f"<<< PROCESS_MEDIA [{task_id}]: END")


# --- on_message (Main Message Processing) ---
@bot.event
async def on_message(message: nextcord.Message):
    """Handles incoming messages, checks for commands or media."""
    if message.author.bot or not message.guild: return
    if message.content.startswith(bot.command_prefix): await bot.process_commands(message); return

    # Whitelist Check
    if database.is_channel_whitelisted(message.guild.id, message.channel.id): return

    if not message.attachments and not message.embeds: return
    if message.type not in (nextcord.MessageType.default, nextcord.MessageType.reply): return

    # Basic permission check
    if not isinstance(message.channel, (nextcord.TextChannel, nextcord.Thread, nextcord.VoiceChannel)): return
    try:
        perms = message.channel.permissions_for(message.guild.me)
        if not perms.read_message_history or not perms.send_messages or not perms.manage_messages:
            log.warning(f"ON_MESSAGE [{message.id}]: Missing baseline permissions in Ch:{message.channel.id}. Skipping media check.")
            return
    except Exception: return # Ignore channels we can't check perms in

    # Identify and queue media processing tasks
    tasks_to_create = []
    processed_urls = set()

    for attachment in message.attachments:
        if attachment.url in processed_urls or not (0 < attachment.size <= MAX_FILE_SIZE_BYTES): continue
        file_ext = os.path.splitext(attachment.filename)[1].lower(); media_type = None
        if file_ext in SUPPORTED_IMAGE_EXTENSIONS: media_type = 'image'
        elif file_ext in SUPPORTED_VIDEO_EXTENSIONS and OPENCV_AVAILABLE: media_type = 'video'
        if media_type:
            tasks_to_create.append(process_media(bot, message, attachment.url, f"attachment '{attachment.filename}'", media_type))
            processed_urls.add(attachment.url)

    for embed in message.embeds:
        media_url, potential_media_type = None, None
        if embed.video and embed.video.url: media_url, potential_media_type = embed.video.url, 'video'
        elif embed.image and embed.image.url: media_url, potential_media_type = embed.image.url, 'image'
        elif embed.thumbnail and embed.thumbnail.url: media_url, potential_media_type = embed.thumbnail.url, 'image'
        if not media_url or media_url in processed_urls: continue

        final_media_type = None
        try:
            file_ext = os.path.splitext(media_url.split('?')[0])[1].lower()
            if file_ext in SUPPORTED_IMAGE_EXTENSIONS: final_media_type = 'image'
            elif file_ext in SUPPORTED_VIDEO_EXTENSIONS and OPENCV_AVAILABLE: final_media_type = 'video'
            elif not file_ext: final_media_type = potential_media_type # Trust embed type if no extension
        except Exception: continue # Ignore URL parsing errors

        if final_media_type == 'video' and not OPENCV_AVAILABLE: continue
        if final_media_type:
            tasks_to_create.append(process_media(bot, message, media_url, f"{embed.type} embed", final_media_type))
            processed_urls.add(media_url)

    if tasks_to_create:
        log.info(f"ON_MESSAGE [{message.id}]: Running {len(tasks_to_create)} media processing tasks...")
        results = await asyncio.gather(*tasks_to_create, return_exceptions=True)
        for idx, res in enumerate(results):
             if isinstance(res, Exception): log.error(f"ON_MESSAGE [{message.id}]: Task {idx+1} failed: {res}", exc_info=False)


# --- Commands (using @bot.command) ---
@bot.command(name="setalertchannel", help="REQUIRED setup: Run in the channel for repost alerts. Needs 'Manage Server'.", aliases=['setalerts'])
@commands.guild_only()
@commands.has_permissions(manage_guild=True)
async def set_alert_channel_command(ctx: commands.Context):
    """Sets the alert channel to the command's channel."""
    if not isinstance(ctx.channel, nextcord.TextChannel): await ctx.reply("⚠️ Must be run in a text channel.", delete_after=30); return
    try:
        perms = ctx.channel.permissions_for(ctx.guild.me)
        if not perms.send_messages or not perms.read_message_history:
            await ctx.reply(f"⚠️ I need `Send Messages` and `Read Message History` permissions in {ctx.channel.mention}.")
            return
    except Exception as e: await ctx.reply("❌ Error checking permissions."); log.error(f"CMD_SETALERT Perm check failed: {e}"); return

    if database.set_alert_channel(ctx.guild.id, ctx.channel.id):
        await ctx.reply(f"✅ Repost alerts active. Will send alerts to {ctx.channel.mention}.")
    else: await ctx.reply("❌ Error saving channel setting.")

@set_alert_channel_command.error
async def set_alert_channel_error(ctx: commands.Context, error: commands.CommandError):
    """Error handler for setalertchannel."""
    if isinstance(error, commands.MissingPermissions): await ctx.reply("🚫 'Manage Server' permission required.")
    elif isinstance(error, commands.NoPrivateMessage): await ctx.reply("🚫 Use in a server text channel.")
    else: log.error(f"CMD_SETALERT Error: {error}", exc_info=True); await ctx.reply("❌ Unexpected error.")

@bot.command(name="whitelist", help="Ignore reposts in a channel. Usage: !whitelist [#channel/ID]. Needs 'Manage Server'.", aliases=['wl'])
@commands.guild_only()
@commands.has_permissions(manage_guild=True)
async def whitelist_command(ctx: commands.Context, channel: Optional[nextcord.TextChannel] = None):
    """Adds a channel to the repost check whitelist."""
    target_channel = channel or ctx.channel
    if not isinstance(target_channel, nextcord.TextChannel): await ctx.reply("⚠️ Please specify a valid text channel.", delete_after=30); return
    added = database.add_whitelist_channel(ctx.guild.id, target_channel.id)
    if added: await ctx.reply(f"✅ Repost checks ignored in {target_channel.mention}.")
    elif database.is_channel_whitelisted(ctx.guild.id, target_channel.id): await ctx.reply(f"ℹ️ {target_channel.mention} already whitelisted.")
    else: await ctx.reply(f"❌ Error whitelisting {target_channel.mention}.")

@whitelist_command.error
async def whitelist_error(ctx: commands.Context, error: commands.CommandError):
    """Error handler for whitelist."""
    if isinstance(error, commands.MissingPermissions): await ctx.reply("🚫 'Manage Server' permission required.")
    elif isinstance(error, (commands.ChannelNotFound, commands.BadArgument)): await ctx.reply(f"❓ Couldn't find channel. Use command in channel or provide #channel/ID.")
    elif isinstance(error, commands.NoPrivateMessage): await ctx.reply("🚫 Use in a server.")
    else: log.error(f"CMD_WHITELIST Error: {error}", exc_info=True); await ctx.reply("❌ Unexpected error.")

@bot.command(name="unwhitelist", help="Re-enable repost checks in a channel. Usage: !unwhitelist [#channel/ID]. Needs 'Manage Server'.", aliases=['unwl'])
@commands.guild_only()
@commands.has_permissions(manage_guild=True)
async def unwhitelist_command(ctx: commands.Context, channel: Optional[nextcord.TextChannel] = None):
    """Removes a channel from the repost check whitelist."""
    target_channel = channel or ctx.channel
    if not isinstance(target_channel, nextcord.TextChannel): await ctx.reply("⚠️ Please specify a valid text channel.", delete_after=30); return
    removed = database.remove_whitelist_channel(ctx.guild.id, target_channel.id)
    if removed: await ctx.reply(f"✅ Repost checks re-enabled in {target_channel.mention}.")
    elif not database.is_channel_whitelisted(ctx.guild.id, target_channel.id): await ctx.reply(f"ℹ️ {target_channel.mention} was not whitelisted.")
    else: await ctx.reply(f"❌ Error unwhitelisting {target_channel.mention}.")

@unwhitelist_command.error
async def unwhitelist_error(ctx: commands.Context, error: commands.CommandError):
    """Error handler for unwhitelist."""
    if isinstance(error, commands.MissingPermissions): await ctx.reply("🚫 'Manage Server' permission required.")
    elif isinstance(error, (commands.ChannelNotFound, commands.BadArgument)): await ctx.reply(f"❓ Couldn't find channel. Use command in channel or provide #channel/ID.")
    elif isinstance(error, commands.NoPrivateMessage): await ctx.reply("🚫 Use in a server.")
    else: log.error(f"CMD_UNWHITELIST Error: {error}", exc_info=True); await ctx.reply("❌ Unexpected error.")


# --- Graceful Shutdown Signal Handling ---
async def shutdown_signal_handler(bot_instance: commands.Bot, signal_type: signal.Signals):
    """Handles OS signals for graceful shutdown."""
    if not isinstance(bot_instance, RepostBotState) or bot_instance._shutdown_event.is_set(): return
    log.warning(f"SIGNAL: Received {signal_type.name}. Initiating shutdown...");
    if not bot_instance.is_closed(): asyncio.create_task(bot_instance.close(), name="SignalShutdownTask")


# --- Main Execution Block ---
if __name__ == "__main__":
    if not BOT_TOKEN: log.critical("BOT: FATAL - DISCORD_BOT_TOKEN not set!"); sys.exit(1)
    try: log_config()
    except Exception as cfg_log_err: log.error(f"Failed log config: {cfg_log_err}")

    try: loop = asyncio.get_event_loop()
    except RuntimeError: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)

    graceful_signals = (signal.SIGINT, signal.SIGTERM) if os.name != 'nt' else (signal.SIGINT,)
    try:
        for sig in graceful_signals: loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown_signal_handler(bot, s)))
    except Exception as e: log.error(f"Error setting signal handlers: {e}", exc_info=True)

    try:
        log.info(f"--- Starting Repost Detector Bot ({USER_AGENT}) ---")
        bot.run(BOT_TOKEN)
    except nextcord.LoginFailure: log.critical("BOT: Login failed. Check token.")
    except Exception as e: log.critical(f"BOT: Bot run failed: {e}", exc_info=True)
    finally:
        log.info("BOT: Main execution scope finished.")
        # Final cleanup logic remains the same
        if isinstance(bot, RepostBotState) and not bot.is_closed() and not bot._shutdown_event.is_set():
             log.warning("BOT: Forcing final close...")
             try:
                 if loop.is_running(): loop.run_until_complete(asyncio.wait_for(bot.close(), timeout=10.0))
                 else: asyncio.run(bot.close()) # Run in new loop if needed
             except Exception as fe: log.error(f"Error during final forced close: {fe}", exc_info=True)
        if not loop.is_closed():
            tasks = [t for t in asyncio.all_tasks(loop=loop) if not t.done()]
            if tasks:
                log.info(f"BOT: Cancelling {len(tasks)} outstanding tasks...");
                for task in tasks: task.cancel()
                try: loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                except Exception: pass # Ignore errors during cleanup gather
            if loop.is_running(): loop.stop()
            loop.close(); log.info("BOT: Loop closed.")
        log.info(f"--- Repost Detector Bot ({USER_AGENT}) process finished ---")