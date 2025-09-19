# repostBot.py
import nextcord
from nextcord.ext import commands, tasks
import os
import io
import logging
import asyncio
import aiohttp
from PIL import Image, UnidentifiedImageError, ImageSequence
import imagehash # Ensure this is imported in repostBot.py as well if used directly
from dotenv import load_dotenv
import time
from typing import Optional, Union, Dict, Any, List, Tuple, Set
import sys
import signal
import tempfile
import numpy as np
import shutil
import re
import json

_initial_log = logging.getLogger("InitialSetup")
try:
    import cv2; OPENCV_AVAILABLE = True; _initial_log.info("OpenCV imported.")
except ImportError: _initial_log.warning("OpenCV not found. Video processing DISABLED."); OPENCV_AVAILABLE = False; cv2 = None
except Exception as e: _initial_log.warning(f"OpenCV import error: {e}. Video disabled."); OPENCV_AVAILABLE = False; cv2 = None

try: import database; _initial_log.debug("Database imported.")
except Exception as e: print(f"CRITICAL DB Import Error: {e}"); sys.exit(1)

load_dotenv()
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
try:
    HASH_SIZE = int(os.getenv("HASH_SIZE", "8"))
    # GLOBAL_SIMILARITY_THRESHOLD from .env serves as a fallback
    GLOBAL_SIMILARITY_THRESHOLD = int(os.getenv("SIMILARITY_THRESHOLD", "5"))
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "30")); LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
    BOT_COMMAND_PREFIX = os.getenv("BOT_PREFIX", "!")
    BLACK_FRAME_THRESHOLD = int(os.getenv("BLACK_FRAME_THRESHOLD", "10"))
except Exception as e:
    print(f"ERROR parsing .env: {e}"); HASH_SIZE=8; GLOBAL_SIMILARITY_THRESHOLD=5; MAX_FILE_SIZE_MB=30; LOG_LEVEL_STR="INFO"; BOT_COMMAND_PREFIX="!"; BLACK_FRAME_THRESHOLD = 10

MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB*1024*1024
SUPPORTED_IMAGE_EXTENSIONS = ('.png','.jpg','.jpeg','.gif','.bmp','.webp')
SUPPORTED_VIDEO_EXTENSIONS = ('.mp4', '.webm', '.mov', '.avi', '.mkv', '.flv', '.wmv')
USER_AGENT = "DiscordBot RepostDetector (v1.25 - GuildThresholdCmd)"

LOG_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}; LOG_LEVEL = LOG_LEVELS.get(LOG_LEVEL_STR, logging.INFO)
LOG_FORMAT = '%(asctime)s [%(levelname)-8s] [%(name)s]: %(message)s'; LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
log_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT); log_handler = logging.StreamHandler(sys.stdout); log_handler.setFormatter(log_formatter)
logging.getLogger("aiohttp").setLevel(logging.WARNING); logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("nextcord").setLevel(logging.INFO)
root_logger = logging.getLogger(); root_logger.setLevel(LOG_LEVEL);
for handler in root_logger.handlers[:]: root_logger.removeHandler(handler);
root_logger.addHandler(log_handler); log = logging.getLogger(__name__)

def log_config():
    config_log = logging.getLogger("ConfigLoader"); config_log.info("-" * 40); config_log.info(" Bot Configuration Loaded:"); config_log.info("-" * 40); config_log.info(f"  Log Level           : {LOG_LEVEL_STR} ({LOG_LEVEL})"); config_log.info(f"  Hash Size           : {HASH_SIZE}"); config_log.info(f"  Similarity Threshold (Global Default): {GLOBAL_SIMILARITY_THRESHOLD}"); config_log.info(f"  Max File Size (MB)  : {MAX_FILE_SIZE_MB}"); config_log.info(f"  Bot Prefix          : {BOT_COMMAND_PREFIX}"); config_log.info(f"  Black Frame Thresh  : {BLACK_FRAME_THRESHOLD}"); config_log.info(f"  OpenCV Available    : {OPENCV_AVAILABLE}"); config_log.info("-" * 40)

intents = nextcord.Intents.default()
intents.message_content = True; intents.messages = True; intents.guilds = True

class RepostBotState(commands.Bot):
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
        async with self._session_lock:
            if self._http_session is None or self._http_session.closed:
                log.info("HTTP_SESSION_PROP: Creating new session...")
                try: self._http_session = aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}); self._session_ready_event.set()
                except Exception as e: log.critical(f"HTTP_SESSION_PROP: FAILED creation: {e}", exc_info=True); self._session_ready_event.clear(); raise RuntimeError("Failed HTTP session init") from e
        if not self._session_ready_event.is_set():
             try: await asyncio.wait_for(self._session_ready_event.wait(), timeout=15.0)
             except asyncio.TimeoutError: raise RuntimeError("HTTP session init timed out.")
        if self._http_session is None or self._http_session.closed: raise RuntimeError("HTTP Session invalid after ready event!")
        return self._http_session

    async def setup_hook(self):
        log.info("BOT: --- Running setup_hook ---")
        self.loop.create_task(self._create_http_session(), name="CreateHttpSessionTask")
        log.info("BOT: --- Setup_hook finished ---")

    async def _create_http_session(self):
        async with self._session_lock:
            if self._http_session and not self._http_session.closed: return
            try:
                if self._http_session and self._http_session.closed: self._http_session = None
                self._http_session = aiohttp.ClientSession(headers={"User-Agent": USER_AGENT})
                self._session_ready_event.set(); log.info(f"SESSION_CREATE_TASK: New session CREATED.")
            except Exception as e: log.critical(f"SESSION_CREATE_TASK: FAILED create session: {e}", exc_info=True); self._http_session = None; self._session_ready_event.clear()

    async def close(self):
        if self._shutdown_event.is_set(): return
        self._shutdown_event.set(); log.warning("BOT: Close called...")
        if hasattr(self, '_http_session') and self._http_session and not self._http_session.closed:
            await self._http_session.close(); log.info("BOT: Aiohttp session closed.")
        await super().close()
        log.info(f"BOT: Shutdown complete. Uptime: {time.monotonic() - self._start_time:.2f}s.")

bot = RepostBotState(command_prefix=BOT_COMMAND_PREFIX, intents=intents)

@bot.event
async def on_ready():
    log.info(f"BOT: --- on_ready started ---")
    try: database.setup_database()
    except Exception as e: log.critical(f"BOT: FATAL - DB setup failed in on_ready: {e}", exc_info=True)
    log.info("-" * 30); log.info(f'Logged in as: {bot.user.name} ({bot.user.id})'); log.info(f'Nextcord version: {nextcord.__version__}'); log.info(f'Connected to {len(bot.guilds)} guilds.'); log.info(f"BOT: Registered commands: {list(bot.all_commands.keys())}"); log.info("-" * 30)
    try: await bot.change_presence(activity=nextcord.Activity(type=nextcord.ActivityType.watching, name="for reposts"))
    except Exception as e: log.error(f"Failed to set presence: {e}")
    log.info(">>>> Repost Detector Bot is online and ready! <<<<")
    update_stats_task.start()

@bot.event
async def on_disconnect(): log.warning("BOT: Disconnected from Discord Gateway.")
@bot.event
async def on_resumed(): log.info("BOT: Session resumed successfully.")

async def download_media(bot_instance: commands.Bot, url: str) -> Optional[bytes]:
    try:
        session = await bot_instance.http_session
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
            if response.status != 200: log.warning(f"DOWNLOADER: HTTP {response.status} for {url}"); return None
            content_length = response.headers.get('Content-Length'); data = bytearray(); bytes_downloaded = 0
            if content_length:
                 try:
                     if int(content_length) > MAX_FILE_SIZE_BYTES: log.warning(f"DOWNLOADER: Exceeds size via header: {url}"); return None
                 except ValueError: pass
            async for chunk in response.content.iter_chunked(1024 * 128):
                bytes_downloaded += len(chunk); data.extend(chunk);
                if bytes_downloaded > MAX_FILE_SIZE_BYTES: log.warning(f"DOWNLOADER: Exceeded max size: {url}"); return None
            log.info(f"DOWNLOADER: Downloaded {bytes_downloaded / (1024*1024):.2f} MB from {url}"); return bytes(data)
    except Exception as e: log.error(f"DOWNLOADER: Error downloading {url}: {e}", exc_info=True); return None

async def get_image_phash(image_bytes: bytes) -> Optional[imagehash.ImageHash]:
    try:
        loop = asyncio.get_running_loop()
        img = await loop.run_in_executor(None, Image.open, io.BytesIO(image_bytes))
        target_frame = img
        if getattr(img, "is_animated", False) and getattr(img, "n_frames", 1) > 1:
            iterator = ImageSequence.Iterator(img)
            target_frame = await loop.run_in_executor(None, next(iterator).copy)
        if target_frame.mode not in ('L', 'RGB'): target_frame = await loop.run_in_executor(None, target_frame.convert, 'RGB')
        hash_val = await loop.run_in_executor(None, imagehash.phash, target_frame, HASH_SIZE)
        img.close(); log.info(f"HASHING: Hashed IMAGE/GIF OK. Hash: {hash_val}"); return hash_val
    except Exception as e: log.error(f"HASHING: Error hashing image: {e}", exc_info=True); return None

def _is_frame_black(frame: Optional[np.ndarray], threshold: int) -> bool:
    if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0: return False
    try: return np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) < threshold
    except Exception as e: log.error(f"FRAME_CHECK: Error: {e}", exc_info=True); return False

def _blocking_video_multi_hash_cv2(video_bytes: bytes, hash_size_local: int, black_thresh: int) -> Optional[List[imagehash.ImageHash]]:
    if not OPENCV_AVAILABLE or cv2 is None: log.error("CV2 Hash: OpenCV unavailable."); return None
    tmp_filepath, cap = None, None; unique_hashes_set: Set[imagehash.ImageHash] = set(); processed_indices = set()
    try:
        with tempfile.NamedTemporaryFile(prefix="repvid_", suffix=".tmp", delete=False) as tmp_file: tmp_filepath = tmp_file.name; tmp_file.write(video_bytes)
        cap = cv2.VideoCapture(tmp_filepath)
        if not cap.isOpened(): log.error(f"CV2 Hash: Failed to open video: {tmp_filepath}"); return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0: log.error(f"CV2 Hash: Video has no frames: {tmp_filepath}"); return None
        log.info(f"CV2 Hash: Video Frame Count={frame_count}")
        actual_first_idx = -1
        for idx in range(min(frame_count, max(10, frame_count // 10))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ret, frame = cap.read()
            if ret and frame is not None and not _is_frame_black(frame, black_thresh): actual_first_idx = idx; log.debug(f"CV2 Hash: First non-black frame index {actual_first_idx}."); break
        else: actual_first_idx = 0; log.warning(f"CV2 Hash: No non-black in scan. Using index 0.")
        middle_idx = max(0, frame_count // 2)
        actual_last_idx = -1
        for i in range(min(frame_count, max(10, frame_count // 10))):
            idx = max(0, frame_count - 1 - i)
            if idx < 0: break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ret, frame = cap.read()
            if ret and frame is not None and not _is_frame_black(frame, black_thresh): actual_last_idx = idx; log.debug(f"CV2 Hash: Last non-black frame index {actual_last_idx}."); break
        else: actual_last_idx = max(0, frame_count - 1); log.warning(f"CV2 Hash: No non-black end frame. Using index {actual_last_idx}.")
        for target_idx, frame_name in [(actual_first_idx, "FirstNB"), (middle_idx, "Mid"), (actual_last_idx, "LastNB")]:
            if target_idx == -1 or target_idx in processed_indices: continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx); ret, frame = cap.read()
            if ret and frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                if _is_frame_black(frame, black_thresh): log.warning(f"CV2 Hash: Selected '{frame_name}' frame {target_idx} is black. Skipping hash."); processed_indices.add(target_idx); continue
                try:
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    hash_val = imagehash.phash(pil_image, hash_size=hash_size_local)
                    unique_hashes_set.add(hash_val); pil_image.close(); log.debug(f"CV2 Hash: Hashed '{frame_name}' frame {target_idx}. Hash: {hash_val}")
                except Exception as e_hash: log.error(f"CV2 Hash: Error hashing '{frame_name}' frame {target_idx}: {e_hash}", exc_info=True)
            else: log.warning(f"CV2 Hash: Failed read/invalid '{frame_name}' frame {target_idx}.")
            processed_indices.add(target_idx)
    except cv2.error as e:
        log.error(f"CV2 Hash: OpenCV error during video processing: {e}", exc_info=True)
    except Exception as e: log.error(f"CV2 Hash: Error during video processing: {e}", exc_info=True)
    finally:
        if cap: cap.release()
        if tmp_filepath and os.path.exists(tmp_filepath):
            try: os.remove(tmp_filepath); log.debug("CV2 Hash: Temp file deleted.")
            except Exception as e_del: log.error(f"CV2 Hash: Unexpected error deleting temp file {tmp_filepath}: {e_del}")
    final_hashes_list = list(unique_hashes_set)
    log.info(f"CV2 Hash: Video hashing generated {len(final_hashes_list)} unique, non-black keyframe hashes.")
    return final_hashes_list if final_hashes_list else None

async def get_video_multi_frame_phashes(video_bytes: bytes) -> Optional[List[imagehash.ImageHash]]:
    if not OPENCV_AVAILABLE: log.error("HASHING: OpenCV unavailable for video."); return None
    try: return await asyncio.get_running_loop().run_in_executor(None, _blocking_video_multi_hash_cv2, video_bytes, HASH_SIZE, BLACK_FRAME_THRESHOLD)
    except Exception as e: log.error(f"HASHING: Error in async video hash wrapper: {e}", exc_info=True); return None

async def handle_repost(bot_instance: commands.Bot, repost_message: nextcord.Message, original_post_info: dict):
    if not repost_message or not repost_message.guild or not repost_message.channel: return
    guild_id = repost_message.guild.id; repost_channel = repost_message.channel
    alert_channel_id = database.get_alert_channel(guild_id)
    if alert_channel_id is None:
        log.warning(f"HANDLE_REPOST: Guild {guild_id} setup incomplete. Halting action.")
        try:
            if repost_channel.permissions_for(repost_message.guild.me).send_messages:
                await repost_channel.send((f"⚠️ **Repost Bot Setup Required!**\nAdmin run `{bot_instance.command_prefix}setalertchannel` in the desired alert channel."), delete_after=120)
        except Exception as e: log.error(f"HANDLE_REPOST: Failed send setup reminder: {e}")
        return
    alert_target_channel: Optional[nextcord.abc.GuildChannel] = None
    try: alert_target_channel = await bot_instance.fetch_channel(alert_channel_id)
    except Exception as e: log.warning(f"HANDLE_REPOST: Failed validate alert channel {alert_channel_id}: {e}. Falling back.")
    if not isinstance(alert_target_channel, (nextcord.TextChannel, nextcord.Thread, nextcord.VoiceChannel)): alert_target_channel = repost_channel
    perms_alert_target = alert_target_channel.permissions_for(repost_message.guild.me)
    perms_repost_channel = repost_channel.permissions_for(repost_message.guild.me)
    can_send_alert = perms_alert_target.send_messages and perms_alert_target.embed_links
    can_delete_repost = perms_repost_channel.manage_messages
    original_author_name = f"User ID: {original_post_info['author_id']}"
    try:
        original_author = repost_message.guild.get_member(original_post_info['author_id']) or await bot_instance.fetch_user(original_post_info['author_id'])
        if original_author: original_author_name = f"{original_author.display_name} ({original_author})"
    except Exception: pass
    # Use the effective threshold for display
    effective_threshold = database.get_guild_similarity_threshold(guild_id) or GLOBAL_SIMILARITY_THRESHOLD
    similarity_score = original_post_info.get('similarity', 'N/A')
    original_timestamp = int(original_post_info['timestamp'])
    original_link = original_post_info['link']
    embed_title = "🗑️ Repost Detected & Removed!" if can_delete_repost else "⚠️ Repost Detected!"
    embed_color = nextcord.Color.red() if can_delete_repost else nextcord.Color.orange()
    embed = nextcord.Embed(title=embed_title, description=f"{repost_message.author.mention} posted content similar to an earlier post.", color=embed_color, timestamp=repost_message.created_at)
    embed.add_field(name="Original Post", value=f"[Link]({original_link})", inline=True)
    embed.add_field(name="Original Poster", value=original_author_name, inline=True)
    embed.add_field(name="Similarity", value=f"`{similarity_score}` (Thresh: `{effective_threshold}`)", inline=True)
    embed.add_field(name="Original Post Time", value=f"<t:{original_timestamp}:F> (<t:{original_timestamp}:R>)", inline=False)
    if alert_target_channel.id != repost_channel.id: embed.add_field(name="Detected In", value=repost_channel.mention, inline=False)
    embed.set_footer(text=f"Repost Bot | Repost Msg ID: {repost_message.id}")
    if can_send_alert:
        try: await alert_target_channel.send(embed=embed, allowed_mentions=nextcord.AllowedMentions(users=True))
        except nextcord.Forbidden: log.error(f"HANDLE_REPOST: Missing Send/Embed permission in #{alert_target_channel.name}.")
        except Exception as e: log.error(f"HANDLE_REPOST: Error sending alert embed: {e}", exc_info=True)
    if can_delete_repost:
        try: await repost_message.delete()
        except Exception as e: log.error(f"HANDLE_REPOST: Error deleting repost {repost_message.id}: {e}", exc_info=True)

    database.log_repost(guild_id, repost_message.author.id, original_post_info['message_id'], repost_message.id, repost_channel.id)

async def get_video_codec(video_bytes: bytes) -> Optional[str]:
    try:
        with tempfile.NamedTemporaryFile(prefix="repvid_", suffix=".tmp", delete=False) as tmp_file:
            tmp_filepath = tmp_file.name
            tmp_file.write(video_bytes)

        command = f"ffprobe -v quiet -print_format json -show_streams \"{tmp_filepath}\""
        process = await asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            log.error(f"FFPROBE: Error running ffprobe: {stderr.decode()}")
            return None

        streams = json.loads(stdout).get("streams", [])
        for stream in streams:
            if stream.get("codec_type") == "video":
                return stream.get("codec_name")

        return None

    except Exception as e:
        log.error(f"FFPROBE: Error getting video codec: {e}", exc_info=True)
        return None
    finally:
        if tmp_filepath and os.path.exists(tmp_filepath):
            try:
                os.remove(tmp_filepath)
            except Exception as e_del:
                log.error(f"FFPROBE: Unexpected error deleting temp file {tmp_filepath}: {e_del}")

async def process_media(bot_instance: commands.Bot, message: nextcord.Message, media_url: str, source_description: str, media_type: str):
    task_id = f"MsgID {message.id} ({media_type} via {source_description})"; log.info(f">>> PROCESS_MEDIA [{task_id}]: START")
    if not message.guild: return
    try: message = await message.channel.fetch_message(message.id)
    except Exception: log.warning(f"PROCESS_MEDIA [{task_id}]: Msg deleted/inaccessible."); return
    if not message or not message.guild: return

    media_bytes = await download_media(bot_instance, media_url)
    if not media_bytes: log.warning(f"PROCESS_MEDIA [{task_id}]: Download failed."); return

    if media_type == 'video':
        codec = await get_video_codec(media_bytes)
        if codec == "av1":
            log.info(f"PROCESS_MEDIA [{task_id}]: Skipping AV1 video.")
            return

    current_hashes: Optional[Union[imagehash.ImageHash, List[imagehash.ImageHash]]] = None
    if media_type == 'image': current_hashes = await get_image_phash(media_bytes)
    elif media_type == 'video': current_hashes = await get_video_multi_frame_phashes(media_bytes)
    if not current_hashes: log.warning(f"PROCESS_MEDIA [{task_id}]: Hashing failed."); return
    try:
        # Get effective similarity threshold for this guild
        guild_id = message.guild.id
        effective_threshold = database.get_guild_similarity_threshold(guild_id)
        if effective_threshold is None:
            effective_threshold = GLOBAL_SIMILARITY_THRESHOLD # Fallback to global
        log.debug(f"PROCESS_MEDIA [{task_id}]: Using effective threshold {effective_threshold} for Guild {guild_id}")

        existing_post = database.find_similar_hash(guild_id, current_hashes, effective_threshold)
        if existing_post and existing_post["message_id"] != message.id:
            log.info(f"!!! PROCESS_MEDIA [{task_id}]: REPOST DETECTED !!! Similar to {existing_post['message_id']}");
            await handle_repost(bot_instance, message, existing_post)
        elif not existing_post:
            log.info(f"PROCESS_MEDIA [{task_id}]: No match found. Adding hash(es).");
            database.add_hash(guild_id, message.channel.id, message.id, message.author.id, current_hashes, media_url)
    except Exception as e: log.error(f"PROCESS_MEDIA [{task_id}]: Error during DB/Action: {e}", exc_info=True)
    log.info(f"<<< PROCESS_MEDIA [{task_id}]: END")

@bot.event
async def on_message(message: nextcord.Message):
    if message.author.bot or not message.guild: return
    if message.content.startswith(bot.command_prefix): await bot.process_commands(message); return
    if database.is_channel_whitelisted(message.guild.id, message.channel.id): return
    if not message.attachments and not message.embeds and not re.search(r'(https?://[\S]+)', message.content):
        return
    if message.type not in (nextcord.MessageType.default, nextcord.MessageType.reply): return
    if not isinstance(message.channel, (nextcord.TextChannel, nextcord.Thread, nextcord.VoiceChannel)): return
    try:
        perms = message.channel.permissions_for(message.guild.me)
        if not perms.read_message_history or not perms.send_messages or not perms.manage_messages: return
    except Exception: return

    tasks_to_create = []; processed_urls = set()
    for attachment in message.attachments:
        if attachment.url in processed_urls or not (0 < attachment.size <= MAX_FILE_SIZE_BYTES): continue
        file_ext = os.path.splitext(attachment.filename)[1].lower(); media_type = None
        if file_ext in SUPPORTED_IMAGE_EXTENSIONS: media_type = 'image'
        elif file_ext in SUPPORTED_VIDEO_EXTENSIONS and OPENCV_AVAILABLE: media_type = 'video'
        if media_type: tasks_to_create.append(process_media(bot, message, attachment.url, f"att", media_type)); processed_urls.add(attachment.url)
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
            elif not file_ext: final_media_type = potential_media_type
        except Exception as url_parse_err: log.warning(f"ON_MESSAGE [{message.id}]: Failed parse embed URL '{media_url}': {url_parse_err}"); continue
        if final_media_type == 'video' and not OPENCV_AVAILABLE: continue
        if final_media_type: tasks_to_create.append(process_media(bot, message, media_url, f"{embed.type or 'unk'} emb", final_media_type)); processed_urls.add(media_url)

    urls = re.findall(r'(https?://[\S]+)', message.content)
    for url in urls:
        if url in processed_urls: continue
        try:
            file_ext = os.path.splitext(url.split('?')[0])[1].lower()
            media_type = None
            if file_ext in SUPPORTED_IMAGE_EXTENSIONS: media_type = 'image'
            elif file_ext in SUPPORTED_VIDEO_EXTENSIONS and OPENCV_AVAILABLE: media_type = 'video'
            if media_type: tasks_to_create.append(process_media(bot, message, url, f"link", media_type)); processed_urls.add(url)
        except Exception as url_parse_err: log.warning(f"ON_MESSAGE [{message.id}]: Failed parse URL '{url}': {url_parse_err}"); continue

    if tasks_to_create:
        log.info(f"ON_MESSAGE [{message.id}]: Running {len(tasks_to_create)} media tasks...")
        results = await asyncio.gather(*tasks_to_create, return_exceptions=True)
        for idx, res in enumerate(results):
             if isinstance(res, Exception): log.error(f"ON_MESSAGE [{message.id}]: Task {idx+1} failed: {res}", exc_info=False)

@bot.command(name="setalertchannel", help="REQUIRED setup: Run in the channel for alerts. Needs 'Manage Server'.", aliases=['setalerts'])
@commands.guild_only()
@commands.has_permissions(manage_guild=True)
async def set_alert_channel_command(ctx: commands.Context):
    if not isinstance(ctx.channel, nextcord.TextChannel): await ctx.reply("⚠️ Must be run in a text channel.", delete_after=30); return
    try:
        perms = ctx.channel.permissions_for(ctx.guild.me)
        missing_perms = []
        if not perms.send_messages: missing_perms.append("Send Messages")
        if not perms.read_message_history: missing_perms.append("Read Message History")
        if not perms.embed_links: missing_perms.append("Embed Links")
        if missing_perms: await ctx.reply(f"⚠️ I need permissions in {ctx.channel.mention}:\n- `{'`, `'.join(missing_perms)}`"); return
    except Exception as e: await ctx.reply("❌ Error checking permissions."); log.error(f"CMD_SETALERT Perm check failed: {e}"); return
    if database.set_alert_channel(ctx.guild.id, ctx.channel.id): await ctx.reply(f"✅ Repost alerts active in {ctx.channel.mention}.")
    else: await ctx.reply("❌ Error saving channel setting.")

@set_alert_channel_command.error
async def set_alert_channel_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.MissingPermissions): await ctx.reply("🚫 'Manage Server' permission required.")
    elif isinstance(error, commands.NoPrivateMessage): await ctx.reply("🚫 Use in a server text channel.")
    else: log.error(f"CMD_SETALERT Error: {error}", exc_info=True); await ctx.reply("❌ Unexpected error.")

@bot.command(name="whitelist", help="Ignore reposts in channel. Usage: !whitelist [#ch/ID]. Needs 'Manage Server'.", aliases=['wl'])
@commands.guild_only()
@commands.has_permissions(manage_guild=True)
async def whitelist_command(ctx: commands.Context, channel: Optional[nextcord.TextChannel] = None):
    target = channel or ctx.channel
    if not isinstance(target, nextcord.TextChannel): await ctx.reply("⚠️ Please specify a valid text channel.", delete_after=30); return
    added = database.add_whitelist_channel(ctx.guild.id, target.id)
    if added: await ctx.reply(f"✅ Repost checks ignored in {target.mention}.")
    elif database.is_channel_whitelisted(ctx.guild.id, target.id): await ctx.reply(f"ℹ️ {target.mention} already whitelisted.")
    else: await ctx.reply(f"❌ Error whitelisting {target.mention}.")

@whitelist_command.error
async def whitelist_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.MissingPermissions): await ctx.reply("🚫 'Manage Server' permission required.")
    elif isinstance(error, (commands.ChannelNotFound, commands.BadArgument)): await ctx.reply(f"❓ Couldn't find channel.")
    elif isinstance(error, commands.NoPrivateMessage): await ctx.reply("🚫 Use in a server.")
    else: log.error(f"CMD_WHITELIST Error: {error}", exc_info=True); await ctx.reply("❌ Unexpected error.")

@bot.command(name="unwhitelist", help="Re-enable checks in channel. Usage: !unwhitelist [#ch/ID]. Needs 'Manage Server'.", aliases=['unwl'])
@commands.guild_only()
@commands.has_permissions(manage_guild=True)
async def unwhitelist_command(ctx: commands.Context, channel: Optional[nextcord.TextChannel] = None):
    target = channel or ctx.channel
    if not isinstance(target, nextcord.TextChannel): await ctx.reply("⚠️ Please specify a valid text channel.", delete_after=30); return
    removed = database.remove_whitelist_channel(ctx.guild.id, target.id)
    if removed: await ctx.reply(f"✅ Repost checks re-enabled in {target.mention}.")
    elif not database.is_channel_whitelisted(ctx.guild.id, target.id): await ctx.reply(f"ℹ️ {target.mention} was not whitelisted.")
    else: await ctx.reply(f"❌ Error unwhitelisting {target.mention}.")

@unwhitelist_command.error
async def unwhitelist_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.MissingPermissions): await ctx.reply("🚫 'Manage Server' permission required.")
    elif isinstance(error, (commands.ChannelNotFound, commands.BadArgument)): await ctx.reply(f"❓ Couldn't find channel.")
    elif isinstance(error, commands.NoPrivateMessage): await ctx.reply("🚫 Use in a server.")
    else: log.error(f"CMD_UNWHITELIST Error: {error}", exc_info=True); await ctx.reply("❌ Unexpected error.")


# --- New Similarity Threshold Commands ---
@bot.command(name="setthreshold", help="Set repost similarity threshold (0-15, lower=stricter). Needs 'Manage Server'. Usage: !setthreshold <value>")
@commands.guild_only()
@commands.has_permissions(manage_guild=True)
async def set_threshold_command(ctx: commands.Context, threshold_value: int):
    if not 0 <= threshold_value <= 15: # Sensible range for pHash difference
        await ctx.reply("⚠️ Threshold must be between 0 and 15 (higher is stricter)."); return
    if database.set_guild_similarity_threshold(ctx.guild.id, threshold_value):
        await ctx.reply(f"✅ Server similarity threshold set to `{threshold_value}`.")
    else: await ctx.reply("❌ Error saving threshold setting.")

@set_threshold_command.error
async def set_threshold_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.MissingRequiredArgument): await ctx.reply(f"❓ Usage: `{bot.command_prefix}setthreshold <value>` (e.g., 5)")
    elif isinstance(error, commands.BadArgument): await ctx.reply("❓ Threshold must be a whole number (0-15).")
    elif isinstance(error, commands.MissingPermissions): await ctx.reply("🚫 'Manage Server' permission required.")
    elif isinstance(error, commands.NoPrivateMessage): await ctx.reply("🚫 Use in a server.")
    else: log.error(f"CMD_SETTHRESHOLD Error: {error}", exc_info=True); await ctx.reply("❌ Unexpected error.")

@bot.command(name="getthreshold", help="Show current repost similarity threshold for this server.", aliases=["showthreshold"])
@commands.guild_only()
async def get_threshold_command(ctx: commands.Context):
    guild_threshold = database.get_guild_similarity_threshold(ctx.guild.id)
    if guild_threshold is not None:
        await ctx.reply(f"ℹ️ Current similarity threshold for this server is `{guild_threshold}` (higher is stricter).")
    else:
        await ctx.reply(f"ℹ️ This server is using the global default similarity threshold: `{GLOBAL_SIMILARITY_THRESHOLD}` (higher is stricter).")

@bot.command(name="stats", help="Show statistics about the bot's activity.")
@commands.guild_only()
async def stats_command(ctx: commands.Context):
    stats = database.get_stats(ctx.guild.id)
    if not stats:
        await ctx.reply("❌ Could not retrieve statistics."); return

    embed = nextcord.Embed(title=f"🤖 Repost Bot Statistics for {ctx.guild.name}", color=nextcord.Color.blue())
    embed.add_field(name="Total Media Processed", value=f"`{stats.get('total_media_processed', 0)}`", inline=True)
    embed.add_field(name="Reposts Detected", value=f"`{stats.get('total_reposts_detected', 0)}`", inline=True)

    repost_rate = 0
    if stats.get('total_media_processed', 0) > 0:
        repost_rate = (stats.get('total_reposts_detected', 0) / stats.get('total_media_processed', 0)) * 100
    embed.add_field(name="Repost Rate", value=f"`{repost_rate:.2f}%`", inline=True)

    top_reposters = stats.get("top_reposters")
    if top_reposters:
        value = ""
        for i, (reposter_id, count) in enumerate(top_reposters):
            user = await bot.fetch_user(reposter_id)
            value += f"{i+1}. {user.mention}: {count} reposts\n"
        embed.add_field(name="🏆 Top Reposters", value=value, inline=False)

    top_channels = stats.get("top_reposted_channels")
    if top_channels:
        value = ""
        for i, (channel_id, count) in enumerate(top_channels):
            channel = bot.get_channel(channel_id)
            value += f"{i+1}. {channel.mention}: {count} reposts\n"
        embed.add_field(name="📢 Top Reposted Channels", value=value, inline=False)

    guild_threshold = database.get_guild_similarity_threshold(ctx.guild.id)
    embed.set_footer(text=f"Bot is using a similarity threshold of `{guild_threshold or GLOBAL_SIMILARITY_THRESHOLD}`.")

    await ctx.reply(embed=embed)

async def update_stats_for_guild(guild_id: int):
    stats_channel_info = database.get_stats_channel(guild_id)
    if not stats_channel_info: return

    channel_id = stats_channel_info.get("channel_id")
    message_id = stats_channel_info.get("message_id")
    if not channel_id or not message_id: return

    try:
        channel = await bot.fetch_channel(channel_id)
        message = await channel.fetch_message(message_id)
    except (nextcord.NotFound, nextcord.Forbidden):
        log.warning(f"STATS_UPDATE: Could not find channel or message for guild {guild_id}. Clearing from DB.")
        database.set_stats_channel(guild_id, None, None)
        return
    except Exception as e:
        log.error(f"STATS_UPDATE: Error fetching channel/message for guild {guild_id}: {e}")
        return

    stats = database.get_stats(guild_id)
    if not stats:
        log.warning(f"STATS_UPDATE: Could not retrieve stats for guild {guild_id}.")
        return

    guild = bot.get_guild(guild_id)
    embed = nextcord.Embed(title=f"📊 Live Statistics for {guild.name}", color=nextcord.Color.green())
    embed.add_field(name="Total Media Processed", value=f"`{stats.get('total_media_processed', 0)}`", inline=True)
    embed.add_field(name="Reposts Detected", value=f"`{stats.get('total_reposts_detected', 0)}`", inline=True)

    repost_rate = 0
    if stats.get('total_media_processed', 0) > 0:
        repost_rate = (stats.get('total_reposts_detected', 0) / stats.get('total_media_processed', 0)) * 100
    embed.add_field(name="Repost Rate", value=f"`{repost_rate:.2f}%`", inline=True)

    top_reposters = stats.get("top_reposters")
    if top_reposters:
        value = ""
        for i, (reposter_id, count) in enumerate(top_reposters):
            try:
                user = await bot.fetch_user(reposter_id)
                value += f"{i+1}. {user.mention}: {count} reposts\n"
            except nextcord.NotFound:
                value += f"{i+1}. User ID {reposter_id}: {count} reposts\n"
        embed.add_field(name="🏆 Top Reposters", value=value, inline=False)

    top_channels = stats.get("top_reposted_channels")
    if top_channels:
        value = ""
        for i, (channel_id, count) in enumerate(top_channels):
            channel = bot.get_channel(channel_id)
            if channel:
                value += f"{i+1}. {channel.mention}: {count} reposts\n"
            else:
                value += f"{i+1}. Channel ID {channel_id}: {count} reposts\n"
        embed.add_field(name="📢 Top Reposted Channels", value=value, inline=False)

    guild_threshold = database.get_guild_similarity_threshold(guild.id)
    embed.set_footer(text=f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')} UTC | Threshold: {guild_threshold or GLOBAL_SIMILARITY_THRESHOLD}")

    try:
        await message.edit(embed=embed)
        log.info(f"STATS_UPDATE: Updated stats for guild {guild.id}")
    except Exception as e:
        log.error(f"STATS_UPDATE: Error updating stats for guild {guild.id}: {e}")

@tasks.loop(minutes=15)
async def update_stats_task():
    for guild in bot.guilds:
        stats_channel_info = database.get_stats_channel(guild.id)
        if not stats_channel_info: continue

        channel_id = stats_channel_info.get("channel_id")
        message_id = stats_channel_info.get("message_id")
        if not channel_id or not message_id: continue

        try:
            channel = await bot.fetch_channel(channel_id)
            message = await channel.fetch_message(message_id)
        except (nextcord.NotFound, nextcord.Forbidden):
            log.warning(f"STATS_TASK: Could not find channel or message for guild {guild.id}. Clearing from DB.")
            database.set_stats_channel(guild.id, None, None)
            continue
        except Exception as e:
            log.error(f"STATS_TASK: Error fetching channel/message for guild {guild.id}: {e}")
            continue

        stats = database.get_stats(guild.id)
        if not stats:
            log.warning(f"STATS_TASK: Could not retrieve stats for guild {guild.id}.")
            continue

        embed = nextcord.Embed(title=f"📊 Live Statistics for {guild.name}", color=nextcord.Color.green())
        embed.add_field(name="Total Media Processed", value=f"`{stats.get('total_media_processed', 0)}`", inline=True)
        embed.add_field(name="Reposts Detected", value=f"`{stats.get('total_reposts_detected', 0)}`", inline=True)

        repost_rate = 0
        if stats.get('total_media_processed', 0) > 0:
            repost_rate = (stats.get('total_reposts_detected', 0) / stats.get('total_media_processed', 0)) * 100
        embed.add_field(name="Repost Rate", value=f"`{repost_rate:.2f}%`", inline=True)

        top_reposters = stats.get("top_reposters")
        if top_reposters:
            value = ""
            for i, (reposter_id, count) in enumerate(top_reposters):
                try:
                    user = await bot.fetch_user(reposter_id)
                    value += f"{i+1}. {user.mention}: {count} reposts\n"
                except nextcord.NotFound:
                    value += f"{i+1}. User ID {reposter_id}: {count} reposts\n"
            embed.add_field(name="🏆 Top Reposters", value=value, inline=False)

        top_channels = stats.get("top_reposted_channels")
        if top_channels:
            value = ""
            for i, (channel_id, count) in enumerate(top_channels):
                channel = bot.get_channel(channel_id)
                if channel:
                    value += f"{i+1}. {channel.mention}: {count} reposts\n"
                else:
                    value += f"{i+1}. Channel ID {channel_id}: {count} reposts\n"
            embed.add_field(name="📢 Top Reposted Channels", value=value, inline=False)

        guild_threshold = database.get_guild_similarity_threshold(guild.id)
        embed.set_footer(text=f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')} UTC | Threshold: {guild_threshold or GLOBAL_SIMILARITY_THRESHOLD}")

        try:
            await message.edit(embed=embed)
            log.info(f"STATS_TASK: Updated stats for guild {guild.id}")
        except Exception as e:
            log.error(f"STATS_TASK: Error updating stats for guild {guild.id}: {e}")

@update_stats_task.before_loop
async def before_update_stats_task():
    await bot.wait_until_ready()

@bot.command(name="setstatschannel", help="Set a channel for auto-updating stats. Needs 'Manage Server'.")
@commands.guild_only()
@commands.has_permissions(manage_guild=True)
async def set_stats_channel_command(ctx: commands.Context, channel: Optional[nextcord.TextChannel] = None):
    target_channel = channel or ctx.channel
    if not isinstance(target_channel, nextcord.TextChannel):
        await ctx.reply("⚠️ Please specify a valid text channel.", delete_after=30); return

    try:
        perms = target_channel.permissions_for(ctx.guild.me)
        if not perms.send_messages or not perms.embed_links:
            await ctx.reply(f"⚠️ I need `Send Messages` and `Embed Links` permissions in {target_channel.mention}."); return
    except Exception as e:
        await ctx.reply("❌ Error checking permissions."); log.error(f"CMD_SETSTATSCHANNEL Perm check failed: {e}"); return

    try:
        stats_embed = nextcord.Embed(title="📊 Live Statistics", description="Initializing...", color=nextcord.Color.green())
        stats_message = await target_channel.send(embed=stats_embed)
    except Exception as e:
        await ctx.reply(f"❌ Could not send initial stats message in {target_channel.mention}."); log.error(f"CMD_SETSTATSCHANNEL Send failed: {e}"); return

    if database.set_stats_channel(ctx.guild.id, target_channel.id, stats_message.id):
        await ctx.reply(f"✅ Stats channel set to {target_channel.mention}. The stats message will update automatically.")
        await update_stats_for_guild(ctx.guild.id)
    else:
        await ctx.reply("❌ Error saving stats channel setting.")


async def shutdown_signal_handler(bot_instance: commands.Bot, signal_type: signal.Signals):
    if not isinstance(bot_instance, RepostBotState) or bot_instance._shutdown_event.is_set(): return
    log.warning(f"SIGNAL: Received {signal_type.name}. Initiating shutdown...");
    if not bot_instance.is_closed(): asyncio.create_task(bot_instance.close(), name="SignalShutdownTask")

if __name__ == "__main__":
    if not BOT_TOKEN: log.critical("BOT: FATAL - DISCORD_BOT_TOKEN not set!"); sys.exit(1)
    try: log_config()
    except Exception as e: log.error(f"Failed log config: {e}")
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
        if isinstance(bot, RepostBotState) and not bot.is_closed() and not bot._shutdown_event.is_set():
             log.warning("BOT: Forcing final close...")
             try:
                 if loop.is_running(): loop.run_until_complete(asyncio.wait_for(bot.close(), timeout=10.0))
                 else: asyncio.run(bot.close())
             except Exception as fe: log.error(f"Error during final forced close: {fe}", exc_info=True)
        if not loop.is_closed():
            tasks = [t for t in asyncio.all_tasks(loop=loop) if not t.done()];
            if tasks: log.info(f"BOT: Cancelling {len(tasks)} tasks..."); [t.cancel() for t in tasks];
            try:
                if tasks: loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            except RuntimeError: pass
            except Exception: pass
            if loop.is_running(): loop.stop()
            loop.close(); log.info("BOT: Loop closed.")
        log.info(f"--- Repost Detector Bot ({USER_AGENT}) process finished ---")