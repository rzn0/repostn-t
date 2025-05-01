# Discord Repost Detector Bot

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python-based Discord bot designed to detect and manage reposted media (images, GIFs, videos) within your server channels. It requires a one-time setup to designate an alert channel and allows whitelisting specific channels where reposts are permitted.

## Features

*   **Media Monitoring:** Listens to messages containing media across server channels where it has permission and is not whitelisted.
*   **Repost Detection:** Detects reposts of:
    *   Images (PNG, JPG, JPEG, WEBP, BMP, GIF)
    *   Video Files (analyzes first non-black, middle, and last frames - requires OpenCV)
*   **Perceptual Hashing:** Uses image hashing (pHash via `ImageHash` and Pillow) to identify visually similar content, resilient to minor edits, format changes, and compression.
*   **Mandatory Server Setup:** Requires an administrator to designate a specific channel for repost alerts using the `!setalertchannel` command before the bot actively handles reposts.
*   **Dedicated Alert Channel:** Once configured, all repost alerts are sent to the designated channel, keeping others cleaner.
*   **Channel Whitelisting:** Administrators can whitelist specific channels using the `!whitelist` command, preventing the bot from checking for reposts in those channels. Whitelisting is managed via the database.
*   **Duplicate Handling (Post-Setup & Non-Whitelisted):**
    *   Sends an alert message to the configured alert channel, mentioning the reposter and linking to the original post.
    *   Deletes the repost message from its original channel (requires `Manage Messages` permission there).
*   **Database:** Stores media hashes (`media_hashes`), alert channel configuration (`guild_config`), and channel whitelist status (`channel_whitelist`) using SQLite.
*   **Configurable:** Settings like hash sensitivity, file size limits, command prefix, and logging level can be adjusted via an environment file (`.env`).

## Requirements

*   **Python 3.8+**
*   **Pip** (Python package installer)
*   **Git** (for cloning the repository)
*   **OpenCV Python Bindings:** Required for video frame analysis. Usually installed via pip (`opencv-python`), but may require system dependencies (see Step 3 below).
*   **FFmpeg:** Required by OpenCV for reliable video decoding. Install system-wide:
    *   **Debian/Ubuntu/Kali:** `sudo apt update && sudo apt install ffmpeg -y`
    *   **macOS (Homebrew):** `brew install ffmpeg`
    *   **Windows:** Download from the official FFmpeg website and add `ffmpeg.exe` to your system's PATH.
*   **(Potentially) Build Tools:** Installing `opencv-python` and `numpy` might require system build tools if pre-built wheels aren't available for your system/architecture (especially on Raspberry Pi/ARM):
    *   **Debian/Ubuntu/Kali:** `sudo apt install build-essential cmake pkg-config python3-dev`
    *   May also need image format libraries: `sudo apt install libjpeg-dev libpng-dev libtiff-dev libgtk-3-dev libatlas-base-dev gfortran` (Install as needed based on pip errors).

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd <repository_directory> # e.g., cd discord-repost-bot
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    *(Watch this step carefully for any errors, especially during `numpy` or `opencv-python` installation. Install system prerequisites mentioned in Requirements if needed.)*

4.  **Configure Environment Variables:**
    *   Copy the example environment file (if provided, e.g., `.env.example`) or create a new `.env` file in the root directory.
    *   **Edit the `.env` file:**
        ```dotenv
        # Discord Bot Token (REQUIRED) - Get from Discord Developer Portal
        DISCORD_BOT_TOKEN=YOUR_BOT_TOKEN_GOES_HERE

        # --- Optional Configuration ---
        # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        LOG_LEVEL=INFO

        # Perceptual Hash Settings (Lower threshold = stricter)
        HASH_SIZE=8
        SIMILARITY_THRESHOLD=5

        # Max file size (MB) to process
        MAX_FILE_SIZE_MB=30

        # Bot command prefix (Used for setup/whitelist commands)
        BOT_PREFIX="!"

        # Video Processing - Black Frame Detection Threshold (Lower = stricter black detection)
        BLACK_FRAME_THRESHOLD=10
        ```
    *   **Replace `YOUR_BOT_TOKEN_GOES_HERE` with your actual bot token.**
    *   Adjust other settings as needed.
    *   **IMPORTANT:** Ensure the `.env` file is listed in your `.gitignore` file and **never commit it** to version control.

5.  **Discord Application Setup:**
    *   Go to the [Discord Developer Portal](https://discord.com/developers/applications).
    *   Create a **New Application**.
    *   Go to the **Bot** tab and click **Add Bot**.
    *   **Token:** Get your bot token (Reset/View Token) and put it in the `.env` file. **Keep this token secret!**
    *   **Privileged Gateway Intents:** Scroll down and **ENABLE** the **`MESSAGE CONTENT INTENT`**. Click **Save Changes**.

6.  **Invite Bot to Your Server:**
    *   Go to **OAuth2 -> URL Generator**.
    *   **Scopes:** Select `bot`.
    *   **Bot Permissions:** Select:
        *   `View Channels` (Read Messages)
        *   `Send Messages`
        *   `Manage Messages`
        *   `Read Message History`
        *   *(Recommended)* `Embed Links`
    *   Copy the **Generated URL** and use it to add the bot to your server.

7.  **Initial Bot Run & Setup (Mandatory):**
    *   Run the bot (see next section). It automatically creates/updates the database (`db/repost_hashes.db`) on startup. Check logs for success/errors.
    *   **After the bot is online:** An administrator with **"Manage Server"** permission must go to the **text channel** where they want repost alerts sent.
    *   In that channel, run the setup command (default: `!setalertchannel`).
    *   The bot needs `Send Messages` and `Read Message History` in that channel. It will confirm success. Repost handling is now active.
    *   Optionally, use the `!whitelist` command in channels where you want to disable repost checks entirely.

## Running the Bot

1.  **Activate Virtual Environment** (if used):
    ```bash
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate   # Windows
    ```
2.  **Run the Python script:**
    ```bash
    python repostBot.py
    ```
3.  Check console logs for readiness messages and errors.
4.  **Remember the mandatory `!setalertchannel` step** if this is the first time running the bot in a server.

## Commands

*(Default prefix is `!`. Requires "Manage Server" permission.)*

*   `setalertchannel` (Alias: `setalerts`)
    *   **Usage:** Run this command **in the text channel** where you want repost alerts delivered.
    *   **Action:** Configures the bot to send all future repost alerts for this server to the channel where the command was run. This is **required** to activate the bot's alert/delete functionality.
*   `whitelist [channel]` (Alias: `wl`)
    *   **Usage:** Run in a channel (`!whitelist`) or specify one (`!whitelist #channel-name` or `!whitelist <channel_id>`).
    *   **Action:** Adds the target channel to the whitelist. The bot will ignore all messages in this channel for repost checks.
*   `unwhitelist [channel]` (Alias: `unwl`)
    *   **Usage:** Run in a channel (`!unwhitelist`) or specify one (`!unwhitelist #channel-name` or `!unwhitelist <channel_id>`).
    *   **Action:** Removes the target channel from the whitelist, re-enabling repost checks there.

## How it Works (Briefly)

1.  **Event Listener:** Listens for `on_message`.
2.  **Command Check:** If the message starts with the prefix, attempts to process it as a command (`setalertchannel`, `whitelist`, `unwhitelist`).
3.  **Whitelist Check:** If not a command, checks if the message's channel is in the database whitelist for the server. If yes, ignores the message.
4.  **Media Check:** If not whitelisted, checks for supported media attachments/embeds.
5.  **Download & Hash:** Downloads valid media. Calculates pHash for images/GIFs. Extracts and hashes first non-black, middle, and last frames for videos (using OpenCV).
6.  **Database Query:** Queries the SQLite database (`media_hashes` table) for visually similar hashes (Hamming distance <= `SIMILARITY_THRESHOLD`) within the same server.
7.  **Action:**
    *   **If a similar hash is found:** Calls `handle_repost`.
    *   **Inside `handle_repost`:**
        *   Checks the `guild_config` table for the configured alert channel ID.
        *   **If NOT configured:** Sends a "Setup Required" message to the repost channel and stops.
        *   **If configured:** Sends the alert to the configured channel and deletes the repost from its original channel.
    *   **If no similar hash is found:** Adds the new hash(es) to the `media_hashes` table.

## Limitations

*   **Hashing Imperfections:** pHash isn't foolproof against heavy edits, crops, mirrors, or overlays. `SIMILARITY_THRESHOLD` requires tuning.
*   **Video Analysis:** Analyzes keyframes only. Doesn't detect heavily trimmed/altered videos reliably.
*   **Resource Usage:** Media processing, especially multi-frame video hashing with OpenCV, can be CPU/RAM intensive.
*   **SQLite Scalability:** May slow down with millions of hashes.
*   **Setup Requirement:** Core functionality paused until `!setalertchannel` is used.
*   **Whitelist Management:** Whitelist is managed per-channel via commands and stored in the database.

## Contributing

Contributions welcome! Please submit pull requests or open issues.
