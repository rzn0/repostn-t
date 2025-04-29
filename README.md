# Discord Repost Detector Bot

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python-based Discord bot designed to detect and manage reposted media (images, GIFs, videos) within your server channels, redirecting alerts to a designated channel after a one-time setup.

## Features

*   **Media Monitoring:** Listens to messages containing media across server channels where it has permission.
*   **Repost Detection:** Detects reposts of:
    *   Images (PNG, JPG, JPEG, WEBP, BMP, GIF)
    *   Video Files (analyzes first non-black, middle, and last frames - requires OpenCV)
*   **Perceptual Hashing:** Uses image hashing (specifically pHash via `ImageHash` and Pillow) to identify visually similar content, making it resilient to minor edits, format changes, and compression differences.
*   **Mandatory Server Setup:** Requires an administrator to designate a specific channel for repost alerts using a simple command (`!setalertchannel` by default) before the bot takes action on reposts.
*   **Dedicated Alert Channel:** Once configured, all repost alerts ("shame messages") are sent to the designated channel, keeping other channels cleaner.
*   **Duplicate Handling (Post-Setup):**
    *   Sends an alert message to the configured alert channel, mentioning the reposter and linking to the original post.
    *   Deletes the repost message from its original channel (requires `Manage Messages` permission there).
*   **Database:** Stores media hashes (using SQLite) to remember previously posted content within a server (`media_hashes` table) and the configured alert channel (`guild_config` table).
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
    This isolates dependencies for this project.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```

3.  **Install Dependencies:**
    *(Create a `requirements.txt` file in the bot's root directory with the following content first):*
    ```txt
    # requirements.txt
    nextcord==3.1.0
    numpy
    opencv-python
    Pillow
    ImageHash
    python-dotenv
    aiohttp
    ```
    *Then run:*
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    *(Watch this step carefully for any errors, especially during `numpy` or `opencv-python` installation. Install system prerequisites mentioned in Requirements if needed.)*

4.  **Configure Environment Variables:**
    *   Create a new `.env` file in the root directory of the bot.
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

        # Bot command prefix (Used for the setup command)
        BOT_PREFIX="!"

        # Video Processing - Black Frame Detection Threshold (Lower = stricter black detection)
        BLACK_FRAME_THRESHOLD=10
        ```
    *   **Replace `YOUR_BOT_TOKEN_GOES_HERE` with your actual bot token.**
    *   Adjust other settings as needed (the defaults are generally reasonable).
    *   **IMPORTANT:** Ensure the `.env` file is listed in your `.gitignore` file and **never commit it** to version control.

5.  **Discord Application Setup:**
    *   Go to the [Discord Developer Portal](https://discord.com/developers/applications).
    *   Create a **New Application**.
    *   Go to the **Bot** tab on the left sidebar.
    *   Click **Add Bot** and confirm.
    *   **Token:** Under the bot's username, click **Reset Token** (or **View Token** if visible) to get your bot token. Copy this token and paste it into your `.env` file for `DISCORD_BOT_TOKEN`. **Keep this token secret!**
    *   **Privileged Gateway Intents:** Scroll down and **ENABLE** the **`MESSAGE CONTENT INTENT`**. This is **mandatory** for the bot to read commands and potentially message content in the future. Click **Save Changes**.

6.  **Invite Bot to Your Server:**
    *   Go back to your Application in the Developer Portal.
    *   Go to **OAuth2 -> URL Generator**.
    *   Under **Scopes**, select `bot`.
    *   Under **Bot Permissions**, select the following (crucial for operation):
        *   `View Channels` (Implies Read Messages)
        *   `Send Messages` (Needed for alerts and setup messages)
        *   `Manage Messages` (Needed to delete reposts)
        *   `Read Message History` (Needed for context and finding original messages)
        *   *(Recommended)* `Embed Links` (For potentially nicer alert formatting later)
    *   Copy the **Generated URL** at the bottom.
    *   Paste the URL into your web browser and select the server you want to add the bot to. Authorize the permissions.

7.  **Initial Bot Run & Setup (Mandatory):**
    *   Run the bot for the first time (see next section). It will automatically create the database file (`db/repost_hashes.db`) and tables if they don't exist (check console logs for success or errors).
    *   **After the bot is online:** An administrator with **"Manage Server"** permission must go to the specific text channel where they want repost alerts to be sent.
    *   In that channel, run the setup command (default: `!setalertchannel`).
    *   The bot **must** have `Send Messages` and `Read Message History` permissions in that chosen alert channel. It will check this when you run the command.
    *   The bot will confirm if the channel was set successfully. Repost detection and handling will now be active for the server.
    *   If a repost is detected *before* setup is complete, the bot will post a reminder message in the channel where the repost occurred, prompting an admin to run the setup command.

## Running the Bot

1.  **Activate Virtual Environment** (if you created one):
    ```bash
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate   # Windows
    ```
2.  **Run the Python script:**
    ```bash
    python repostBot.py
    ```
3.  The bot should log into Discord and print readiness messages to the console. Check the console for errors if it doesn't start correctly (especially database errors on the first run).
4.  **Remember to perform the mandatory `!setalertchannel` setup step** (see step 7 above) after the bot is running.

## Running with PM2 (Optional)

If you're using PM2 on your server/Raspberry Pi:

1.  **Make sure you are in the bot's directory:** `cd /path/to/your/discord-repost-bot`
2.  **Start the bot:**
    *   **If NOT using a venv:**
        ```bash
        pm2 start repostBot.py --name repostBot --interpreter python3
        ```
    *   **If using a venv:** Use the full path to the Python interpreter inside the venv.
        ```bash
        pm2 start venv/bin/python --name repostBot -- repostBot.py
        # Note the '--' separating pm2 options from the script and its arguments
        ```
3.  **Monitor Logs:**
    ```bash
    pm2 logs repostBot
    ```
4.  **Other PM2 Commands:**
    *   `pm2 stop repostBot`
    *   `pm2 restart repostBot`
    *   `pm2 delete repostBot`
    *   `pm2 save` (To save the process list for restarts)
    *   `pm2 startup` (To make PM2 automatically start bots on system reboot)

## How it Works (Briefly)

1.  **Event Listener:** The bot listens for new messages (`on_message`). If a message starts with the command prefix, it processes it as a command (`!setalertchannel`).
2.  **Media Check:** If not a command, it checks if the message contains supported image, GIF, or video attachments/embeds.
3.  **Download:** It downloads the media content into memory.
4.  **Hashing:**
    *   **Images/GIFs:** Calculates a perceptual hash (pHash) using `ImageHash` and `Pillow`.
    *   **Videos:** Extracts the first non-black, middle, and last frames using `OpenCV`, converts them, and calculates their pHashes.
5.  **Database Query:** It queries the SQLite database for existing hashes within the same server (guild) that are visually similar (Hamming distance <= `SIMILARITY_THRESHOLD`), ordered by timestamp.
6.  **Action:**
    *   **If a similar hash is found:** It calls the `handle_repost` function.
    *   **Inside `handle_repost`:**
        *   It first checks the database (`guild_config` table) to see if an alert channel has been configured for this server.
        *   **If NOT configured:** It sends a temporary "Setup Required" message to the channel where the repost occurred and takes no further action.
        *   **If configured:** It attempts to fetch the configured alert channel, validates permissions, sends the alert message there (mentioning the user, linking the original), and then deletes the repost message from its original channel (requires `Manage Messages` permission there).
    *   **If no similar hash is found:** It adds the new hash(es), message details, and timestamp to the database (`media_hashes` table).

## Limitations

*   **Perceptual Hashing Isn't Perfect:** While good, pHash can occasionally have:
    *   **False Positives:** Flagging two different but visually simple/similar images as reposts.
    *   **False Negatives:** Missing reposts if the image/video is heavily edited (cropped significantly, mirrored, large overlays, heavy filters). The `SIMILARITY_THRESHOLD` needs tuning.
*   **Video Processing:** Analyzes only keyframes (first non-black, middle, last). Won't reliably detect reposts of heavily trimmed videos, sped-up/slowed-down videos, or videos with significant filters/overlays applied differently across frames. It's not full video fingerprinting.
*   **Resource Usage:** Processing images and especially videos requires CPU and RAM. OpenCV and hashing multiple frames can be resource-intensive on low-power devices or high-traffic servers.
*   **SQLite Scalability:** SQLite is simple but may become slow on extremely large servers with millions of stored hashes.
*   **Ephemeral Media:** Doesn't handle media that disappears quickly (e.g., certain image hosts or edited messages where the original attachment is removed before processing).
*   **Setup Requirement:** Bot functionality is paused until the mandatory setup command is run.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs or feature suggestions.
