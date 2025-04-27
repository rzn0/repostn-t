# Discord Repost Detector Bot

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python-based Discord bot designed to detect and manage reposted media (images, GIFs, video first frames) within your server channels.

## Features

*   **Media Monitoring:** Listens to messages in configured channels.
*   **Repost Detection:** Detects reposts of:
    *   Images (PNG, JPG, WEBP, BMP)
    *   GIFs
    *   Video Files (first frame analysis - requires OpenCV & FFmpeg)
*   **Perceptual Hashing:** Uses image hashing (specifically pHash via `ImageHash`) to identify visually similar content, making it resilient to minor edits, format changes, and compression differences.
*   **Duplicate Handling:**
    *   Replies to the repost message, linking to the original post.
    *   Optionally deletes the repost message.
    *   Mentions ("shames") the user who reposted.
*   **Database:** Stores media hashes (using SQLite) to remember previously posted content within a server.
*   **Configurable:** Settings like hash sensitivity, file size limits, and logging level can be adjusted via an environment file.

## Requirements

*   **Python 3.8+**
*   **Pip** (Python package installer)
*   **Git** (for cloning the repository)
*   **FFmpeg:** Required by OpenCV for reliable video decoding. Install system-wide:
    *   **Debian/Ubuntu/Kali:** `sudo apt update && sudo apt install ffmpeg -y`
    *   **macOS (Homebrew):** `brew install ffmpeg`
    *   **Windows:** Download from the official FFmpeg website and add it to your system's PATH.
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
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    *(Watch this step carefully for any errors, especially during `numpy` or `opencv-python` installation. Install system prerequisites mentioned in Requirements if needed.)*

4.  **Configure Environment Variables:**
    *   Create a new .env file in the root directory of the bot.
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

        # Bot command prefix (if commands are added later)
        BOT_PREFIX="!"
        ```
    *   **Replace `YOUR_BOT_TOKEN_GOES_HERE` with your actual bot token.**
    *   Adjust other settings as needed.
    *   **IMPORTANT:** Ensure the `.env` file is listed in your `.gitignore` file and **never commit it** to version control.

5.  **Discord Application Setup:**
    *   Go to the [Discord Developer Portal](https://discord.com/developers/applications).
    *   Create a **New Application**.
    *   Go to the **Bot** tab on the left sidebar.
    *   Click **Add Bot** and confirm.
    *   **Token:** Under the bot's username, click **Reset Token** (or **View Token** if visible) to get your bot token. Copy this token and paste it into your `.env` file for `DISCORD_BOT_TOKEN`. **Keep this token secret!**
    *   **Privileged Gateway Intents:** Scroll down and **ENABLE** the **`MESSAGE CONTENT INTENT`**. This is **mandatory** for the bot to read message attachments and embeds. Click **Save Changes**.

6.  **Invite Bot to Your Server:**
    *   Go back to your Application in the Developer Portal.
    *   Go to **OAuth2 -> URL Generator**.
    *   Under **Scopes**, select `bot`.
    *   Under **Bot Permissions**, select the following:
        *   `View Channels` (Read Messages)
        *   `Send Messages`
        *   `Manage Messages` (To delete reposts)
        *   `Read Message History`
        *   *(Optional)* `Embed Links` (For cleaner output)
    *   Copy the **Generated URL** at the bottom.
    *   Paste the URL into your web browser and select the server you want to add the bot to. Authorize the permissions.

7.  **Database Setup (Manual - Recommended for Stability):**
    The bot attempts to create the database schema on startup, but this can fail due to permissions or other issues. It's recommended to create it manually once before the first run, especially if you encounter "no such table" errors.
    *   Ensure you are in the bot's main directory (`/home/admin/repostBot/`).
    *   Make sure the `db` directory exists and is writable by the user running the bot (see Requirements/Permissions).
    *   Run the database script directly:
        ```bash
        # Ensure your virtual environment is active if using one
        python database.py
        ```
    *   Check the output for `--- Database setup script finished successfully. ---`. If errors occur, address them (likely permissions).
    *   You can verify with `sqlite3 db/repost_hashes.db ".schema media_hashes"`.

## Running the Bot

1.  **Activate Virtual Environment** (if you created one):
    ```bash
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate   # Windows
    ```
2.  **Run the Python script:**
    ```bash
    python bot.py
    ```
3.  The bot should log into Discord and print readiness messages to the console. Check the console for errors if it doesn't start correctly.

## Running with PM2 (Optional)

If you're using PM2 on your server/Raspberry Pi:

1.  **Make sure you are in the bot's directory:** `cd /path/to/your/repost-bot`
2.  **Start the bot:**
    *   **If NOT using a venv:**
        ```bash
        pm2 start bot.py --name repostBot --interpreter python3
        ```
    *   **If using a venv:** Use the full path to the Python interpreter inside the venv.
        ```bash
        pm2 start venv/bin/python --name repostBot -- bot.py
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

1.  **Event Listener:** The bot listens for new messages (`on_message`).
2.  **Media Check:** It checks if the message contains supported image, GIF, or video attachments/embeds.
3.  **Download:** It downloads the media content into memory (or temporarily to disk for video processing).
4.  **Hashing:**
    *   **Images/GIFs:** Calculates a perceptual hash (pHash) of the image (or the first frame of a GIF) using `ImageHash` and `Pillow`.
    *   **Videos:** Extracts the first frame using `OpenCV` (requires FFmpeg backend), converts it to an image, and calculates its pHash.
5.  **Database Query:** It queries the SQLite database (`db/repost_hashes.db`) for existing hashes within the same server (guild) that are visually similar (Hamming distance <= `SIMILARITY_THRESHOLD`).
6.  **Action:**
    *   **If a similar hash is found:** It replies to the new message, linking the original post, and deletes the new message (requires `Manage Messages` permission).
    *   **If no similar hash is found:** It adds the new hash, message details, and timestamp to the database.

## Limitations

*   **Perceptual Hashing Isn't Perfect:** While good, pHash can occasionally have:
    *   **False Positives:** Flagging two different but visually simple/similar images as reposts.
    *   **False Negatives:** Missing reposts if the image/video is heavily edited, cropped significantly, mirrored, or has large overlays added. The `SIMILARITY_THRESHOLD` needs tuning for your server's content.
*   **Video Processing:** Only the **first frame** of a video is analyzed. Videos that are reposted but trimmed to start at a different point will likely not be detected. Robust video fingerprinting is computationally expensive and complex.
*   **Resource Usage:** Processing images and especially videos requires CPU and RAM. Running on very low-resource devices (like older Raspberry Pi models) with high message volume might lead to performance issues or delays. OpenCV can be resource-intensive.
*   **SQLite Scalability:** SQLite is simple but may become slow on extremely large servers with millions of stored hashes. A different database (like PostgreSQL) might be better for massive scale.
*   **Ephemeral Media:** Doesn't handle media that disappears quickly (e.g., certain image hosts or edited messages where the original attachment is removed before processing).

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs or feature suggestions.
