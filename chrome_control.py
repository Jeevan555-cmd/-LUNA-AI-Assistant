import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote_plus

WEBSITES = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "github": "https://github.com",
    "gmail": "https://mail.google.com",
    "stackoverflow": "https://stackoverflow.com",
    "chatgpt": "https://chatgpt.com",
    "linkedin": "https://www.linkedin.com",
}


def find_chrome():
    chrome = shutil.which("chrome") or shutil.which("chrome.exe")
    if chrome:
        return chrome

    paths = [
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
    ]

    for path in paths:
        if path.is_file():
            return str(path)

    return None


def open_chrome(url=None):
    executable = find_chrome()
    if not executable:
        return False

    try:
        args = [executable]
        args.append(url if url else "--new-window")
        subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except OSError as error:
        print(f"Chrome launch error: {error}")
        return False


def youtube_search(query):
    query = query.strip()
    if not query:
        return "Tell me what to search for, boss."

    url = "https://www.youtube.com/results?search_query=" + quote_plus(query)

    if open_chrome(url):
        return f"Searching YouTube for {query}, boss."

    return "I couldn't open Google Chrome."


def handle_command(text):
    command = text.lower().strip()

    command = re.sub(r"^(hey\s+)?luna[\s,.:!?-]*", "", command)
    command = re.sub(r"^(?:can|could|would)\s+you\s+", "", command)
    command = re.sub(r"^please\s+", "", command)
    command = re.sub(r"[.!?,]+$", "", command).strip()
    command = re.sub(r"\s+", " ", command)

    if not command:
        return None

    if re.fullmatch(r"(?:open|launch|start)(?: google)? chrome", command):
        return "Opening Google Chrome, boss." if open_chrome() else "Google Chrome was not found."

    # Open YouTube and play/search a song
    match = re.search(
        r"(?:open|launch|visit|go to)\s+(?:the\s+)?youtube\s+and\s+(?:play|search for|find)\s+(.+)",
        command
    )
    if match:
        return youtube_search(match.group(1))

    # Play a song on YouTube
    match = re.fullmatch(r"(?:play|find|search for)\s+(.+?)\s+(?:on|in)\s+youtube", command)
    if match:
        return youtube_search(match.group(1))

    # Direct play command
    match = re.fullmatch(r"(?:play|listen to)\s+(.+)", command)
    if match:
        return youtube_search(match.group(1))

    # Search YouTube
    match = re.fullmatch(r"(?:search youtube for|find youtube for)\s+(.+)", command)
    if match:
        return youtube_search(match.group(1))

    # Google search
    match = re.fullmatch(r"(?:search google for|google|search for)\s+(.+)", command)
    if match:
        query = match.group(1)
        url = "https://www.google.com/search?q=" + quote_plus(query)
        return f"Searching Google for {query}, boss." if open_chrome(url) else "I couldn't open Google Chrome."

    # Open websites
    match = re.fullmatch(
        r"(?:open|launch|visit|go to)\s+(?:the\s+)?(youtube|google|github|gmail|stackoverflow|chatgpt|linkedin)",
        command
    )
    if match:
        site = match.group(1)
        return f"Opening {site}, boss." if open_chrome(WEBSITES[site]) else "I couldn't open Google Chrome."

    return None
