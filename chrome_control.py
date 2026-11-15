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
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)"))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Google/Chrome/Application/chrome.exe",
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

        if url:
            args.append(url)
        else:
            args.append("--new-window")

        subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True

    except OSError as error:
        print(f"(Chrome launch error: {error})")
        return False


def handle_command(text):
    command = text.strip().lower()

    if not command:
        return None

    command = re.sub(
        r"^(hey\s+)?luna[\s,.:!?-]*",
        "",
        command
    ).strip()

    command = re.sub(r"[.!?,]+$", "", command).strip()
    command = re.sub(r"\s+", " ", command)

    if re.fullmatch(r"(open|launch|start)( google)? chrome", command):
        if open_chrome():
            return "Opening Google Chrome, boss."
        return "Google Chrome was not found."

    match = re.fullmatch(
        r"(?:play|find|search for)\s+(.+?)\s+(?:on|in)\s+youtube",
        command
    )

    if match:
        query = match.group(1).strip()
        url = "https://www.youtube.com/results?search_query=" + quote_plus(query)
        if open_chrome(url):
            return f"Opening YouTube results for {query}, boss."
        return "I couldn't open Google Chrome."

    match = re.fullmatch(
        r"(?:search youtube for|find youtube for)\s+(.+)",
        command
    )

    if match:
        query = match.group(1).strip()
        url = "https://www.youtube.com/results?search_query=" + quote_plus(query)
        if open_chrome(url):
            return f"Searching YouTube for {query}, boss."
        return "I couldn't open Google Chrome."

    match = re.fullmatch(
        r"(?:search google for|google|search for)\s+(.+)",
        command
    )

    if match:
        query = match.group(1).strip()
        url = "https://www.google.com/search?q=" + quote_plus(query)
        if open_chrome(url):
            return f"Searching Google for {query}, boss."
        return "I couldn't open Google Chrome."

    match = re.fullmatch(
        r"(?:open|launch|visit|go to)\s+(?:the\s+)?"
        r"(youtube|google|github|gmail|stackoverflow|chatgpt|linkedin)",
        command
    )

    if match:
        website = match.group(1)
        if open_chrome(WEBSITES[website]):
            return f"Opening {website}, boss."
        return "I couldn't open Google Chrome."

    return None
