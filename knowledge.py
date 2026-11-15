
"""What Luna knows about you.

The same data lives in TWO places:
  1. data/memory.db    - Luna's real memory
  2. data/profile.json - a readable copy you can edit
"""

import copy
import hashlib
import json
import re
import threading

import config
import memory

_lock = threading.RLock()


# ---------- Small helpers ----------

def dump(data) -> str:
    """Return readable JSON text."""
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load() -> dict:
    """Load Luna's profile from the database."""
    return memory.profile_load()


def prompt_text() -> str:
    """Return profile data for Luna's LLM prompt."""
    data = load()
    return json.dumps(data, ensure_ascii=False) if data else ""


def diff(old: dict, new: dict) -> dict:
    """Return only entries that changed."""
    out = {}

    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            out[key] = {
                "old": old.get(key),
                "new": new.get(key)
            }

    return out


def _write_file(data: dict):
    """Write profile data to profile.json."""
    text = dump(data) + "\n"

    config.PROFILE_JSON.write_text(
        text,
        encoding="utf-8"
    )

    memory.meta_set("json_hash", _hash(text))


# ---------- Keep database and profile.json in sync ----------

def sync() -> dict:
    """Synchronize profile.json with the memory database."""

    with _lock:
        db = memory.profile_load()
        path = config.PROFILE_JSON

        if not path.exists():
            _write_file(db)
            return {}

        text = path.read_text(encoding="utf-8")

        if _hash(text) == memory.meta_get("json_hash"):
            return {}

        try:
            data = json.loads(text)

            if not isinstance(data, dict):
                raise ValueError(
                    "Profile must contain a JSON object."
                )

        except Exception as e:
            path.with_name(
                "profile.broken.json"
            ).write_text(text, encoding="utf-8")

            _write_file(db)

            print(
                f"\nprofile.json contains an error: {e}. "
                "I restored it from memory. "
                "Your edited version is saved as profile.broken.json."
            )

            return {}

        data = {
            str(key): value
            for key, value in data.items()
        }

        changed = diff(db, data)

        memory.profile_replace(data)
        memory.meta_set("json_hash", _hash(text))

        return changed


# ---------- Applying profile changes ----------

def _key(raw) -> str:
    return re.sub(
        r"[^a-z0-9_]+",
        "_",
        str(raw or "").strip().lower()
    ).strip("_")


def _value(raw):
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return raw

    value = str(raw if raw is not None else "").strip()

    if re.fullmatch(r"-?\d+", value):
        return int(value)

    return value[:200]


def _same(a, b) -> bool:
    return str(a).strip().lower() == str(b).strip().lower()


def apply_changes(changes) -> dict:
    """Apply changes to Luna's profile.

    Modes:
      set    Replace an existing value.
      add    Add a value to a list.
      remove Remove a value or an entire key.
    """

    with _lock:
        sync()

        old = memory.profile_load()
        new = copy.deepcopy(old)

        for change in changes or []:

            if not isinstance(change, dict):
                continue

            key = _key(change.get("key"))
            mode = change.get("mode", "set")
            value = _value(change.get("value"))

            if not key:
                continue

            current = new.get(key)

            if mode == "set" and value != "":
                new[key] = value

            elif mode == "add" and value != "":

                if current is None:
                    new[key] = [value]

                elif isinstance(current, list):
                    if not any(
                        _same(item, value)
                        for item in current
                    ):
                        new[key] = current + [value]

                elif not _same(current, value):
                    new[key] = [current, value]

            elif mode == "remove" and current is not None:

                if value == "":
                    del new[key]

                else:
                    items = (
                        current
                        if isinstance(current, list)
                        else [current]
                    )

                    remaining = [
                        item for item in items
                        if not _same(item, value)
                    ]

                    if not remaining:
                        del new[key]

                    else:
                        new[key] = (
                            remaining
                            if isinstance(current, list)
                            else remaining[0]
                        )

        changed = diff(old, new)

        if changed:
            memory.profile_replace(new)
            _write_file(new)

        return changed


# ---------- Vocabulary for Whisper ----------

def vocabulary() -> list:
    """Return useful words from Luna's profile for Whisper hints."""

    data = load()

    words = []
    seen = set()

    def collect(value):

        if isinstance(value, dict):
            for item in value.values():
                collect(item)

        elif isinstance(value, list):
            for item in value:
                collect(item)

        elif isinstance(value, str):

            for word in re.findall(
                r"[A-Za-z][A-Za-z'-]*",
                value
            ):

                normalized = word.lower()

                if len(word) >= 2 and normalized not in seen:
                    seen.add(normalized)
                    words.append(word)

    collect(data)

    return words[:100]
