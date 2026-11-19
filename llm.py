"""
LUNA AI - LANGUAGE MODEL ENGINE

Features:
- Ollama integration
- Streaming responses
- Conversation history
- Personal profile retrieval
- Personal information extraction
- Permission-based memory proposals
- Wake-word recognition
- Faster first-text delivery
"""

import json
import re
import time
from datetime import datetime

import requests

import config
import knowledge
import memory


# =====================================================
# WAKE WORD DETECTION
# =====================================================

_WAKE_WORDS = sorted(
    (re.escape(w) for w in config.WAKE_WORDS),
    key=len,
    reverse=True
)

_ACTIVATE_WORDS = sorted(
    (re.escape(w) for w in config.ACTIVATE_WORDS),
    key=len,
    reverse=True
)

_WAKE = "|".join(_WAKE_WORDS)
_ACT = "|".join(_ACTIVATE_WORDS)

_WAKE_RE = re.compile(
    rf"\b(?:{_WAKE})\b",
    re.IGNORECASE
)

_ACT_ONLY_RE = re.compile(
    rf"^\W*(?:{_ACT})\W*$",
    re.IGNORECASE
)

_LEAD_RE = re.compile(
    rf"^\s*(?:(?:hey|hi|hello|ok|okay)\s+)?"
    rf"(?:{_WAKE})\b"
    rf"(?:[\s,.:!?-]+(?:{_ACT})\b)?"
    rf"[\s,.:!?-]*",
    re.IGNORECASE
)


# =====================================================
# PERSONAL INFORMATION DETECTION
# =====================================================

_PERSONAL_RE = re.compile(
    r"\b("
    r"my|i am|i'm|i was|i have|i've|i live|i stay|"
    r"i work|i study|i like|i love|i hate|i prefer|"
    r"call me|remember|save|store|update|change|"
    r"forget|delete|remove|note"
    r")\b",
    re.IGNORECASE
)


# =====================================================
# MEMORY EXTRACTION SCHEMA
# =====================================================

_CHANGE_SCHEMA = {
    "type": "object",
    "properties": {
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["set", "add", "remove"]
                    }
                },
                "required": ["key", "value", "mode"]
            }
        }
    },
    "required": ["changes"]
}


# =====================================================
# MEMORY EXTRACTION PROMPT
# =====================================================

_EXTRACT_PROMPT = """
You are Luna's personal memory analyzer.

Analyze the user's latest message and identify explicit
personal information that may belong in a long-term profile.

RULES:

1. Only extract facts explicitly stated by the user.
2. Never guess or invent personal information.
3. Ignore questions, hypothetical examples and fiction.
4. Ignore temporary information and ordinary small talk.
5. Do not extract information about other people unless
   the user explicitly identifies their relationship.
6. Do not infer occupation, age or location.
7. If the user corrects an existing fact, use mode "set".
8. If the user introduces another family member or friend,
   use mode "add".
9. If the user asks to forget information, use mode "remove".
10. If there is no useful information, return an empty list.

PROFILE KEYS:

name, age, birthday, city, hometown, job, education,
father, mother, brother, sister, friend, wife,
favorite_color, hobby, career_goal.

You may create another descriptive snake_case key
when appropriate.

MODES:

set:
Replace an existing value.

add:
Add another value to a list.

remove:
Remove a particular value or the entire key.

EXAMPLES:

"My age is 23"
-> {"changes":[{"key":"age","value":"23","mode":"set"}]}

"My brother's name is Ravi"
-> {"changes":[{"key":"brother","value":"Ravi","mode":"add"}]}

"I live in Mysuru"
-> {"changes":[{"key":"city","value":"Mysuru","mode":"set"}]}

"I am studying engineering"
-> {"changes":[{"key":"education","value":"engineering","mode":"set"}]}

"Forget my age"
-> {"changes":[{"key":"age","value":"","mode":"remove"}]}

"Tell me a joke"
-> {"changes":[]}

Current profile:

<PROFILE>
"""


# =====================================================
# OLLAMA SETTINGS
# =====================================================

def _options(**extra):
    """Build Ollama generation options."""

    return {
        "num_ctx": config.LLM_CONTEXT,
        **extra
    }


def _timeout():
    return getattr(config, "LLM_TIMEOUT", 120)


# =====================================================
# WAKE WORD FUNCTIONS
# =====================================================

def is_addressed(text: str) -> bool:
    """Check whether the user addressed Luna."""

    if _WAKE_RE.search(text):
        return True

    return (
        len(text.split()) <= 3
        and bool(_ACT_ONLY_RE.fullmatch(text))
    )


def strip_wake(text: str) -> str:
    """Remove Luna's wake word from the beginning."""

    if _ACT_ONLY_RE.fullmatch(text.strip()):
        return ""

    return _LEAD_RE.sub(
        "",
        text.strip(),
        count=1
    ).strip()


# =====================================================
# STREAMING TEXT BUFFER
# =====================================================

_SPLIT = re.compile(
    r"(?<=[.!?])\s+|\n+"
)


def pop_sentences(buf: str):
    """
    Release completed sentences.

    If a sentence becomes long, release a partial
    chunk at a word boundary so Piper can start sooner.
    """

    parts = _SPLIT.split(buf)

    ready = parts[:-1]
    rest = parts[-1]

    output = [
        part.strip()
        for part in ready
        if part.strip()
    ]

    # Start speaking before the entire sentence is generated.
    if len(rest) >= 24:

        cut = rest.rfind(" ", 0, 24)

        if cut >= 12:

            chunk = rest[:cut].strip()

            if chunk:
                output.append(chunk)

            rest = rest[cut + 1:].lstrip()

    return output, rest


# =====================================================
# MAIN ASSISTANT CLASS
# =====================================================

class Assistant:

    def __init__(self):

        self.history = memory.load_recent(
            config.MEMORY_MESSAGES
        )

        if not isinstance(self.history, list):
            self.history = []

    # =================================================
    # BUILD MESSAGES
    # =================================================

    def _build_messages(self, user_text: str):

        system = config.SYSTEM_PROMPT
        system += """

STRICT ACTION AND RESPONSE RULES:

1. You are LUNA, a helpful personal AI assistant.
2. Never claim that YouTube, Chrome, Google, ChatGPT,
   or any website was opened, closed, searched, or is still open
   unless an automation handler explicitly confirmed that action.
3. You cannot independently see or control the user's browser.
4. Never invent completed actions or pretend an action succeeded.
5. If a website command was not recognized by LUNA's automation,
   politely ask the user to repeat the command.
6. For greetings such as hello or hi, respond with a natural greeting.
7. Answer normal questions directly and relevantly.
8. Do not repeat unrelated statements from previous conversation history.
9. Previous assistant claims about browser activity are not proof
   that an action actually happened.

"""


        current_time = datetime.now().strftime(
            "%A, %d %B %Y, %I:%M %p"
        )

        system += (
            "\n\nCurrent date and time: "
            + current_time
        )

        profile = knowledge.prompt_text()

        if profile:
            system += (
                "\n\nUSER PROFILE "
                "(trusted personal facts):\n"
                + profile
            )
        else:
            system += (
                "\n\nNo personal profile information "
                "is available yet."
            )

        return (
            [
                {
                    "role": "system",
                    "content": system
                }
            ]
            + self.history
            + [
                {
                    "role": "user",
                    "content": user_text
                }
            ]
        )

    # =================================================
    # SAVE CONVERSATION
    # =================================================

    def remember(self, user_text: str, reply: str):

        memory.save("user", user_text)
        memory.save("assistant", reply)

        self.history.extend([
            {
                "role": "user",
                "content": user_text
            },
            {
                "role": "assistant",
                "content": reply
            }
        ])

        self.history = self.history[
            -config.MEMORY_MESSAGES:
        ]

    # =================================================
    # WARM UP MODEL
    # =================================================

    def warm_up(self):
        """Load the Ollama model before the first conversation."""

        try:

            response = requests.post(
                config.OLLAMA_URL,
                json={
                    "model": config.LLM_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Hello"
                        }
                    ],
                    "stream": False,
                    "keep_alive": config.LLM_KEEP_ALIVE,
                    "options": _options(num_predict=1)
                },
                timeout=300
            )

            response.raise_for_status()

            print("(Luna AI model is ready.)")

        except requests.RequestException as error:

            print(
                f"\n(Ollama connection failed: {error})"
            )

    # =================================================
    # ANALYZE PERSONAL INFORMATION
    # =================================================

    def propose_changes(self, user_text: str) -> list:
        """
        Analyze personal information.

        Does not save anything.
        main.py must ask permission before applying changes.
        """

        text = user_text.strip()

        if not text:
            return []

        if text.endswith("?"):
            return []

        if not _PERSONAL_RE.search(text):
            return []

        profile = knowledge.prompt_text() or "{}"

        extraction_prompt = _EXTRACT_PROMPT.replace(
            "<PROFILE>",
            profile
        )

        payload = {
            "model": config.LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": extraction_prompt
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            "format": _CHANGE_SCHEMA,
            "stream": False,
            "keep_alive": config.LLM_KEEP_ALIVE,
            "options": _options(
                num_predict=200,
                temperature=0
            )
        }

        try:

            response = requests.post(
                config.OLLAMA_URL,
                json=payload,
                timeout=_timeout()
            )

            if response.status_code in (400, 404, 422):

                payload["format"] = "json"

                response = requests.post(
                    config.OLLAMA_URL,
                    json=payload,
                    timeout=_timeout()
                )

            response.raise_for_status()

            result = response.json()

            content = result.get(
                "message", {}
            ).get(
                "content",
                "{}"
            )

            data = json.loads(content)

            changes = data.get("changes", [])

            if not isinstance(changes, list):
                return []

            valid = []

            for change in changes:

                if not isinstance(change, dict):
                    continue

                key = change.get("key", "")
                value = change.get("value", "")
                mode = change.get("mode", "")

                if not isinstance(key, str):
                    continue

                if not isinstance(value, str):
                    continue

                if mode not in ("set", "add", "remove"):
                    continue

                if not key.strip():
                    continue

                valid.append({
                    "key": key,
                    "value": value,
                    "mode": mode
                })

            return valid

        except (
            requests.RequestException,
            ValueError,
            TypeError,
            KeyError
        ) as error:

            print(
                f"\n(Could not analyze personal information: {error})"
            )

            return []

    # =================================================
    # STREAM AI RESPONSE
    # =================================================

    def stream_sentences(self, user_text: str):
        """
        Stream Ollama output to Piper.

        The first useful text is released as soon as
        enough words are generated.
        """

        full = []
        buf = ""

        started = time.perf_counter()

        first_token_logged = False
        first_text_logged = False

        payload = {
            "model": config.LLM_MODEL,
            "messages": self._build_messages(user_text),
            "stream": True,
            "keep_alive": config.LLM_KEEP_ALIVE,
            "options": _options(
                num_predict=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE
            )
        }

        try:

            with requests.post(
                config.OLLAMA_URL,
                json=payload,
                stream=True,
                timeout=_timeout()
            ) as response:

                response.raise_for_status()

                for line in response.iter_lines(
                    chunk_size=1,
                    decode_unicode=True
                ):

                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                    except (
                        json.JSONDecodeError,
                        TypeError
                    ):
                        continue

                    # ---------------------------------
                    # OLLAMA GENERATION METADATA
                    # ---------------------------------

                    if data.get("done"):

                        reason = data.get(
                            "done_reason",
                            "unknown"
                        )

                        generated_tokens = data.get(
                            "eval_count",
                            0
                        )

                        print(
                            f"\n[Generation finished: {reason}]"
                        )

                        print(
                            f"[Generated tokens: {generated_tokens}]"
                        )

                        if reason == "length":

                            print(
                                "[Warning: Response reached "
                                "the token limit.]"
                            )

                    # ---------------------------------
                    # READ GENERATED TEXT
                    # ---------------------------------

                    piece = data.get(
                        "message",
                        {}
                    ).get(
                        "content",
                        ""
                    )

                    if not piece:
                        continue

                    # ---------------------------------
                    # FIRST TOKEN TIMING
                    # ---------------------------------

                    if not first_token_logged:

                        print(
                            f"\n[Ollama first token: "
                            f"{time.perf_counter() - started:.2f}s]"
                        )

                        first_token_logged = True

                    full.append(piece)
                    buf += piece

                    # ---------------------------------
                    # SPLIT INTO SPEAKABLE CHUNKS
                    # ---------------------------------

                    sentences, buf = pop_sentences(buf)

                    for sentence in sentences:

                        if not sentence.strip():
                            continue

                        if not first_text_logged:

                            print(
                                f"\n[First speakable text: "
                                f"{time.perf_counter() - started:.2f}s]"
                            )

                            first_text_logged = True

                        yield sentence

            # -----------------------------------------
            # SEND REMAINING TEXT
            # -----------------------------------------

            if buf.strip():

                if not first_text_logged:

                    print(
                        f"\n[First speakable text: "
                        f"{time.perf_counter() - started:.2f}s]"
                    )

                    first_text_logged = True

                yield buf.strip()

            # -----------------------------------------
            # SAVE COMPLETE CONVERSATION
            # -----------------------------------------

            complete_reply = "".join(full).strip()

            if complete_reply:

                self.remember(
                    user_text,
                    complete_reply
                )

            print(
                f"\n[LLM generation completed: "
                f"{time.perf_counter() - started:.2f}s]"
            )

        except requests.RequestException as error:

            print(
                f"\n(Luna LLM error: {error})"
            )

            if not full:

                yield (
                    "I'm sorry boss, I'm having trouble "
                    "connecting to my AI model right now."
                )

    # =================================================
    # NORMAL NON-STREAMING RESPONSE
    # =================================================

    def ask(self, user_text: str) -> str:
        """Return a complete response as one string."""

        return " ".join(
            self.stream_sentences(user_text)
        )
