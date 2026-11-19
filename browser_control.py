import re
from pathlib import Path
from playwright.sync_api import sync_playwright

PROFILE = Path(__file__).resolve().parent / "data" / "luna_browser_profile"

CHATGPT_NAMES = r"chatgpt|chat\s*gpt|charge it in|chargeability"


def handle_command(text):
    command = re.sub(
        r"^\s*(?:hey\s+)?luna[\s,:]*",
        "",
        text.strip(),
        flags=re.IGNORECASE
    )

    if not re.search(CHATGPT_NAMES, command, re.IGNORECASE):
        return None

    parts = re.split(
        CHATGPT_NAMES, command, maxsplit=1, flags=re.IGNORECASE
    )
    remainder = parts[1].strip(" ,.-") if len(parts) > 1 else ""

    if not remainder or re.fullmatch(
        r"(please|now|browser|website)?", remainder, re.IGNORECASE
    ):
        import webbrowser
        webbrowser.open("https://chatgpt.com")
        return "Opening ChatGPT, boss."

    question = re.sub(
        r"^\s*(?:and\s+)?(?:please\s+)?(?:ask\s+)?",
        "",
        remainder,
        flags=re.IGNORECASE
    ).strip(" ,.-")

    if not question:
        return "Boss, what should I ask ChatGPT?"

    return ask_chatgpt(question)


def ask_chatgpt(question):
    try:
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(PROFILE),
                headless=False
            )

            page = context.pages[0] if context.pages else context.new_page()

            page.goto(
                "https://chatgpt.com",
                wait_until="domcontentloaded",
                timeout=60000
            )

            print("\nWaiting for ChatGPT to load...")

            selectors = [
                "#prompt-textarea",
                'div[contenteditable="true"]',
                'textarea',
                '[role="textbox"]'
            ]

            editor = None

            for attempt in range(120):
                if page.is_closed():
                    raise Exception("ChatGPT browser page closed.")

                for selector in selectors:
                    elements = page.locator(selector)

                    for i in range(elements.count()):
                        candidate = elements.nth(i)
                        if candidate.is_visible():
                            editor = candidate
                            break

                    if editor:
                        break

                if editor:
                    break

                page.wait_for_timeout(1000)

            if not editor:
                print("Current page:", page.url)
                print("Page title:", page.title())
                return (
                    "ChatGPT message box not found. "
                    "Check that you are logged in and ChatGPT has loaded."
                )

            previous_count = page.locator(
                '[data-message-author-role="assistant"]'
            ).count()

            editor.fill(question)
            editor.press("Enter")

            print("Question sent. Waiting for ChatGPT's answer...")

            page.wait_for_function(
                """(previous) =>
                    document.querySelectorAll(
                        '[data-message-author-role="assistant"]'
                    ).length > previous
                """,
                arg=previous_count,
                timeout=180000
            )

            page.wait_for_timeout(10000)

            answers = page.locator(
                '[data-message-author-role="assistant"]'
            )

            answer = answers.last.inner_text().strip()

            if not answer:
                return "ChatGPT returned an empty answer."

            return answer

    except Exception as error:
        return f"ChatGPT browser error: {error}"
