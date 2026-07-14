"""Shared, bot-agnostic building blocks for the psycho journal bot.

Reusable pieces the bot is assembled from: Telegram I/O (``tg``), an OpenAI
chat/tool agent (``openai_agent``), Whisper speech-to-text (``stt``), redis state
(``state``), optional Notion task lookup (``notion``), configuration loading
(``config``), a resilient interval scheduler (``scheduler``), and shared error
types (``errors``).
"""

__version__ = "0.1.0"
