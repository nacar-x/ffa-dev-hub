import asyncio
import os
import re

import discord
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv
from openai import OpenAI

from rag import ResourceStore

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GUILD_ID = int(os.getenv("GUILD_ID", "1499756701006696478"))
AUTO_INDEX_LIMIT = int(os.getenv("AUTO_INDEX_LIMIT", "200"))
RESOURCE_CHANNEL_IDS = {
    int(x) for x in os.getenv("RESOURCE_CHANNEL_IDS", "").split(",") if x.strip().isdigit()
}
RESOURCE_CATEGORY_IDS = {
    int(x)
    for x in os.getenv(
        "RESOURCE_CATEGORY_IDS",
        "1517695517587673150,1517706715804598364,1499761054656237658",
    ).split(",")
    if x.strip().isdigit()
}

# Gemini exposes an OpenAI-compatible endpoint, so we can reuse the standard
# chat.completions interface instead of a separate SDK.
ai_client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Groq is also OpenAI-compatible, and acts as an instant fallback if Gemini's
# free-tier quota runs out.
groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
) if GROQ_API_KEY else None

store = ResourceStore()

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)

SYSTEM_PROMPT = (
    "You are the AI helper for the Discord server 'FFA Development Hub', a community "
    "about Minecraft server development, especially Skript and plugins. "
    "Answer questions about Skript syntax, plugin usage/config, and general Minecraft "
    "server admin topics clearly and concisely, using code blocks for Skript/YAML when helpful. "
    "If a 'Server channel directory' is provided, use it to answer questions about where a "
    "channel is located or what categories/channels exist. "
    "If 'Server resource context' is provided, prioritize it for questions about specific posted "
    "resources, and mention the resource by author/link when relevant. "
    "If neither is relevant to the question, just answer from your own knowledge — never say you "
    "don't have access to channels or can't view the server; use the directory/context given above "
    "instead. Keep answers focused, no unnecessary repetition. "
    "Always respond in 2-4 short, direct sentences unless a longer explanation or code block is "
    "genuinely needed. No filler like 'I'd be happy to help' or 'Great question!' — go straight "
    "into the answer, like a knowledgeable Discord mod would. "
    "Important: only use 'Server resource context' if it is clearly and directly relevant to the "
    "question — never force a connection between unrelated context and the question just because "
    "it was provided. If the message you're replying to isn't actually a real question (e.g. it's "
    "a casual remark, a request directed at another person, or general server chatter with no "
    "Skript/plugin/resource question in it), say briefly that you're here for Skript/plugin "
    "questions and don't have anything to add — do not invent an interpretation or guess who or "
    "what it's about. "
    "If a matching channel is flagged as IMPORTANT in the context, always point the user to that "
    "channel first, before anything else, since it's the most direct source for that specific "
    "plugin/resource. Never fabricate specific installation steps, config file syntax, download "
    "sources, or version details for a specific/custom/niche plugin or skript unless that exact "
    "information is present in the provided context — if you don't have confirmed specifics, say "
    "you don't have exact details and point them to the matching channel or resource instead of "
    "guessing. Only give generic install steps (drop the jar in /plugins, restart server) for "
    "well-known, standard plugins you're confident about."
)

# Extra instructions appended ONLY for the Groq/Llama fallback model. If you are not the Groq
# model (i.e. this text was somehow included for Gemini), ignore this entire block.
GROQ_SYSTEM_EXTRA = (
    "\n\nFormatting rules (Discord markdown) — follow these strictly: "
    "use **bold** for key terms, plugin/skript names, and important warnings; "
    "use # or ## for section titles only in longer, multi-part answers; "
    "use `inline code` for command names, single settings, or short syntax; "
    "use triple-backtick code blocks for any multi-line Skript/YAML/config; "
    "use bullet points (- item) for lists of steps or options instead of run-on sentences; "
    "use > for quoting something the user referenced; "
    "never use raw asterisks or markdown syntax incorrectly (e.g. no unmatched ** or single *). "
    "Keep formatting clean and professional, not excessive — don't bold every sentence."
)


def find_matching_channels(question: str, guild: discord.Guild | None) -> str:
    """Finds channels whose name matches a word/phrase in the question (e.g. a
    plugin name that also happens to be a channel name). This is checked
    directly against live channel names, independent of the resource search,
    so an exact channel like #altar-s1 always gets surfaced."""
    if guild is None:
        return ""
    q_clean = re.sub(r"[^a-z0-9]+", " ", question.lower())
    q_words = set(q_clean.split())
    q_joined = q_clean.replace(" ", "")

    matches = []
    for channel in guild.text_channels:
        name_clean = re.sub(r"[^a-z0-9]+", " ", channel.name.lower()).strip()
        if not name_clean:
            continue
        name_words = name_clean.split()
        if len(name_clean) >= 3 and (
            all(w in q_words for w in name_words) or name_clean.replace(" ", "") in q_joined
        ):
            matches.append(channel)
    if not matches:
        return ""
    return "Channels whose name directly matches this question: " + ", ".join(f"#{c.name}" for c in matches)


def build_context(question: str) -> str:
    hits = store.search(question, top_k=4)
    if not hits:
        return ""
    blocks = []
    for h in hits:
        snippet = h["content"][:500]
        blocks.append(f"Resource by {h['author']} in #{h['channel']} ({h.get('url', 'no link')}):\n{snippet}")
    return "\n\n".join(blocks)


def build_channel_directory(guild: discord.Guild | None) -> str:
    """Lists the server's actual categories and text channels, so the bot can
    answer navigation questions like 'where is #skripts?' using live data
    instead of the indexed resource content."""
    if guild is None:
        return ""
    lines = []
    for category, channels in guild.by_category():
        cat_name = category.name if category else "No Category"
        visible = [c for c in channels if isinstance(c, discord.TextChannel)]
        if not visible:
            continue
        lines.append(f"{cat_name}: " + ", ".join(f"#{c.name}" for c in visible))
    return "\n".join(lines)


def friendly_ai_error(e: Exception) -> str:
    """Turns raw API errors into a readable message instead of dumping JSON.
    Only reached if Gemini failed AND the Groq fallback also failed (or isn't configured)."""
    text = str(e)
    if is_rate_limit_error(e):
        return (
            "I've hit today's free AI usage limit on all configured providers, so I can't "
            "answer right now. Try again a bit later!"
        )
    return f"Sorry, I hit an error talking to the AI: `{text[:300]}`"


def is_rate_limit_error(e: Exception) -> bool:
    text = str(e).lower()
    return "429" in text or "resource_exhausted" in text or "quota" in text or "rate limit" in text


async def ask_ai(question: str, guild: discord.Guild | None = None) -> str:
    channel_match = find_matching_channels(question, guild)
    context = build_context(question)
    directory = build_channel_directory(guild)

    parts = []
    if channel_match:
        parts.append(f"IMPORTANT — {channel_match}. Always mention this channel first if relevant.")
    if directory:
        parts.append(f"Server channel directory (category: channels):\n{directory}")
    if context:
        parts.append(f"Server resource context:\n{context}")
    parts.append(f"Question: {question}")
    user_content = "\n\n".join(parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        completion = await asyncio.to_thread(
            ai_client.chat.completions.create,
            model=MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=1500,
        )
        return completion.choices[0].message.content
    except Exception as e:
        # Gemini's free-tier quota ran out (or errored) — instantly retry on Groq.
        if groq_client is None or not is_rate_limit_error(e):
            raise
        groq_messages = [
            {"role": "system", "content": SYSTEM_PROMPT + GROQ_SYSTEM_EXTRA},
            {"role": "user", "content": user_content},
        ]
        completion = await asyncio.to_thread(
            groq_client.chat.completions.create,
            model=GROQ_MODEL,
            messages=groq_messages,
            temperature=0.4,
            max_tokens=1500,
        )
        return completion.choices[0].message.content


AI_DISCLAIMER = "-# This response was **AI Generated**, it may contain incorrect information"


def format_reply(answer: str) -> str:
    """Truncates the answer if needed and appends the AI disclaimer, keeping
    the whole message under Discord's 2000 character limit."""
    max_answer_len = 2000 - len(AI_DISCLAIMER) - 2  # 2 chars for the blank line
    if len(answer) > max_answer_len:
        answer = answer[: max_answer_len - 1].rstrip() + "…"
    return f"{answer}\n\n{AI_DISCLAIMER}"


async def update_presence():
    guild = bot.get_guild(GUILD_ID)
    if guild is None:
        return
    activity = discord.Activity(
        type=discord.ActivityType.watching,
        name=f"FFA Dev Hub | {guild.member_count} members",
    )
    await bot.change_presence(activity=activity)


@tasks.loop(minutes=10)
async def presence_refresh():
    await update_presence()


_startup_indexed = False


async def auto_index_categories():
    """Indexes every channel in the configured resource categories. Runs
    once on startup in the background so it doesn't delay bot readiness."""
    guild = bot.get_guild(GUILD_ID)
    if guild is None:
        print("Auto-index skipped: guild not found (check GUILD_ID).")
        return
    total_added = 0
    channels_scanned = 0
    for category in guild.categories:
        if category.id not in RESOURCE_CATEGORY_IDS:
            continue
        for channel in category.text_channels:
            total_added += await index_text_channel(channel, AUTO_INDEX_LIMIT)
            channels_scanned += 1
    print(
        f"Auto-indexed {channels_scanned} channels on startup: "
        f"+{total_added} new resources (total: {store.count()})"
    )


@bot.event
async def on_ready():
    global _startup_indexed
    await bot.tree.sync()
    await update_presence()
    if not presence_refresh.is_running():
        presence_refresh.start()
    if not _startup_indexed:
        _startup_indexed = True
        asyncio.create_task(auto_index_categories())
    print(f"Logged in as {bot.user} | resources indexed: {store.count()}")


@bot.event
async def on_member_join(member: discord.Member):
    if member.guild.id == GUILD_ID:
        await update_presence()


@bot.event
async def on_member_remove(member: discord.Member):
    if member.guild.id == GUILD_ID:
        await update_presence()


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # auto-index new messages posted in designated resource channels
    if message.channel.id in RESOURCE_CHANNEL_IDS and message.content.strip():
        url = message.attachments[0].url if message.attachments else message.jump_url
        store.add(
            {
                "id": str(message.id),
                "channel": message.channel.name,
                "author": str(message.author),
                "content": message.content,
                "url": url,
            }
        )

    # respond when the bot is @mentioned, anywhere
    if bot.user in message.mentions:
        question = message.content.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()
        if question:
            async with message.channel.typing():
                try:
                    answer = await ask_ai(question, message.guild)
                    await message.reply(format_reply(answer))
                except Exception as e:
                    await message.reply(friendly_ai_error(e))

    await bot.process_commands(message)


@bot.tree.command(name="ask", description="Ask the AI about Skript, plugins, or posted resources")
@app_commands.describe(question="Your question")
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    try:
        answer = await ask_ai(question, interaction.guild)
        await interaction.followup.send(format_reply(answer))
    except Exception as e:
        await interaction.followup.send(friendly_ai_error(e))


async def index_text_channel(channel: discord.TextChannel, limit: int) -> int:
    """Scans a channel's history and adds any new messages to the resource
    store. Returns how many new messages were added."""
    added = 0
    async for message in channel.history(limit=limit):
        if message.author.bot or not message.content.strip():
            continue
        url = message.attachments[0].url if message.attachments else message.jump_url
        if store.add(
            {
                "id": str(message.id),
                "channel": channel.name,
                "author": str(message.author),
                "content": message.content,
                "url": url,
            }
        ):
            added += 1
    return added


@bot.tree.command(name="index_channel", description="(Admin) Index past messages in a resource channel")
@app_commands.describe(channel="Channel to scan", limit="How many recent messages to scan (max 500)")
@app_commands.checks.has_permissions(manage_guild=True)
async def index_channel(interaction: discord.Interaction, channel: discord.TextChannel, limit: int = 200):
    await interaction.response.defer(thinking=True)
    limit = min(limit, 500)
    added = await index_text_channel(channel, limit)
    await interaction.followup.send(
        f"Indexed {added} new messages from #{channel.name}. Total resources: {store.count()}"
    )


@bot.tree.command(
    name="index_categories",
    description="(Admin) Index past messages from every channel in the configured resource categories",
)
@app_commands.describe(limit="How many recent messages to scan per channel (max 500)")
@app_commands.checks.has_permissions(manage_guild=True)
async def index_categories(interaction: discord.Interaction, limit: int = 200):
    await interaction.response.defer(thinking=True)
    limit = min(limit, 500)
    guild = interaction.guild
    if guild is None:
        await interaction.followup.send("This command only works inside a server.")
        return

    total_added = 0
    summary_lines = []
    for category in guild.categories:
        if category.id not in RESOURCE_CATEGORY_IDS:
            continue
        for channel in category.text_channels:
            added = await index_text_channel(channel, limit)
            total_added += added
            summary_lines.append(f"#{channel.name}: +{added}")

    if not summary_lines:
        await interaction.followup.send(
            "No matching channels found. Check that RESOURCE_CATEGORY_IDS is set correctly."
        )
        return

    summary = "\n".join(summary_lines)
    reply = f"Indexed {total_added} new messages across {len(summary_lines)} channels:\n{summary}\n\nTotal resources: {store.count()}"
    await interaction.followup.send(reply[:2000])


@bot.tree.command(name="resource_count", description="See how many resources are indexed")
async def resource_count(interaction: discord.Interaction):
    await interaction.response.send_message(f"Currently indexed resources: {store.count()}")


if __name__ == "__main__":
    if not DISCORD_TOKEN or not GEMINI_API_KEY:
        raise SystemExit("Missing DISCORD_TOKEN or GEMINI_API_KEY in environment/.env")
    bot.run(DISCORD_TOKEN)
