import asyncio
import os

import discord
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv
from openai import OpenAI

from rag import ResourceStore

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GUILD_ID = int(os.getenv("GUILD_ID", "1499756701006696478"))
AUTO_INDEX_LIMIT = int(os.getenv("AUTO_INDEX_LIMIT", "200"))
RESOURCE_CHANNEL_IDS = {
    int(x) for x in os.getenv("RESOURCE_CHANNEL_IDS", "").split(",") if x.strip().isdigit()
}
RESOURCE_CATEGORY_IDS = {
    int(x)
    for x in os.getenv("RESOURCE_CATEGORY_IDS", "1517695517587673150,1517706715804598364").split(",")
    if x.strip().isdigit()
}

# Gemini exposes an OpenAI-compatible endpoint, so we can reuse the standard
# chat.completions interface instead of a separate SDK.
ai_client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
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
    "instead. Keep answers focused, no unnecessary repetition."
)


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


async def ask_ai(question: str, guild: discord.Guild | None = None) -> str:
    context = build_context(question)
    directory = build_channel_directory(guild)

    parts = []
    if directory:
        parts.append(f"Server channel directory (category: channels):\n{directory}")
    if context:
        parts.append(f"Server resource context:\n{context}")
    parts.append(f"Question: {question}")
    user_content = "\n\n".join(parts)

    completion = await asyncio.to_thread(
        ai_client.chat.completions.create,
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.4,
        max_tokens=800,
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
                except Exception as e:
                    answer = f"Sorry, I hit an error talking to the AI: `{e}`"
            await message.reply(format_reply(answer))

    await bot.process_commands(message)


@bot.tree.command(name="ask", description="Ask the AI about Skript, plugins, or posted resources")
@app_commands.describe(question="Your question")
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    try:
        answer = await ask_ai(question, interaction.guild)
    except Exception as e:
        answer = f"Sorry, I hit an error talking to the AI: `{e}`"
    await interaction.followup.send(format_reply(answer))


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
