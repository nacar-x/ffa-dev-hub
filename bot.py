import asyncio
import os

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from openai import OpenAI

from rag import ResourceStore

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
RESOURCE_CHANNEL_IDS = {
    int(x) for x in os.getenv("RESOURCE_CHANNEL_IDS", "").split(",") if x.strip().isdigit()
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
    "If 'server resource context' is provided below, prioritize it and mention the resource "
    "by author/link when relevant. If nothing relevant is provided, just answer from your own "
    "knowledge. Keep answers focused, no unnecessary repetition."
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


async def ask_ai(question: str) -> str:
    context = build_context(question)
    user_content = question if not context else f"Server resource context:\n{context}\n\nQuestion: {question}"
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


@bot.event
async def on_ready():
    await bot.tree.sync()
    print(f"Logged in as {bot.user} | resources indexed: {store.count()}")


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
                    answer = await ask_ai(question)
                except Exception as e:
                    answer = f"Sorry, I hit an error talking to the AI: `{e}`"
            await message.reply(answer[:2000])

    await bot.process_commands(message)


@bot.tree.command(name="ask", description="Ask the AI about Skript, plugins, or posted resources")
@app_commands.describe(question="Your question")
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    try:
        answer = await ask_ai(question)
    except Exception as e:
        answer = f"Sorry, I hit an error talking to the AI: `{e}`"
    await interaction.followup.send(answer[:2000])


@bot.tree.command(name="index_channel", description="(Admin) Index past messages in a resource channel")
@app_commands.describe(channel="Channel to scan", limit="How many recent messages to scan (max 500)")
@app_commands.checks.has_permissions(manage_guild=True)
async def index_channel(interaction: discord.Interaction, channel: discord.TextChannel, limit: int = 200):
    await interaction.response.defer(thinking=True)
    limit = min(limit, 500)
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
    await interaction.followup.send(
        f"Indexed {added} new messages from #{channel.name}. Total resources: {store.count()}"
    )


@bot.tree.command(name="resource_count", description="See how many resources are indexed")
async def resource_count(interaction: discord.Interaction):
    await interaction.response.send_message(f"Currently indexed resources: {store.count()}")


if __name__ == "__main__":
    if not DISCORD_TOKEN or not GEMINI_API_KEY:
        raise SystemExit("Missing DISCORD_TOKEN or GEMINI_API_KEY in environment/.env")
    bot.run(DISCORD_TOKEN)
