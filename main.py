# main.py
import io
import os
import re
import asyncio
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import pytesseract
import sympy as sp
from sympy import Eq, symbols
from sympy.core.relational import Relational, Lt, Le, Gt, Ge
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
)

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

# ----- ENV / Discord setup -----
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")

if TOKEN:
    print(f"[INFO] Token detected (length {len(TOKEN)})")
else:
    print("[WARN] DISCORD_BOT_TOKEN not set. Set it in Railway Variables.")

INTENTS = discord.Intents.default()
bot = commands.Bot(command_prefix="/", intents=INTENTS)

x, y = symbols("x y")
TRANSFORMS = (
    standard_transformations
    + (implicit_multiplication_application, convert_xor, function_exponentiation)
)

# ----- Parsing helpers -----
def _normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("≤", "<=").replace("≥", ">=").replace("^", "**")
    s = s.replace("—", "-").replace("–", "-")
    s = re.sub(r"\s+", " ", s)
    return s

def _split_items(text: str) -> List[str]:
    return [i.strip() for i in re.split(r"[;\n]|,|\band\b|\bAND\b", text) if i.strip()]

def _ensure_eq(expr_str: str):
    s = _normalize_text(expr_str)
    if any(op in s for op in ["<=", ">=", "<", ">", "="]):
        return parse_expr(s, transformations=TRANSFORMS, evaluate=False)
    else:
        expr = parse_expr(s, transformations=TRANSFORMS)
        return Eq(y, expr)

def parse_input_to_sympy_list(text: str):
    return [_ensure_eq(it) for it in _split_items(text)]

def _relation_to_function(rel):
    if isinstance(rel, Relational):
        F = sp.simplify(rel.lhs - rel.rhs)
        if isinstance(rel, Le): return F, "<="
        if isinstance(rel, Lt): return F, "<"
        if isinstance(rel, Ge): return F, ">="
        if isinstance(rel, Gt): return F, ">"
        return F, "="
    elif isinstance(rel, Eq):
        return sp.simplify(rel.lhs - rel.rhs), "="
    else:
        return sp.simplify(rel), "="

# ----- Plotting -----
def _make_grid(xmin, xmax, ymin, ymax, n=400):
    xx = np.linspace(xmin, xmax, n)
    yy = np.linspace(ymin, ymax, n)
    return np.meshgrid(xx, yy)

def _plot_items(items, xlim, ylim):
    xmin, xmax = xlim
    ymin, ymax = ylim
    Xg, Yg = _make_grid(xmin, xmax, ymin, ymax)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

    has_any_ineq = False
    mask_all = np.ones_like(Xg, dtype=bool)

    for rel in items:
        F, op = _relation_to_function(rel)
        Fxy = sp.lambdify((x, y), F, "numpy")
        with np.errstate(all="ignore"):
            Z = Fxy(Xg, Yg)

        # boundary curve F=0
        try:
            ax.contour(Xg, Yg, Z, levels=[0], linewidths=1.2)
        except Exception:
            pass

        # shade inequality intersection
        if op in {"<=", "<", ">=", ">"}:
            has_any_ineq = True
            if op in {"<=", "<"}:
                mask = Z <= 0 if op == "<=" else Z < 0
            else:
                mask = Z >= 0 if op == ">=" else Z > 0
            mask_all &= np.nan_to_num(mask, nan=False)

    if has_any_ineq:
        ax.imshow(
            np.flipud(mask_all),
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            alpha=0.25,
            cmap="Greys",
            interpolation="nearest",
            aspect="auto",
        )

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ----- OCR -----
def ocr_image_to_text(img_bytes: bytes) -> str:
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return _normalize_text(text.strip())

# ----- Async wrappers -----
async def _graph_from_text(input_text, x_min, x_max, y_min, y_max):
    exprs = parse_input_to_sympy_list(input_text)
    return await asyncio.to_thread(_plot_items, exprs, (x_min, x_max), (y_min, y_max))

# ----- Discord events & commands -----
@bot.event
async def on_ready():
    try:
        await bot.tree.sync()
        print(f"Logged in as {bot.user}")
    except Exception as e:
        print("[SYNC ERROR]", e)

@bot.tree.command(name="graph", description="Plot equations/inequalities")
async def graph(
    interaction: discord.Interaction,
    input: str,
    x_min: float = -10.0,
    x_max: float = 10.0,
    y_min: float = -10.0,
    y_max: float = 10.0,
):
    await interaction.response.defer(thinking=True)
    try:
        png = await _graph_from_text(input, x_min, x_max, y_min, y_max)
        await interaction.followup.send(file=discord.File(io.BytesIO(png), "graph.png"))
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {e}")

@bot.tree.command(name="imagegraph", description="OCR an image and plot")
async def imagegraph(
    interaction: discord.Interaction,
    image: discord.Attachment,
    x_min: float = -10.0,
    x_max: float = 10.0,
    y_min: float = -10.0,
    y_max: float = 10.0,
):
    await interaction.response.defer(thinking=True)
    try:
        img_bytes = await image.read()
        text = await asyncio.to_thread(ocr_image_to_text, img_bytes)
        if not text:
            await interaction.followup.send("⚠️ Couldn't read any text from the image.")
            return
        png = await _graph_from_text(text, x_min, x_max, y_min, y_max)
        await interaction.followup.send(
            content=f"**Detected:** `{text}`",
            file=discord.File(io.BytesIO(png), "graph.png"),
        )
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {e}")

if __name__ == "__main__":
    bot.run(TOKEN)
