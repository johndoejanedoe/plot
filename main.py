import io
import os
import re
import asyncio
from typing import List, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import pytesseract
import sympy as sp
from sympy import Eq, symbols, lambdify
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

# Load .env locally if present
load_dotenv()

TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
if not TOKEN:
    print("[WARN] DISCORD_BOT_TOKEN not set. Set it in Railway Variables.")

INTENTS = discord.Intents.default()
bot = commands.Bot(command_prefix="/", intents=INTENTS)

x, y = symbols("x y")
TRANSFORMS = (
    standard_transformations
    + (implicit_multiplication_application, convert_xor, function_exponentiation)
)

def _normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("≤", "<=").replace("≥", ">=").replace("^", "**")
    return s

def _split_items(text: str) -> List[str]:
    return [i.strip() for i in re.split(r"[;\n]|,| and ", text) if i.strip()]

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
    ax.grid(True, alpha=0.3)

    has_any = False
    mask_all = np.ones_like(Xg, dtype=bool)

    for rel in items:
        F, op = _relation_to_function(rel)
        Fxy = sp.lambdify((x, y), F, "numpy")
        Z = Fxy(Xg, Yg)

        ax.contour(Xg, Yg, Z, levels=[0], colors="k", linewidths=1)

        if op in {"<=", "<", ">=", ">"}:
            has_any = True
            if op in {"<=", "<"}:
                mask = Z <= 0
            else:
                mask = Z >= 0
            mask_all &= mask

    if has_any:
        ax.imshow(np.flipud(mask_all), extent=[xmin, xmax, ymin, ymax],
                  alpha=0.25, cmap="Greys", origin="lower")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def ocr_image_to_text(img_bytes: bytes) -> str:
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

async def _graph_from_text(input_text, x_min, x_max, y_min, y_max):
    exprs = parse_input_to_sympy_list(input_text)
    return await asyncio.to_thread(_plot_items, exprs, (x_min, x_max), (y_min, y_max))

@bot.event
async def on_ready():
    await bot.tree.sync()
    print(f"Logged in as {bot.user}")

@bot.tree.command(name="graph", description="Plot equations/inequalities")
async def graph(interaction: discord.Interaction, input: str, 
                x_min: float=-10, x_max: float=10, y_min: float=-10, y_max: float=10):
    await interaction.response.defer()
    try:
        png = await _graph_from_text(input, x_min, x_max, y_min, y_max)
        await interaction.followup.send(file=discord.File(io.BytesIO(png), "graph.png"))
    except Exception as e:
        await interaction.followup.send(f"Error: {e}")

@bot.tree.command(name="imagegraph", description="OCR an image and plot")
async def imagegraph(interaction: discord.Interaction, image: discord.Attachment,
                     x_min: float=-10, x_max: float=10, y_min: float=-10, y_max: float=10):
    await interaction.response.defer()
    try:
        img_bytes = await image.read()
        text = await asyncio.to_thread(ocr_image_to_text, img_bytes)
        png = await _graph_from_text(text, x_min, x_max, y_min, y_max)
        await interaction.followup.send(content=f"Detected: `{text}`",
                                        file=discord.File(io.BytesIO(png), "graph.png"))
    except Exception as e:
        await interaction.followup.send(f"Error: {e}")

if __name__ == "__main__":
    bot.run(TOKEN)
