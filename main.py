# ============================
# main.py
# ============================

"""
Discord bot that:
  • Plots functions (y = f(x), implicit equations F(x,y)=0)
  • Shades regions for inequalities (e.g., y <= 2x + 1, x^2 + y^2 < 9)
  • Reads problems from TEXT and from IMAGE (OCR via Tesseract)

Slash commands:
  /graph input:"y<=2x+1; y>=x-2; x>=-1" x_min:-10 x_max:10 y_min:-10 y_max:10
  /imagegraph image:<attach image> x_min:-10 x_max:10 y_min:-10 y_max:10

Deploy with Docker (includes Tesseract). See README.md for step-by-step.
"""

import io
import os
import re
import asyncio
from typing import List, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt

from PIL import Image
import cv2
import pytesseract

from sympy import Eq, symbols, lambdify
from sympy.core.relational import Relational, Lt, Le, Gt, Ge
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
)
import sympy as sp

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

# Load .env locally if present
load_dotenv()

TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
if not TOKEN:
    print("[WARN] DISCORD_BOT_TOKEN not set. Set it in your hosting env.")

INTENTS = discord.Intents.default()
INTENTS.message_content = False  # we use slash commands

bot = commands.Bot(command_prefix="/", intents=INTENTS)

tx = symbols("x")
ty = symbols("y")
X, Y = tx, ty

TRANSFORMS = (
    standard_transformations + (implicit_multiplication_application, convert_xor, function_exponentiation)
)

REL_SEPS = ["<=", ">=", "<", ">", "="]


def _normalize_text(s: str) -> str:
    s = s.strip()
    # normalize unicode comparators
    s = s.replace("≤", "<=").replace("≥", ">=").replace("≠", "!=").replace("＝", "=")
    # caret to power
    s = s.replace("^", "**")
    # common OCR quirks
    s = s.replace("—", "-").replace("–", "-")
    # remove double spaces
    s = re.sub(r"\s+", " ", s)
    return s


def _split_items(text: str) -> List[str]:
    # Split on ;, newline, or explicit ANDs/commas
    items = re.split(r"[;\n]|\band\b|\bAND\b|,", text)
    return [i.strip() for i in items if i.strip()]


def _has_relational(s: str) -> bool:
    return any(op in s for op in REL_SEPS)


def _ensure_eq(expr_str: str):
    """If user gives y=f(x) or x=g(y) etc., return a SymPy Eq(...). If F(x,y)=0 implicit, return Eq(F,0)."""
    s = _normalize_text(expr_str)
    if _has_relational(s):
        # parse relational/equality as-is
        try:
            rel = parse_expr(s, transformations=TRANSFORMS, evaluate=False)
            return rel
        except Exception:
            # Fallback: try sympify
            return sp.sympify(s, evaluate=False)
    else:
        # No comparator: treat as implicit equation F(x,y)=0 OR y=f(x)
        # Heuristic: if contains 'y' or 'x' both; else assume y=f(x)
        # If it's in form y - (something), OK; else treat as y = expr(x)
        try:
            expr = parse_expr(s, transformations=TRANSFORMS)
        except Exception:
            expr = sp.sympify(s)
        free = expr.free_symbols
        if ty in free or (tx in free and ty in free):
            return Eq(expr, 0)
        else:
            # treat as y = expr(x)
            return Eq(ty, expr)


def parse_input_to_sympy_list(text: str) -> List[Union[Relational, Eq]]:
    text = _normalize_text(text)
    items = _split_items(text)
    parsed = []
    for it in items:
        if not it:
            continue
        parsed.append(_ensure_eq(it))
    return parsed


def _relation_to_function(rel: Union[Relational, Eq]):
    """
    For a relational like y <= x+1, return (F(x,y), comparator)
    where comparator in {"<=", ">=", "<", ">", "="}
    For Eq, comparator is "=" and F is lhs - rhs
    """
    if isinstance(rel, Relational):
        lhs = sp.simplify(rel.lhs)
        rhs = sp.simplify(rel.rhs)
        F = sp.simplify(lhs - rhs)
        if isinstance(rel, Le):
            return F, "<="
        if isinstance(rel, Lt):
            return F, "<"
        if isinstance(rel, Ge):
            return F, ">="
        if isinstance(rel, Gt):
            return F, ">"
        # Eq or Ne (we won't plot Ne regions; treat as Eq)
        return F, "="
    elif isinstance(rel, Eq):
        F = sp.simplify(rel.lhs - rel.rhs)
        return F, "="
    else:
        F = sp.simplify(rel)
        return F, "="


def _make_grid(xmin, xmax, ymin, ymax, n=400):
    xx = np.linspace(xmin, xmax, n)
    yy = np.linspace(ymin, ymax, n)
    Xg, Yg = np.meshgrid(xx, yy)
    return Xg, Yg


def _plot_items(items: List[Union[Relational, Eq]], xlim: Tuple[float, float], ylim: Tuple[float, float]) -> bytes:
    xmin, xmax = xlim
    ymin, ymax = ylim

    Xg, Yg = _make_grid(xmin, xmax, ymin, ymax, n=400)

    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = plt.gca()
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

    # global mask for intersection of inequalities
    has_any_inequality = False
    intersection_mask = np.ones_like(Xg, dtype=bool)

    # Evaluate and render each item
    for idx, rel in enumerate(items):
        Fsym, op = _relation_to_function(rel)

        # special easy cases: y = f(x) or x = c -> draw crisp curves
        # detect if Eq(y, f(x))
        if op == "=" and isinstance(rel, Eq):
            lhs, rhs = rel.lhs, rel.rhs
            # y = f(x)
            if lhs == ty and (rhs.free_symbols <= {tx}):
                f = lambdify(tx, rhs, "numpy")
                xs = np.linspace(xmin, xmax, 1200)
                with np.errstate(all='ignore'):
                    ys = f(xs)
                ax.plot(xs, ys, linewidth=1.8, label=str(rel))
                continue
            # x = g(y) (vertical function)
            if lhs == tx and (rhs.free_symbols <= {ty}):
                g = lambdify(ty, rhs, "numpy")
                ys = np.linspace(ymin, ymax, 1200)
                with np.errstate(all='ignore'):
                    xs = g(ys)
                ax.plot(xs, ys, linewidth=1.8, label=str(rel))
                continue

        # General implicit: contour at F=0 (the boundary)
        Fxy = lambdify((tx, ty), Fsym, "numpy")
        with np.errstate(all='ignore'):
            Z = Fxy(Xg, Yg)

        # Draw boundary
        try:
            cs = ax.contour(Xg, Yg, Z, levels=[0], linewidths=1.4)
            for c in cs.collections:
                c.set_label(str(rel))
        except Exception:
            # In case contour fails (NaNs everywhere), skip boundary
            pass

        # Shade inequality regions (build intersection)
        if op in {"<=", "<", ">=", ">"}:
            has_any_inequality = True
            if op == "<=" or op == "<":
                mask = Z <= 0 if op == "<=" else Z < 0
            else:
                mask = Z >= 0 if op == ">=" else Z > 0
            # combine as AND by default
            intersection_mask &= np.nan_to_num(mask, nan=False)

    # After processing all, render the final shaded intersection
    if has_any_inequality:
        # Show a semi-transparent shading for True region
        ax.imshow(
            np.flipud(intersection_mask),
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            alpha=0.25,
            cmap="Greys",
            interpolation="nearest",
            aspect="auto",
        )

    if len(ax.get_legend_handles_labels()[0]) > 0:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def ocr_image_to_text(img_bytes: bytes) -> str:
    # Load and pre-process image for better OCR
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # adaptive threshold to clean background
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 35, 10)
    # slight dilation to connect symbols like '<='
    kernel = np.ones((1, 1), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)

    # OCR via pytesseract (ensure tesseract-ocr is installed in container)
    config = "--psm 6 -c preserve_interword_spaces=1"
    text = pytesseract.image_to_string(thr, config=config)

    # Basic cleanup
    text = text.replace("\t", " ")
    text = re.sub(r"[\r\n]+", "\n", text)
    text = text.strip()
    return text


async def _graph_from_text(input_text: str, x_min: float, x_max: float, y_min: float, y_max: float) -> bytes:
    exprs = parse_input_to_sympy_list(input_text)
    png = await asyncio.to_thread(_plot_items, exprs, (x_min, x_max), (y_min, y_max))
    return png


@bot.event
async def on_ready():
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} commands. Logged in as {bot.user}.")
    except Exception as e:
        print("[SYNC ERROR]", e)


@bot.tree.command(name="graph", description="Plot graphs and shade inequality regions from text input")
@app_commands.describe(
    input="Math: functions or inequalities. Separate multiple with ';'",
    x_min="x-axis min",
    x_max="x-axis max",
    y_min="y-axis min",
    y_max="y-axis max",
)
async def graph(interaction: discord.Interaction, input: str, x_min: float = -10, x_max: float = 10, y_min: float = -10, y_max: float = 10):
    await interaction.response.defer(thinking=True)
    try:
        png = await _graph_from_text(input, x_min, x_max, y_min, y_max)
        file = discord.File(io.BytesIO(png), filename="graph.png")
        await interaction.followup.send(file=file)
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {e}")


@bot.tree.command(name="imagegraph", description="OCR an image and plot/shade the detected math")
@app_commands.describe(
    image="Attach a screenshot/photo of the question",
    x_min="x-axis min",
    x_max="x-axis max",
    y_min="y-axis min",
    y_max="y-axis max",
)
async def imagegraph(interaction: discord.Interaction, image: discord.Attachment, x_min: float = -10, x_max: float = 10, y_min: float = -10, y_max: float = 10):
    await interaction.response.defer(thinking=True)
    try:
        img_bytes = await image.read()
        text = await asyncio.to_thread(ocr_image_to_text, img_bytes)
        if not text:
            await interaction.followup.send("⚠️ Couldn't read any text from the image. Try a clearer picture.")
            return

        # Keep only lines that look mathy
        math_lines = []
        for line in text.splitlines():
            line_n = _normalize_text(line)
            if re.search(r"[xy]", line_n) and (re.search(r"[<>=]", line_n) or re.search(r"[\+\-*/^()]", line_n)):
                math_lines.append(line_n)
        if not math_lines:
            math_lines = [text]

        parsed_text = "; ".join(math_lines)
        png = await _graph_from_text(parsed_text, x_min, x_max, y_min, y_max)
        file = discord.File(io.BytesIO(png), filename="graph.png")
        await interaction.followup.send(
            content=f"**Detected:** \n``{parsed_text}``", file=file
        )
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {e}")


if __name__ == "__main__":
    bot.run(TOKEN)


# ============================
# requirements.txt
# ============================
# Pin reasonably recent versions
# Discord & math stack
# Note: opencv-python-headless avoids X11 dependency on servers

discord.py==2.4.0
sympy==1.13.2
matplotlib==3.9.0
numpy==2.0.1
pillow==10.4.0
pytesseract==0.3.13
opencv-python-headless==4.10.0.84
python-dotenv==1.0.1


# ============================
# Dockerfile
# ============================
# Build a lightweight image with Tesseract installed

FROM python:3.11-slim

# Install system deps and Tesseract OCR
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       tesseract-ocr \
       tesseract-ocr-eng \
       build-essential \
       libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set Python to unbuffered for better logs
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]


# ============================
# .env.example
# ============================
# Copy to .env and fill your token for local testing
# NEVER commit your real token.

DISCORD_BOT_TOKEN=PASTE_YOUR_TOKEN_HERE


# ============================
# README.md
# ============================
# Discord Graph & Inequality Bot (Text + Image OCR)

This bot plots functions, shades regions for inequalities, and can read math from images via OCR.

## Quick Start (Local)
1) Install Docker.
2) Put your bot token in a `.env` file (copy `.env.example`).
3) Build and run:
```bash
docker build -t graphbot .
# pass token at runtime
docker run --rm -e DISCORD_BOT_TOKEN=YOUR_TOKEN graphbot
```

Alternatively (no Docker), install Tesseract and Python libs yourself:
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y tesseract-ocr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DISCORD_BOT_TOKEN=YOUR_TOKEN
python main.py
```

## Deploy on Railway or Render
**Easiest:** use the provided Dockerfile.
- Railway: create a new service from repo → Deploy via Dockerfile → set env `DISCORD_BOT_TOKEN`.
- Render: New → Background Worker → use Docker. Start command uses the Dockerfile's `CMD`.

## Slash Commands
- `/graph input:"y<=2x+1; y>=x-2; x>=-1" x_min:-5 x_max:5 y_min:-5 y_max:5`
- `/graph input:"y = x^2 - 4; x = 2"`
- `/graph input:"x^2 + y^2 = 9"`
- `/imagegraph image:<attach a clear photo/screenshot>`

**Tips:**
- Separate multiple items with `;` (semicolon) or new lines.
- Use `<=, >=, <, >, =`. `^` is allowed (auto-converted to `**`).
- Works with implicit equations too, e.g., `x^2 + y^2 = 9` or `x^2 + y^2 <= 16`.

## What the Bot Understands
- Functions: `y = f(x)`, e.g., `y = sin(x) + x/2`.
- Vertical functions: `x = g(y)`.
- Implicit curves: any `F(x,y) = 0`.
- Inequalities: `F(x,y) <= 0`, or typical forms like `y <= 2x + 1`.
- Multiple constraints are **AND**-combined (intersection shaded).

## OCR Notes
- OCR uses Tesseract. For best results: sharp images, high contrast, minimal handwriting.
- The bot filters detected lines to those likely containing math, then graphs them.

## Troubleshooting
- **No image text found** → Try a clearer crop, higher resolution, or typed text.
- **No plot appears / blank** → The inequality region might be off-screen. Adjust `x_min/x_max/y_min/y_max`.
- **Hosting errors** → Ensure `DISCORD_BOT_TOKEN` is set in your service’s environment.
- **Windows + Local** → Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki and make sure `pytesseract` can find it (add to PATH).

## Safety
- Never share your bot token. Rotate it if exposed.
- Dockerfile keeps the image lean and non-root installs minimal.

## License
MIT
