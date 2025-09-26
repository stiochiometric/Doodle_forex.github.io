# forex_discord_bot_full.py
# Full-featured Discord MT5 trading bot (prefix commands + chart + dynamic per-order buttons)
# WARNING: Test with demo account first.

import os
import json,csv,logging
import io
import asyncio
import math
import traceback
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import discord
from discord.ext import commands, tasks
import mplfinance as mpf
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from ta.momentum import StochasticOscillator
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.gridspec import GridSpec
plt.rcParams['font.family'] = 'Segoe UI Emoji' 
# -------------------- Load config --------------------
CONFIG_PATH = "config.json"
if not os.path.exists(CONFIG_PATH):
    raise SystemExit("Missing config.json - create one (see README).")

with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

TOKEN = cfg.get("Token_2")
PREFIX = cfg.get("prefix", "!")
MT5_CFG = cfg.get("mt5", {})
ALERT_CHANNEL_ID = int(cfg.get("alert_channel_id", 0)) if cfg.get("alert_channel_id") else None
WATCH_LIST = cfg.get("watch_list", ["EURUSDm", "BTCUSDm"])
POLL_INTERVAL_MIN = int(cfg.get("poll_interval_min", 1))
CHART_UPDATE_SECONDS = float(cfg.get("chart_update_seconds", 10.0))
ADMIN_ROLE_IDS = [int(r) for r in cfg.get("admin_role_ids", [])]

# -------------------- Discord bot setup --------------------
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents, help_command=None)

# -------------------- MT5 init --------------------
MT5_LOGIN = int(MT5_CFG.get("login", 0))
MT5_SERVER = MT5_CFG.get("server", "")
MT5_PASSWORD = MT5_CFG.get("password", "")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forex_bot")

# ---------- CSV LOGGING ----------
LOG_FILE = "order_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time", "order_id", "symbol", "type", "volume", "entry_price",
            "sl", "tp", "status", "pnl"
        ])

# --------------------------------------
@bot.event
async def on_ready():

    logger.info(f"Bot ready. Logged in as {bot.user}")
    # watcher_loop.start()
    #synced = await bot.tree.sync()
    #print(f"Synced {len(synced)} slash commands.")
    
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')

#------------------------------------------

def log_order(order_id, symbol, order_type, volume, entry, sl, tp, status, pnl=None):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(), order_id, symbol, order_type,
            volume, entry, sl, tp, status, pnl if pnl is not None else ""
        ])



def connect_mt5():
    if mt5.initialize(login=MT5_LOGIN, server=MT5_SERVER, password=MT5_PASSWORD):
        print("MT5 initialized:", mt5.terminal_info())
        return True
    else:
        print("MT5 initialize() failed:", mt5.last_error())
        return False

if not connect_mt5():
    raise SystemExit("Failed to initialize MT5. Fix credentials or terminal access and restart.")

# -------------------- Helpers / Indicators --------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(df: pd.DataFrame, close_col: str = "close", fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(df["close"], fast)
    slow_ema = ema(df["close"], slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    df = df.copy()
    df["MACD"] = macd_line
    df["MACD_SIGNAL"] = signal_line
    df["MACD_HIST"] = hist
    return df

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_vals = 100 - (100 / (1 + rs))
    return rsi_vals.fillna(50)

TF_MAP = {
    "1m": mt5.TIMEFRAME_M1,
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15,
    "30m": mt5.TIMEFRAME_M30,
    "1h": mt5.TIMEFRAME_H1,
    "4h": mt5.TIMEFRAME_H4,
    "1d": mt5.TIMEFRAME_D1,
    "1w": mt5.TIMEFRAME_W1,
    "1M": mt5.TIMEFRAME_MN1,
}

def fetch_history_mt5(symbol: str, period: str = "60d", timeframe: str = "1h") -> pd.DataFrame:
    if timeframe not in TF_MAP:
        raise ValueError("Unsupported timeframe")
    if period.endswith("d"):
        days = int(period[:-1]); delta = timedelta(days=days)
    elif period.endswith("y"):
        years = int(period[:-1]); delta = timedelta(days=365 * years)
    else:
        raise ValueError("Unsupported period format")
    end_time = datetime.now()
    start_time = end_time - delta
    rates = mt5.copy_rates_range(symbol, TF_MAP[timeframe], start_time, end_time)
    if rates is None or rates.size == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = macd(df, close_col="close")
    df["RSI"] = rsi(df["close"])
    df["EMA20"] = ema(df["close"], 20)
    df["EMA50"] = ema(df["close"], 50)
    return df

# -------------------- Trading / MT5 wrappers --------------------
def ensure_symbol_selected(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Symbol {symbol} not available on server.")
    if not info.visible:
        mt5.symbol_select(symbol, True)
    return info



# ---------- HELPERS ----------

def place_order(symbol, volume, order_type, rr=2.0, buffer=None, sl=None, tp=None, comment="Auto"):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None, "Invalid symbol."

    spread = (tick.ask - tick.bid)
    point = mt5.symbol_info(symbol).point
    if buffer is None:
        buffer = spread  # auto mode = spread

    risk = spread + buffer
    rr_ratio = rr

    if order_type == "BUY":
        entry = tick.ask
        if sl is None:
            sl = entry - (risk / point) * point
        if tp is None:
            tp = entry + (risk * rr_ratio / point) * point
        request_type = mt5.ORDER_TYPE_BUY
        price = entry
    else:  # SELL
        entry = tick.bid
        if sl is None:
            sl = entry + (risk / point) * point
        if tp is None:
            tp = entry - (risk * rr_ratio / point) * point
        request_type = mt5.ORDER_TYPE_SELL
        price = entry

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": request_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 123456,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return None, f"Order failed: {result.comment}"

    log_order(result.order, symbol, order_type, volume, entry, sl, tp, "OPENED")
    return result.order, f"{order_type} order placed at {entry}, SL={sl}, TP={tp}"




def list_open_positions():
    positions = mt5.positions_get()
    if positions is None:
        return []
    out = []
    for p in positions:
        out.append({
            "ticket": int(p.ticket),
            "symbol": p.symbol,
            "type": "BUY" if p.type == 0 else "SELL",
            "volume": p.volume,
            "price_open": p.price_open,
            "sl": p.sl,
            "tp": p.tp,
            "profit": p.profit
        })
    return out





def close_position(ticket: int):
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return {"retcode":"no_position"}
    p = positions[0]
    ensure_symbol_selected(p.symbol)
    tick = mt5.symbol_info_tick(p.symbol)
    if tick is None:
        return {"retcode":"no_tick"}
    if p.type == 0: # BUY -> SELL to close
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": p.symbol,
        "volume": float(p.volume),
        "type": order_type,
        "position": int(p.ticket),
        "price": price,
        "deviation": 50,
        "magic": 123456,
        "comment": "Close by bot",
        "type_filling": mt5.ORDER_FILLING_FOK,
        "type_time": mt5.ORDER_TIME_GTC,
    }
    res = mt5.order_send(request)
    return res

def modify_position_sl_tp(position_ticket: int, sl: float = None, tp: float = None):
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": int(position_ticket),
        "sl": float(sl) if sl is not None else 0.0,
        "tp": float(tp) if tp is not None else 0.0,
    }
    return mt5.order_send(request)

def closed_trades_since(days:int=3650):
    # fetch closed deals from long time ago
    utc_from = datetime.now() - timedelta(days=days)
    deals = mt5.history_deals_get(utc_from, datetime.now())
    if deals is None:
        return []
    out = []
    for d in deals:
        # consider only closed deals with profit/loss
        out.append({
            "ticket": int(d.ticket),
            "order": int(d.order),
            "symbol": d.symbol,
            "type": "BUY" if d.entry == mt5.DEAL_ENTRY_IN or d.type==0 else "SELL",
            "profit": float(d.profit),
            "price": float(d.price),
            "time": datetime.fromtimestamp(d.time)
        })
    return out




def generate_signal(df: pd.DataFrame) -> dict:
    """
    Very simple signal:
      - MACD bullish crossover (MACD crosses above signal) -> BUY
      - MACD bearish crossover -> SELL
      - RSI > 70 -> Overbought (weak SELL signal)
      - RSI < 30 -> Oversold (weak BUY signal)

    We return the latest signal plus confidence and short reasoning.
    """
    last = df.iloc[-3:]  # lookback a couple candles to detect cross
    prev = last.iloc[-2]
    curr = last.iloc[-1]

    signal = "HOLD"
    reason = []
    confidence = 0.0

    # MACD crossover detection
    if (prev["MACD"] < prev["MACD_SIGNAL"]) and (curr["MACD"] > curr["MACD_SIGNAL"]):
        signal = "BUY"
        reason.append("MACD bullish crossover")
        confidence += 0.5
    elif (prev["MACD"] > prev["MACD_SIGNAL"]) and (curr["MACD"] < curr["MACD_SIGNAL"]):
        signal = "SELL"
        reason.append("MACD bearish crossover")
        confidence += 0.5

    # RSI checks
    if curr["RSI"] > 70:
        reason.append("RSI overbought")
        if signal == "BUY":
            # conflicting signals
            confidence -= 0.2
        else:
            signal = "SELL"
        confidence += 0.2
    elif curr["RSI"] < 30:
        reason.append("RSI oversold")
        if signal == "SELL":
            confidence -= 0.2
        else:
            signal = "BUY"
        confidence += 0.2

    # EMA trend filter
    if curr["EMA20"] > curr["EMA50"]:
        reason.append("Short-term uptrend (EMA20 > EMA50)")
        if signal == "BUY":
            confidence += 0.1
        elif signal == "HOLD":
            confidence += 0.05
    else:
        reason.append("Short-term downtrend (EMA20 <= EMA50)")
        if signal == "SELL":
            confidence += 0.1
        elif signal == "HOLD":
            confidence -= 0.05

    confidence = max(0.0, min(1.0, confidence))

    return {
        "signal": signal,
        "confidence": round(confidence, 2),
        "reasons": reason,
        "price": float(curr["close"]),
        "time": datetime.now()
    }


# -------------------- Charting & dynamic view --------------------
def plot_candles_with_indicators(df: pd.DataFrame, symbol: str, theme: str = "dark") -> bytes:
    if df is None or df.empty:
        raise ValueError("Empty data for plotting")
    if "open" not in df.columns:
        raise ValueError("Dataframe missing OHLC")

    # Compute indicators if absent
    if "MACD" not in df.columns:
        df = compute_indicators(df)

    # Highest highs/lows for 5y and 10y
    try:
        #hh5 = df.last("1825D")['close'].max()
        end_time = df.index.max()
        start_time = end_time - pd.Timedelta(days=1825)
        hh5 = df.loc[start_time:end_time, 'close'].max()
        #start_time = end_time - pd.Timedelta(days=1825)
        ll5 = df.loc[start_time:end_time, 'close'].min()

    except Exception:
        hh5 = df['close'].max(); ll5 = df['close'].min()
    hh10 = df['close'].max(); ll10 = df['close'].min()

    # Theme
    if theme == "dark":
        plt.style.use("dark_background")
        upcolor = "#00ff7f"
        downcolor = "#ff4d4d"
    else:
        plt.style.use("default")
        upcolor = "#2ca02c"
        downcolor = "#d62728"

    # required size: 200x800 px -> figsize with dpi 100 => (2,8)
    DPI = 100
    fig = plt.figure(figsize=(10, 8), dpi=DPI, constrained_layout=True)
    gs = fig.add_gridspec(3, hspace=0.05, height_ratios=[4, 1, 1])
    ax_price = fig.add_subplot(gs[0])
    ax_vol = fig.add_subplot(gs[1], sharex=ax_price)
    ax_macd = fig.add_subplot(gs[2], sharex=ax_price)

    # Candlesticks: draw rectangles
    idx = np.arange(len(df.index))
    width = (df.index[1] - df.index[0]).total_seconds() / (24*3600) if len(df.index) > 1 else 0.0008
    width_day = width * 0.8

    for i, (ts, row) in enumerate(df.iterrows()):
        o = row['open']; c = row['close']; h = row['high']; l = row['low']
        color = upcolor if c >= o else downcolor
        ax_price.vlines(ts, l, h, color=color, linewidth=0.5)
        rect_bottom = o if c >= o else c
        rect_height = abs(c - o)
        ax_price.add_patch(plt.Rectangle((mdates_date2num(ts) - width_day/2, rect_bottom),
                                         width_day, rect_height, facecolor=color, edgecolor=color, alpha=0.9))
    # EMA lines if present
    if "EMA20" in df.columns:
        ax_price.plot(df.index, df["EMA20"], linestyle="--", label="EMA20")
    if "EMA50" in df.columns:
        ax_price.plot(df.index, df["EMA50"], linestyle=":", label="EMA50")

    # Boundaries
    ax_price.axhline(hh5, color='green', linestyle='--', linewidth=0.8, label='5Y High')
    ax_price.axhline(ll5, color='red', linestyle='--', linewidth=0.8, label='5Y Low')
    ax_price.axhline(hh10, color='darkgreen', linestyle='-', linewidth=0.8, label='10Y High')
    ax_price.axhline(ll10, color='darkred', linestyle='-', linewidth=0.8, label='10Y Low')
    ax_price.set_title(f"{symbol} — Latest")
    ax_price.legend(loc="upper left", fontsize='x-small')
    ax_price.grid(alpha=0.2)

    # Volume
    if "tick_volume" in df.columns:
        ax_vol.bar(df.index, df['tick_volume'], label='Tick Volume', width=width_day*0.9, color='gray')
    ax_vol.set_title("Volume")
    ax_vol.legend(loc="upper left", fontsize='x-small')
    ax_vol.grid(alpha=0.2)

    # MACD
    ax_macd.bar(df.index, df.get("MACD_HIST", np.zeros(len(df))), label="MACD Hist", width=width_day*0.9)
    ax_macd.plot(df.index, df.get("MACD"), label="MACD", linewidth=0.8)
    ax_macd.plot(df.index, df.get("MACD_SIGNAL"), label="Signal", linewidth=0.8)
    ax_macd.set_title("MACD")
    ax_macd.legend(loc="upper left", fontsize='x-small')
    ax_macd.grid(alpha=0.2)

    #plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=DPI)
    buf.seek(0)
    plt.close(fig)
    return buf.read()

# mdates helper for converting datetimes to Matplotlib numbers
import matplotlib.dates as mdates
def mdates_date2num(dt):
    return mdates.date2num(dt)

# -------------------- Dynamic view class --------------------
MAX_BUTTONS_PER_PAGE = 20  # reserve others for control buttons

class OrdersView(discord.ui.View):
    def __init__(self, symbol: str, message_author_id: int, *, timeout=None, page:int=0):
        super().__init__(timeout=timeout)
        self.symbol = symbol
        self.author_id = message_author_id
        self.page = page
        self.message = None  # will be set later
        self.pagination_total = 1
        self._build_buttons()

    def _build_buttons(self):
        # Clear existing children
        self.clear_items()
        # fetch current positions for symbol
        positions = mt5.positions_get(symbol=self.symbol)
        tick = mt5.symbol_info_tick(self.symbol)
        #close_price = tick.bid if ticket.type == mt5.ORDER_TYPE_BUY else tick.ask
        if positions is None:
            positions = []
        positions = list(positions)
        # pagination
        total = len(positions)
        if total == 0:
            self.pagination_total = 1
            # add a placeholder button
            #b = discord.ui.Button(label=f"{close_price}", style=discord.ButtonStyle.secondary, disabled=True)
            buy_btn = discord.ui.Button(label=f"⬆️ BUY {tick.ask:.5f}", style=discord.ButtonStyle.success)
            sell_btn = discord.ui.Button(label=f"⬇️ SELL {tick.bid:.5f}", style=discord.ButtonStyle.danger)
            self.add_item(buy_btn)
            self.add_item(sell_btn)
            # admin close button
            close_btn = discord.ui.Button(label="Close Chart (Admin)", style=discord.ButtonStyle.danger)
            close_btn.callback = self._close_chart_callback
            self.add_item(close_btn)
            return

        pages = math.ceil(total / MAX_BUTTONS_PER_PAGE)
        self.pagination_total = pages
        start = self.page * MAX_BUTTONS_PER_PAGE
        end = start + MAX_BUTTONS_PER_PAGE
        chunk = positions[start:end]
        for p in chunk:
            tick = mt5.symbol_info_tick(p.symbol)
            close_price = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask
            ticket = int(p.ticket)
            profit = float(p.profit)
            # label show ticket & P/L
            pl_text = f"+{round(profit,2)}" if profit >= 0 else f"{round(profit,2)}"
            lbl = f"{ticket} {pl_text} | {close_price:.5f}"
            btn = discord.ui.Button(label=lbl, style=discord.ButtonStyle.primary)
            # bind callback with ticket capture
            async def _cb(interaction, _ticket=ticket):
                await self._close_ticket_interaction(interaction, _ticket)
            btn.callback = _cb
            self.add_item(btn)
        # pagination controls if needed
        if pages > 1:
            prev_btn = discord.ui.Button(label="Prev", style=discord.ButtonStyle.secondary)
            next_btn = discord.ui.Button(label="Next", style=discord.ButtonStyle.secondary)
            async def _prev_cb(interaction):
                if interaction.user.id != self.author_id and not _is_admin_interaction(interaction):
                    await interaction.response.send_message("Only the chart owner or admin may paginate.", ephemeral=True)
                    return
                self.page = max(0, self.page - 1)
                self._build_buttons()
                # edit message with same image but new view
                await self._refresh_message()
            async def _next_cb(interaction):
                if interaction.user.id != self.author_id and not _is_admin_interaction(interaction):
                    await interaction.response.send_message("Only the chart owner or admin may paginate.", ephemeral=True)
                    return
                self.page = min(self.pagination_total-1, self.page + 1)
                self._build_buttons()
                await self._refresh_message()
            prev_btn.callback = _prev_cb
            next_btn.callback = _next_cb
            self.add_item(prev_btn)
            self.add_item(next_btn)
        # Add action buttons row: Buy, AutoBuy, Sell, AutoSell, Close Chart
        buy_btn = discord.ui.Button(label="⬆️ BUY", style=discord.ButtonStyle.success)
        autobuy_btn = discord.ui.Button(label="🔼 AUTO BUY", style=discord.ButtonStyle.success)
        sell_btn = discord.ui.Button(label="⬇️ SELL", style=discord.ButtonStyle.danger)
        autosell_btn = discord.ui.Button(label="🔽 AUTO SELL", style=discord.ButtonStyle.danger)
        close_btn = discord.ui.Button(label="❎Close Chart (Admin)", style=discord.ButtonStyle.danger)
        buy_btn.callback = self._buy_callback
        autobuy_btn.callback = self._autobuy_callback
        sell_btn.callback = self._sell_callback
        autosell_btn.callback = self._autosell_callback
        close_btn.callback = self._close_chart_callback
        self.add_item(buy_btn)
        self.add_item(autobuy_btn)
        self.add_item(sell_btn)
        self.add_item(autosell_btn)
        self.add_item(close_btn)

    async def _refresh_message(self):
        # re-generate chart and edit the message (owner message stored in self.message)
        if not self.message:
            return
        # fetch fresh candles
        try:
            df = fetch_history_mt5(self.symbol, period="7d", timeframe="15m")
            if df is None:
                return
            df = compute_indicators(df)
            img_bytes = _safe_plot_bytes(df, self.symbol)
            file = discord.File(io.BytesIO(img_bytes), filename=f"{self.symbol}_chart.png")
            await self.message.edit(attachments=[file], view=self)
        except Exception as e:
            print("Error refreshing message view:", e)

    async def _close_ticket_interaction(self, interaction: discord.Interaction, ticket: int):
        await interaction.response.defer()
        try:
            res = close_position(ticket)
            await interaction.followup.send(f"Close request result for {ticket}: {res}", ephemeral=True)
            # rebuild buttons
            self._build_buttons()
            await self._refresh_message()
        except Exception as e:
            await interaction.followup.send("Error closing position: " + str(e), ephemeral=True)

    async def _buy_callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        # place manual buy at market with no sl/tp
        symbol = self.symbol
        # default lot
        lot = 0.01
        try:
            res = place_order(symbol, lot, "BUY", sl=None, tp=None, comment="⬆️ BUY")
            await interaction.followup.send(f"Buy result: {res}", ephemeral=True)
        except Exception as e:
            await interaction.followup.send("Buy error: " + str(e), ephemeral=True)

    async def _autobuy_callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        symbol = self.symbol
        lot = 0.01
        rr = 2.0
        try:
            # compute automatic sl/tp with heuristic: risk distance = 0.5% of price
            tick = mt5.symbol_info_tick(symbol)
            price = tick.ask
            risk_dist = price * 0.005
            sl = price - risk_dist
            tp = price + rr * risk_dist
            res = place_order(symbol, lot, "BUY", sl=sl, tp=tp, comment="🔼 AUTO BUY")
            await interaction.followup.send(f"AutoBuy result: {res}", ephemeral=True)
        except Exception as e:
            await interaction.followup.send("AutoBuy error: " + str(e), ephemeral=True)

    async def _sell_callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        symbol = self.symbol
        lot = 0.01
        try:
            res = place_order(symbol, lot, "SELL", sl=None, tp=None, comment="⬇️ SELL")
            await interaction.followup.send(f"Sell result: {res}", ephemeral=True)
        except Exception as e:
            await interaction.followup.send("Sell error: " + str(e), ephemeral=True)

    async def _autosell_callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        symbol = self.symbol
        lot = 0.01
        rr = 2.0
        try:
            tick = mt5.symbol_info_tick(symbol)
            price = tick.bid
            risk_dist = price * 0.005
            sl = price + risk_dist
            tp = price - rr * risk_dist
            res = place_order(symbol, lot, "SELL", sl=sl, tp=tp, comment="🔽 AUTO SELL")
            await interaction.followup.send(f"AutoSell result: {res}", ephemeral=True)
        except Exception as e:
            await interaction.followup.send("AutoSell error: " + str(e), ephemeral=True)

    async def _close_chart_callback(self, interaction: discord.Interaction):
        # only admins or owner allowed
        if interaction.user.id != self.author_id and not _is_admin_interaction(interaction):
            await interaction.response.send_message("Only the chart owner or admin can close the chart.", ephemeral=True)
            return
        await interaction.response.defer()
        try:
            # delete stored message and stop its update loop by setting a global flag
            if self.message:
                try:
                    await self.message.delete()
                except Exception:
                    pass
            AutoChartManager.stop_chart(self.message.id if self.message else None)
            await interaction.followup.send("Chart closed by admin.", ephemeral=True)
        except Exception as e:
            await interaction.followup.send("Failed to close chart: " + str(e), ephemeral=True)

# helper to check admin permission
def _is_admin_interaction(interaction: discord.Interaction):
    if interaction.user.guild_permissions.administrator:
        return True
    # check roles
    for rid in ADMIN_ROLE_IDS:
        if rid in [r.id for r in interaction.user.roles]:
            return True
    return False

def _safe_plot_bytes(df, symbol, theme="dark"):
    try:
        return plot_candles_with_indicators(df, symbol, theme=theme)
    except Exception:
        # if plotting fails, create a simple fallback image
        plt.figure(figsize=(10,8), dpi=100)
        plt.text(0.5,0.5,"Plot error",ha="center")
        buf = io.BytesIO()
        plt.savefig(buf, format="png"); buf.seek(0); plt.close()
        return buf.read()


BASELINE_FILE = "baselines.json"

def load_baselines():
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, "r") as f:
            return json.load(f)
    return {"heartbeat": {}, "volume": {}}

def save_baselines(data):
    with open(BASELINE_FILE, "w") as f:
        json.dump(data, f, indent=2)

# Load at startup
BASELINES = load_baselines()


# ---------- GLOBAL STATE ----------
HEARTBEAT_BASELINES = {}   # per symbol baseline (bpm)
HEARTBEAT_GLOBAL_MINS = {} # per symbol global minima (bpm)
VOLUME_BASELINES = {}      # per symbol baseline (avg tick_volume)
# thresholds (tweakable)
HEARTBEAT_SPIKE_FACTOR = 1.5   # bpm > baseline * factor -> spike
HEARTBEAT_DROP_FACTOR  = 0.5   # bpm < baseline * factor -> baseline update downward
VOLUME_SPIKE_FACTOR    = 1.5
VOLUME_DROP_FACTOR     = 0.5

# ---------- HELPERS ----------
def _parse_timeframe_minutes(tf: str) -> int:
    tf = str(tf).lower().strip()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    return 1

def safe_mpf_plot_to_buffer(df_plot, addplot=None, figsize=(12,8), style='charles', panel_ratios=(3,1)):
    """
    Use mplfinance.plot(..., returnfig=True) and return png bytes.
    """
    fig, axes = mpf.plot(df_plot,
                         type='candle',
                         style=style,
                         addplot=addplot,
                         volume=False,
                         figsize=figsize,
                         returnfig=True,
                         panel_ratios=panel_ratios,
                         tight_layout=True)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


def compute_heartbeat(symbol: str, tick_lookback_seconds: int = 10) -> float:
    """
    Count raw tick arrivals as heartbeat (BPM).
    Every tick = one beat.
    """
    from_time = datetime.now() - timedelta(seconds=tick_lookback_seconds)
    ticks = mt5.copy_ticks_from(symbol, from_time, 100000, mt5.COPY_TICKS_ALL)

    if ticks is None or len(ticks) == 0:
        return 0.0

    # number of ticks in the window
    n_ticks = len(ticks)

    # convert to beats per minute
    bpm = n_ticks * (60.0 / tick_lookback_seconds)
    return bpm


# -------------------- Auto chart manager (global to track running charts) --------------------

# Add/replace these imports if not already present near top of file
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# -------------------- Fast live chart (last N candles) --------------------
def plot_live_bytes(df: pd.DataFrame, symbol: str, theme: str = "dark", bars: int = 50) -> bytes:
    """
    Fast, lightweight chart — last `bars` candles, no indicators.
    Produces image sized 800x200 px (figsize=(8,2), dpi=100).
    Shows bull/bear emoji above last close.
    """
    if df is None or df.empty:
        raise ValueError("Empty data for live plot")
    # use only last `bars` candles
    df_plot = df.tail(bars).copy()
    if df_plot.empty:
        raise ValueError("Not enough data for live plot")

    # theme colors
    if theme == "dark":
        plt.style.use("dark_background")
        upcolor = "#00ff7f"
        downcolor = "#ff4d4d"
        bg = "#000000"
    else:
        plt.style.use("default")
        upcolor = "#2ca02c"
        downcolor = "#d62728"
        bg = "#ffffff"

    DPI = 100
    fig, ax = plt.subplots(figsize=(8, 2), dpi=DPI)
    ax.set_facecolor(bg)

    # x values as matplotlib datenums
    x = mdates.date2num(pd.to_datetime(df_plot.index).to_pydatetime())

    # width of each candle in days
    if len(x) > 1:
        width = (x[1] - x[0]) * 0.7
    else:
        width = 0.0008

    # draw candles
    for i, (ts, row) in enumerate(df_plot.iterrows()):
        o = float(row["open"]); c = float(row["close"]); h = float(row["high"]); l = float(row["low"])
        color = upcolor if c >= o else downcolor
        xi = x[i]
        # wick
        ax.vlines(mdates.num2date(xi), l, h, color=color, linewidth=0.8)
        # body
        bottom = o if c >= o else c
        height = abs(c - o)
        rect = Rectangle((xi - width/2, bottom), width, height if height > 0 else (row["high"]-row["low"]) * 0.001,
                         facecolor=color, edgecolor=color, linewidth=0.5, alpha=1.0)
        ax.add_patch(rect)

    # bull/bear marker on last candle
    last = df_plot.iloc[-1]
    last_x = x[-1]
    last_close = float(last["close"])
    last_open = float(last["open"])
    marker = "🐂" if last_close >= last_open else "🐻"
    # place marker slightly above/below last close
    y_offset = (df_plot["high"].max() - df_plot["low"].min()) * 0.02
    if marker == "🐂":
        ax.text(mdates.num2date(last_x), last_close + y_offset, marker, fontsize=14, ha="center", va="bottom")
    else:
        ax.text(mdates.num2date(last_x), last_close - y_offset, marker, fontsize=14, ha="center", va="top")

    # formatting
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlim(mdates.num2date(x[0]) - timedelta(seconds=1), mdates.num2date(x[-1]) + timedelta(seconds=1))
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=DPI)
    buf.seek(0)
    plt.close(fig)
    return buf.read()



# -------------------- Static chart (indicators) --------------------
def plot_static_indicators_bytes(df: pd.DataFrame, symbol: str, theme: str = "dark") -> bytes:
    """
    Big-picture static chart with indicators (EMA20/50, MACD, Volume).
    Sized to ~800x800 px for readability.
    """
    if df is None or df.empty:
        raise ValueError("Empty data for static plot")

    # ensure indicators are present
    df = compute_indicators(df)

    DPI = 100
    fig = plt.figure(figsize=(8, 8), dpi=DPI)
    gs = fig.add_gridspec(3, hspace=0.05, height_ratios=[4, 1, 1])

    ax_price = fig.add_subplot(gs[0])
    ax_vol = fig.add_subplot(gs[1], sharex=ax_price)
    ax_macd = fig.add_subplot(gs[2], sharex=ax_price)

    # theme colors
    if theme == "dark":
        plt.style.use("dark_background")
        price_color = "cyan"
        vol_color = "lime"
    else:
        plt.style.use("default")
        price_color = "blue"
        vol_color = "green"

    # price + EMAs
    ax_price.plot(df.index, df["close"], label="Close", color=price_color, linewidth=1.2)
    if "EMA20" in df.columns: ax_price.plot(df.index, df["EMA20"], linestyle="--", label="EMA20")
    if "EMA50" in df.columns: ax_price.plot(df.index, df["EMA50"], linestyle=":", label="EMA50")
    ax_price.set_title(f"{symbol} — Static (Indicators)")
    ax_price.legend(loc="upper left", fontsize='small')
    ax_price.grid(alpha=0.2)

    # volume
    if "tick_volume" in df.columns:
        ax_vol.bar(df.index, df["tick_volume"], label="Tick Volume", color=vol_color)
    ax_vol.set_title("Volume")
    ax_vol.legend(loc="upper left", fontsize='x-small')
    ax_vol.grid(alpha=0.2)

    # MACD
    if "MACD" in df.columns:
        ax_macd.bar(df.index, df.get("MACD_HIST", np.zeros(len(df))), label="MACD Hist")
        ax_macd.plot(df.index, df.get("MACD"), label="MACD", linewidth=0.8)
        ax_macd.plot(df.index, df.get("MACD_SIGNAL"), label="Signal", linewidth=0.8)
        ax_macd.set_title("MACD")
        ax_macd.legend(loc="upper left", fontsize='x-small')
        ax_macd.grid(alpha=0.2)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=DPI)
    buf.seek(0)
    plt.close(fig)
    return buf.read()

# -------------------- Update OrdersView._refresh_message to use live plot --------------------
# Replace OrdersView._refresh_message with this implementation
async def ordersview_refresh_message(self):
    # re-generate live chart and edit the message (owner message stored in self.message)
    if not self.message:
        return
    try:
        # fetch fresh short-window candles (fast)
        df = fetch_history_mt5(self.symbol, period="1d", timeframe="15m")
        if df is None or df.empty:
            return
        # use only last 50 bars for live chart
        img_bytes = plot_live_bytes(df, self.symbol, theme="dark", bars=50)
        file = discord.File(io.BytesIO(img_bytes), filename=f"{self.symbol}_live.png")
        await self.message.edit(attachments=[file], view=self)
    except Exception as e:
        print("Error refreshing message view:", e)

# patch OrdersView class method (if OrdersView defined earlier)
if "OrdersView" in globals():
    OrdersView._refresh_message = ordersview_refresh_message

# -------------------- Updated AutoChartManager.start_chart --------------------
class AutoChartManager:
    running = {}  # message_id -> task info

    @classmethod
    def stop_chart(cls, message_id):
        if message_id in cls.running:
            try:
                cls.running[message_id]["task"].cancel()
            except Exception:
                pass
            del cls.running[message_id]

    @classmethod
    async def start_chart(cls, channel: discord.TextChannel, symbol: str, author_id: int, theme="dark"):
        """
        Sends:
          1) Static chart with indicators (sent once, daily candles for long-term)
          2) Live chart message (attached view) which updates every CHART_UPDATE_SECONDS
        """
        # 1) static (big-picture) — use daily timeframe for long-range history (10y if available)
        try:
            df_static = fetch_history_mt5(symbol, period="3650d", timeframe="1d")
            if df_static is not None and not df_static.empty:
                # compute indicators and produce static image
                img_static = plot_static_indicators_bytes(df_static, symbol, theme=theme)
                file_static = discord.File(io.BytesIO(img_static), filename=f"{symbol}_static.png")
                await channel.send(content=f"📊 **{symbol} Static Chart (with indicators)**", file=file_static)
        except Exception as e:
            # still continue — static chart is optional
            print("Static chart generation failed:", e)

        # 2) live chart (fast, last ~50 bars)
        df_live = fetch_history_mt5(symbol, period="1d", timeframe="15m")
        if df_live is None or df_live.empty:
            # fallback to a shorter timeframe if necessary
            df_live = fetch_history_mt5(symbol, period="12h", timeframe="5m")
            if df_live is None or df_live.empty:
                await channel.send("Not enough data to start live chart.")
                return

        # initial live image (zoomed-in)
        try:
            img_live = plot_live_bytes(df_live, symbol, theme=theme, bars=50)
        except Exception:
            # fallback: use entire df_live
            img_live = plot_live_bytes(df_live, symbol, theme=theme, bars=min(50, len(df_live)))

        file_live = discord.File(io.BytesIO(img_live), filename=f"{symbol}_live.png")
        view = OrdersView(symbol, author_id)  # reuses your OrdersView for buttons
        sent = await channel.send(file=file_live, view=view)
        view.message = sent

        # updater loop edits ONLY the live message (fast)
        async def updater():
            try:
                while True:
                    await asyncio.sleep(CHART_UPDATE_SECONDS)
                    # fetch recent small window
                    try:
                        df2 = fetch_history_mt5(symbol, period="1d", timeframe="15m")
                        if df2 is None or df2.empty:
                            # try shorter timeframe
                            df2 = fetch_history_mt5(symbol, period="12h", timeframe="5m")
                        if df2 is None or df2.empty:
                            continue
                        # build small window and plot
                        # prefer last 50 bars
                        try:
                            img_bytes2 = plot_live_bytes(df2, symbol, theme=theme, bars=50)
                        except Exception:
                            img_bytes2 = plot_live_bytes(df2, symbol, theme=theme, bars=min(50, len(df2)))
                        file2 = discord.File(io.BytesIO(img_bytes2), filename=f"{symbol}_live.png")
                        # rebuild buttons and edit message
                        view._build_buttons()
                        try:
                            await sent.edit(attachments=[file2], view=view)
                        except Exception:
                            # message possibly deleted -> stop loop
                            break
                    except Exception:
                        print("Live chart cycle error:", traceback.format_exc())
                        continue
            except asyncio.CancelledError:
                pass
            except Exception:
                print("Auto chart updater exception:", traceback.format_exc())

        task = asyncio.create_task(updater())
        cls.running[sent.id] = {"task": task, "symbol": symbol, "channel_id": channel.id, "view": view, "author_id": author_id}
        return sent

# -------------------- Strategic Actions-------------------- 
def compute_heartbeat_precise(df: pd.DataFrame):
    closes = df["Close"].values.astype(float)
    n = len(closes)
    if n < 8:
        raise ValueError("Need at least 8 samples")

    detrended = closes - np.mean(closes)
    fft_vals = np.fft.rfft(detrended)
    freqs = np.fft.rfftfreq(n, d=1)

    amps = np.abs(fft_vals)
    amps[0] = 0

    threshold = np.mean(amps) * 1.2
    mask = amps > threshold
    filtered_fft = fft_vals * mask
    osc = np.fft.irfft(filtered_fft, n=n)

    osc_std = np.std(osc) if np.std(osc) > 0 else 1.0
    price_std = np.std(closes) if np.std(closes) > 0 else 1.0
    osc_scaled = (osc / osc_std) * price_std * 0.9

    dominant_freq = freqs[np.argmax(amps)]
    return pd.Series(osc_scaled, index=df.index), dominant_freq


def compute_heartbeat_cumulative(df: pd.DataFrame, keep_freqs: int = 8):
    closes = df["Close"].values.astype(float)
    n = len(closes)
    if n < 8:
        raise ValueError("Need at least 8 samples")

    detrended = closes - np.mean(closes)
    fft_vals = np.fft.rfft(detrended)
    freqs = np.fft.rfftfreq(n, d=1)

    amps = np.abs(fft_vals)
    amps[0] = 0

    idx_sorted = np.argsort(amps)[::-1]
    top_idx = idx_sorted[:keep_freqs]
    mask = np.zeros_like(fft_vals, dtype=bool)
    mask[top_idx] = True
    filtered_fft = fft_vals * mask
    osc = np.fft.irfft(filtered_fft, n=n)

    osc_std = np.std(osc) if np.std(osc) > 0 else 1.0
    price_std = np.std(closes) if np.std(closes) > 0 else 1.0
    osc_scaled = (osc / osc_std) * price_std * 0.9

    dominant_freq = freqs[top_idx[0]]
    return pd.Series(osc_scaled, index=df.index), dominant_freq


# -------------------- Strategic Functions -------------------- 


def detect_pattern1(df, avg_body_period=12):
    """
    Detects Three Black Crows / Three White Soldiers.
    Returns ("BUY"/"SELL"/None, info_string)
    """
    o = df["open"]
    c = df["close"]
    h = df["high"]
    l = df["low"]

    def midpoint(i): return (h.iloc[i] + l.iloc[i]) / 2
    def avg_body(i): return abs(o.iloc[i:i+avg_body_period] - c.iloc[i:i+avg_body_period]).mean()

    # Check last 3 candles
    if len(df) < 5:
        return None, "Not enough candles"

    # Indexing last 4 candles
    # [-4], [-3], [-2], [-1] are last candles (like bars 3,2,1 in MQL5 code)
    if (o.iloc[-4] - c.iloc[-4] > avg_body(-2)) and \
       (o.iloc[-3] - c.iloc[-3] > avg_body(-2)) and \
       (o.iloc[-2] - c.iloc[-2] > avg_body(-2)) and \
       (midpoint(-3) < midpoint(-4)) and (midpoint(-2) < midpoint(-3)):
        return "SELL", "3 Black Crows detected"

    if (c.iloc[-4] - o.iloc[-4] > avg_body(-2)) and \
       (c.iloc[-3] - o.iloc[-3] > avg_body(-2)) and \
       (c.iloc[-2] - o.iloc[-2] > avg_body(-2)) and \
       (midpoint(-3) > midpoint(-4)) and (midpoint(-2) > midpoint(-3)):
        return "BUY", "3 White Soldiers detected"

    return None, "No pattern"


def confirm_signal(df, signal, cci_period=37):
    """
    Confirms a signal using CCI
    """
    cci = ta.trend.cci(df["high"], df["low"], df["close"], window=cci_period)
    latest = cci.iloc[-1]

    if signal == "BUY" and latest < -50:
        return True, f"Confirmed: CCI={latest:.2f} (< -50)"
    if signal == "SELL" and latest > 50:
        return True, f"Confirmed: CCI={latest:.2f} (> 50)"
    return False, f"Not confirmed: CCI={latest:.2f}"



# === Parameters ===
AVER_BODY_PERIOD = 12
MFI_PERIOD = 37

# === Helper Functions ===
def average_body(candles, idx):
    return abs(candles['open'].iloc[idx] - candles['close'].iloc[idx:idx+AVER_BODY_PERIOD]).mean()

def detect_pattern2(candles):
    signal, info, direction = 0, "No candlestick pattern found", ""

    # 3 Black Crows
    if ((candles['open'].iloc[-4] - candles['close'].iloc[-4] > average_body(candles, -2)) and
        (candles['open'].iloc[-3] - candles['close'].iloc[-3] > average_body(candles, -2)) and
        (candles['open'].iloc[-2] - candles['close'].iloc[-2] > average_body(candles, -2)) and
        ((candles['high'].iloc[-3]+candles['low'].iloc[-3])/2 < (candles['high'].iloc[-4]+candles['low'].iloc[-4])/2) and
        ((candles['high'].iloc[-2]+candles['low'].iloc[-2])/2 < (candles['high'].iloc[-3]+candles['low'].iloc[-3])/2)):
        signal, info, direction = -1, "3 Black Crows detected", "Sell"

    # 3 White Soldiers
    elif ((candles['close'].iloc[-4] - candles['open'].iloc[-4] > average_body(candles, -2)) and
          (candles['close'].iloc[-3] - candles['open'].iloc[-3] > average_body(candles, -2)) and
          (candles['close'].iloc[-2] - candles['open'].iloc[-2] > average_body(candles, -2)) and
          ((candles['high'].iloc[-3]+candles['low'].iloc[-3])/2 > (candles['high'].iloc[-4]+candles['low'].iloc[-4])/2) and
          ((candles['high'].iloc[-2]+candles['low'].iloc[-2])/2 > (candles['high'].iloc[-3]+candles['low'].iloc[-3])/2)):
        signal, info, direction = 1, "3 White Soldiers detected", "Buy"

    return signal, info, direction

'''def calc_mfi(data, period=14):
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    money_flow = typical_price * data['tick_volume']

    pos_flow, neg_flow = [0], [0]
    for i in range(1, len(data)):
        if typical_price[i] > typical_price[i-1]:
            pos_flow.append(money_flow[i])
            neg_flow.append(0)
        else:
            pos_flow.append(0)
            neg_flow.append(money_flow[i])

    pos_mf = pd.Series(pos_flow).rolling(period).sum()
    neg_mf = pd.Series(neg_flow).rolling(period).sum()

    mfi = 100 - (100 / (1 + (pos_mf / neg_mf)))
    return mfi'''



# === Config ===
STOCH_PERIOD = 14
STOCH_SMOOTH = 3

# === Pattern Detection ===
def detect_pattern3(df: pd.DataFrame):
    """
    Detect 3 White Soldiers (bullish) or 3 Black Crows (bearish).
    Returns (signal, info, direction)
    signal: 0 = no pattern, 1 = bullish, -1 = bearish
    """
    if len(df) < 4:
        return 0, "Not enough candles", None

    last3 = df.iloc[-4:-1]  # last 3 closed candles
    bodies = last3['close'] - last3['open']
    avg_body = df['close'] - df['open']
    avg_body = avg_body.abs().rolling(10).mean().iloc[-1]

    # --- 3 White Soldiers (Bullish) ---
    if all(b > 0 for b in bodies) and all(b.abs() > avg_body * 0.5 for b in bodies):
        if all(last3['Close'].values[i] > last3['Open'].values[i] for i in range(3)):
            return 1, "Three White Soldiers detected", "BUY"

    # --- 3 Black Crows (Bearish) ---
    if all(b < 0 for b in bodies) and all(b.abs() > avg_body * 0.5 for b in bodies):
        if all(last3['Close'].values[i] < last3['Open'].values[i] for i in range(3)):
            return -1, "Three Black Crows detected", "SELL"

    return 0, "No clear pattern", None

# === Stochastic Confirmation ===
def confirm_signal(df: pd.DataFrame, signal: int):
    """
    Confirm signal with Stochastic Oscillator
    """
    stoch = StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=STOCH_PERIOD,
        smooth_window=STOCH_SMOOTH
    )
    stoch_k = stoch.stoch().iloc[-1]

    if signal == 1 and stoch_k < 30:
        return True, f"Stoch {stoch_k:.2f} (<30) oversold → BUY confirmed"
    elif signal == -1 and stoch_k > 70:
        return True, f"Stoch {stoch_k:.2f} (>70) overbought → SELL confirmed"
    else:
        return False, f"Stoch {stoch_k:.2f} → Not confirming"



# --- RSI function ---


# --- Pattern detection (3 Black Crows / 3 White Soldiers) ---
def detect_pattern4(df, avg_body_period=12):
    df["body"] = (df["close"] - df["open"]).abs()
    avg_body = df["body"].rolling(avg_body_period).mean()

    # last 3 candles
    o = df["open"]
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # midpoint
    mid = (h + l) / 2

    # 3 Black Crows
    cond_crows = (
        (o.shift(3) - c.shift(3) > avg_body) &
        (o.shift(2) - c.shift(2) > avg_body) &
        (o.shift(1) - c.shift(1) > avg_body) &
        (mid.shift(2) < mid.shift(3)) &
        (mid.shift(1) < mid.shift(2))
    )

    if cond_crows.iloc[-1]:
        return "SELL", "3 Black Crows detected"

    # 3 White Soldiers
    cond_soldiers = (
        (c.shift(3) - o.shift(3) > avg_body) &
        (c.shift(2) - o.shift(2) > avg_body) &
        (c.shift(1) - o.shift(1) > avg_body) &
        (mid.shift(2) > mid.shift(3)) &
        (mid.shift(1) > mid.shift(2))
    )

    if cond_soldiers.iloc[-1]:
        return "BUY", "3 White Soldiers detected"

    return None, "No clear pattern"

# --- Confirmation with RSI ---
def confirm_with_rsi(df, signal, rsi_period=37):
    rsi = calc_rsi(df["Close"], rsi_period)
    last_rsi = rsi.iloc[-1]

    if signal == "BUY" and last_rsi < 40:
        return True, f"RSI {last_rsi:.2f} (<40) ✅ Confirmed"
    elif signal == "SELL" and last_rsi > 60:
        return True, f"RSI {last_rsi:.2f} (>60) ✅ Confirmed"
    else:
        return False, f"RSI {last_rsi:.2f} ❌ Not Confirmed"

'''def calc_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_mfi(data, period=14):
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    money_flow = typical_price * data['tick_volume']

    # Shifted comparison
    price_change = typical_price.diff()

    pos_flow = np.where(price_change > 0, money_flow, 0)
    neg_flow = np.where(price_change < 0, money_flow, 0)

    pos_mf = pd.Series(pos_flow).rolling(period).sum()
    neg_mf = pd.Series(neg_flow).rolling(period).sum()

    mfi = 100 - (100 / (1 + (pos_mf / neg_mf)))
    return mfi

def calc_stoc(df, k_period=14, d_period=3):
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d  # you can just use stoch_k.iloc[-1]

def calc_cci(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.fabs(x - x.mean()).mean())
    cci = (tp - sma) / (0.015 * mad)
    return cci
'''

# ----------------- helpers -----------------
def _standardize_and_index(df):
    """Return (df_proc, df_plot) where:
       - df_proc: lowercase columns ('open','high','low','close','volume') and DatetimeIndex
       - df_plot: same but with mplfinance-friendly capitalized columns ('Open','High',...)
    """
    df2 = df.copy()

    # unify column names to lowercase
    df2.columns = [c.lower() for c in df2.columns]

    # if "time" column exists, use it as index
    if "time" in df2.columns:
        df2["time"] = pd.to_datetime(df2["time"])
        df2 = df2.set_index("time")
    else:
        # try converting index to datetime
        try:
            df2.index = pd.to_datetime(df2.index)
        except Exception:
            pass

    # ensure required columns exist (try to map common alternatives)
    # tick_volume -> volume, volume already ok
    if "tick_volume" in df2.columns and "volume" not in df2.columns:
        df2["volume"] = df2["tick_volume"]

    # Lowercase canonical names
    for req in ("open", "high", "low", "close"):
        if req not in df2.columns:
            raise KeyError(f"DataFrame missing required column: {req} (case-insensitive). Columns: {list(df.columns)}")

    # sort by time ascending
    df2 = df2.sort_index()

    # create a mplfinance-friendly copy
    df_plot = df2.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })

    return df2, df_plot

# ----------------- indicators -----------------
def calc_rsi(close_series, period=14):
    close = close_series.astype(float)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Wilder's smoothing via ewm (alpha=1/period)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi.name = "RSI"
    return rsi

def calc_macd(close_series, fast=12, slow=26, signal=9):
    fast_ema = close_series.ewm(span=fast, adjust=False).mean()
    slow_ema = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig_line
    macd_line.name = "MACD"
    sig_line.name = "MACD_SIGNAL"
    hist.name = "MACD_HIST"
    return macd_line, sig_line, hist

def calc_stoc(df, k_period=14, d_period=3):
    low_min = df['low'].rolling(window=k_period, min_periods=1).min()
    high_max = df['high'].rolling(window=k_period, min_periods=1).max()
    denom = (high_max - low_min).replace(0, np.nan)
    stoch_k = 100 * ((df['close'] - low_min) / denom)
    stoch_d = stoch_k.rolling(window=d_period, min_periods=1).mean()
    stoch_k.name = "%K"
    stoch_d.name = "%D"
    return stoch_k.fillna(0), stoch_d.fillna(0)

def calc_cci(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    sma = tp.rolling(window=period, min_periods=1).mean()
    mad = tp.rolling(window=period, min_periods=1).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True)
    denom = 0.015 * mad.replace(0, np.nan)
    cci = (tp - sma) / denom
    cci.name = "CCI"
    return cci.fillna(0)

def calc_mfi(df, period=14):
    # use 'volume' if present; fallback to 'tick_volume'
    vol_col = "volume" if "volume" in df.columns else ("tick_volume" if "tick_volume" in df.columns else None)
    if vol_col is None:
        # no volume -> return NaN series
        return pd.Series(index=df.index, data=np.nan, name="MFI")

    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    money_flow = typical_price * df[vol_col]
    price_change = typical_price.diff()

    pos_flow = money_flow.where(price_change > 0, 0.0)
    neg_flow = money_flow.where(price_change < 0, 0.0)

    pos_mf = pos_flow.rolling(window=period, min_periods=1).sum()
    neg_mf = neg_flow.rolling(window=period, min_periods=1).sum()

    # avoid division by zero
    ratio = pos_mf / neg_mf.replace(0, np.nan)
    mfi = 100 - (100 / (1 + ratio))
    mfi.name = "MFI"
    return mfi.fillna(50)  # neutral when insufficient data

# ----------------- signals -----------------
def generate_signals3_from_series(df_proc):
    signals = {}
    if len(df_proc) < 2:
        return {"RSI":"Insufficient","Stoch":"Insufficient","CCI":"Insufficient","MFI":"Insufficient"}

    rsi = calc_rsi(df_proc['close']).iloc[-1]
    stoch_k, stoch_d = calc_stoc(df_proc)
    stoch_val = stoch_k.iloc[-1]
    cci = calc_cci(df_proc).iloc[-1]
    mfi = calc_mfi(df_proc).iloc[-1]

    signals["RSI"]   = "Buy ✅" if rsi < 30 else "Sell ❌" if rsi > 70 else "Neutral ⚪"
    signals["Stoch"] = "Buy ✅" if stoch_val < 20 else "Sell ❌" if stoch_val > 80 else "Neutral ⚪"
    signals["CCI"]   = "Buy ✅" if cci < -100 else "Sell ❌" if cci > 100 else "Neutral ⚪"
    signals["MFI"]   = "Buy ✅" if mfi < 20 else "Sell ❌" if mfi > 80 else "Neutral ⚪"
    return signals




def plot_detailed_signals2(df, signals, symbol, timeframe, final_signal):
    """
    One big figure:
    Top = Candlesticks
    Below = RSI, Stoch, CCI, MFI
    """
    if df is None or df.empty:
        raise ValueError("Empty dataframe passed")

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df = df.set_index(pd.to_datetime(df["time"]))
        else:
            raise ValueError("No datetime index found in df")

    # Normalize column names
    df = df.rename(columns={c: c.lower() for c in df.columns})

    # ----------------- Figure layout -----------------
    fig, axs = plt.subplots(5, 1, figsize=(12, 8), sharex=True, dpi=100,
                             gridspec_kw={"height_ratios":[3,1,1,1,1]})
    ax_price, ax_rsi, ax_stoch, ax_cci, ax_mfi = axs

    # ----------------- Candlesticks -----------------
    df = df.tail(150)
    idx = np.arange(len(df.index))
    '''if len(df.index) > 1:
        width = (df.index[1] - df.index[0]).total_seconds() / (24*3600)
    else:
        width = 0.0008'''
    width_day = 0.03#width * 0.8

    upcolor, downcolor = "#000000", "#ff0000"
    for ts, row in df.iterrows():
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = upcolor if c >= o else downcolor
        ax_price.vlines(ts, l, h, color=color, linewidth=1)
        rect_bottom = min(o, c)
        rect_height = abs(c - o)
        ax_price.add_patch(
            plt.Rectangle(
                (mdates.date2num(ts) - width_day/2, rect_bottom),
                width_day,
                rect_height if rect_height > 0 else 0.001,  # tiny body if doji
                facecolor=color, edgecolor=color, alpha=0.9
            )
        )

    ax_price.set_title(f"{symbol} ({timeframe}) → {final_signal}")
    #ax_price.grid(alpha=0.3)

    # ----------------- RSI -----------------
    rsi = calc_rsi(df["close"])
    ax_rsi.plot(df.index, rsi, color="blue", label="RSI")
    ax_rsi.axhline(30, linestyle="--", color="green", alpha=0.5)
    ax_rsi.axhline(70, linestyle="--", color="red", alpha=0.5)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.legend(loc="upper left", fontsize="x-small")
    # highlight only last point
    if "Buy" in signals["RSI"]:
        ax_rsi.scatter(df.index[-1], rsi.iloc[-1], color="green", marker="^", s=50)
    elif "Sell" in signals["RSI"]:
        ax_rsi.scatter(df.index[-1], rsi.iloc[-1], color="red", marker="v", s=50)

    # ----------------- Stochastic -----------------
    stoch_k, stoch_d = calc_stoc(df)
    ax_stoch.plot(df.index, stoch_k, label="%K", color="orange")
    ax_stoch.plot(df.index, stoch_d, label="%D", color="purple")
    ax_stoch.axhline(20, linestyle="--", color="green", alpha=0.5)
    ax_stoch.axhline(80, linestyle="--", color="red", alpha=0.5)
    ax_stoch.set_ylabel("Stoch")
    ax_stoch.legend(loc="upper left", fontsize="x-small")

    # ----------------- CCI -----------------
    cci = calc_cci(df)
    ax_cci.plot(df.index, cci, label="CCI", color="brown")
    ax_cci.axhline(-100, linestyle="--", color="green", alpha=0.5)
    ax_cci.axhline(100, linestyle="--", color="red", alpha=0.5)
    ax_cci.set_ylabel("CCI")
    ax_cci.legend(loc="upper left", fontsize="x-small")

    # ----------------- MFI -----------------
    mfi = calc_mfi(df)
    ax_mfi.plot(df.index, mfi, label="MFI", color="teal")
    ax_mfi.axhline(20, linestyle="--", color="green", alpha=0.5)
    ax_mfi.axhline(80, linestyle="--", color="red", alpha=0.5)
    ax_mfi.set_ylabel("MFI")
    ax_mfi.legend(loc="upper left", fontsize="x-small")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_detailed_signals3(df, signals, symbol, timeframe, final_signal, engulf_signal):
    """
    Candlestick + RSI/Stoch/CCI/MFI + Engulfing highlight
    """
    if df is None or df.empty:
        raise ValueError("Empty dataframe passed")

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df = df.set_index(pd.to_datetime(df["time"]))
        else:
            raise ValueError("No datetime index found in df")

    df = df.rename(columns={c: c.lower() for c in df.columns})
    df = df.tail(150)
    width_day = 0.03
    upcolor, downcolor = "#000000", "#ff0000"
    bull_engulf_color, bear_engulf_color = "#00ff00", "#ff00ff"

    fig, axs = plt.subplots(5, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios":[3,1,1,1,1]}, dpi=100)
    ax_price, ax_rsi, ax_stoch, ax_cci, ax_mfi = axs

    # ----------------- Candlesticks -----------------
    for ts, row in df.iterrows():
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = upcolor if c >= o else downcolor
        ax_price.vlines(ts, l, h, color=color, linewidth=1)
        rect_bottom = min(o, c)
        rect_height = abs(c - o)
        ax_price.add_patch(
            plt.Rectangle(
                (mdates.date2num(ts) - width_day/2, rect_bottom),
                width_day,
                rect_height if rect_height > 0 else 0.001,
                facecolor=color, edgecolor=color, alpha=0.9
            )
        )

    # Highlight last candle if engulfing
    if "Bullish Engulfing" in engulf_signal:
        ax_price.add_patch(
            plt.Rectangle(
                (mdates.date2num(df.index[-1]) - width_day/2, df['open'].iloc[-1]),
                width_day,
                df['close'].iloc[-1]-df['open'].iloc[-1],
                facecolor=bull_engulf_color,
                alpha=0.7
            )
        )
    elif "Bearish Engulfing" in engulf_signal:
        ax_price.add_patch(
            plt.Rectangle(
                (mdates.date2num(df.index[-1]) - width_day/2, df['open'].iloc[-1]),
                width_day,
                df['close'].iloc[-1]-df['open'].iloc[-1],
                facecolor=bear_engulf_color,
                alpha=0.7
            )
        )

    ax_price.set_title(f"{symbol} ({timeframe}) → {final_signal}\nEngulfing: {engulf_signal}")

    # ----------------- RSI -----------------
    rsi = calc_rsi(df["close"])
    ax_rsi.plot(df.index, rsi, color="blue", label="RSI")
    ax_rsi.axhline(30, linestyle="--", color="green", alpha=0.5)
    ax_rsi.axhline(70, linestyle="--", color="red", alpha=0.5)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.legend(loc="upper left", fontsize="x-small")
    if "Buy" in signals["RSI"]:
        ax_rsi.scatter(df.index[-1], rsi.iloc[-1], color="green", marker="^", s=50)
    elif "Sell" in signals["RSI"]:
        ax_rsi.scatter(df.index[-1], rsi.iloc[-1], color="red", marker="v", s=50)

    # ----------------- Stochastic -----------------
    stoch_k, stoch_d = calc_stoc(df)
    ax_stoch.plot(df.index, stoch_k, label="%K", color="orange")
    ax_stoch.plot(df.index, stoch_d, label="%D", color="purple")
    ax_stoch.axhline(20, linestyle="--", color="green", alpha=0.5)
    ax_stoch.axhline(80, linestyle="--", color="red", alpha=0.5)
    ax_stoch.set_ylabel("Stoch")
    ax_stoch.legend(loc="upper left", fontsize="x-small")

    # ----------------- CCI -----------------
    cci = calc_cci(df)
    ax_cci.plot(df.index, cci, label="CCI", color="brown")
    ax_cci.axhline(-100, linestyle="--", color="green", alpha=0.5)
    ax_cci.axhline(100, linestyle="--", color="red", alpha=0.5)
    ax_cci.set_ylabel("CCI")
    ax_cci.legend(loc="upper left", fontsize="x-small")

    # ----------------- MFI -----------------
    mfi = calc_mfi(df)
    ax_mfi.plot(df.index, mfi, label="MFI", color="teal")
    ax_mfi.axhline(20, linestyle="--", color="green", alpha=0.5)
    ax_mfi.axhline(80, linestyle="--", color="red", alpha=0.5)
    ax_mfi.set_ylabel("MFI")
    ax_mfi.legend(loc="upper left", fontsize="x-small")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    plt.close(fig)
    return buf


# Load watchlist
with open("config.json") as f:
    config = json.load(f)
WATCHLIST = config["watch_list"]

# Example indicator functions (you already have them defined)
# calc_rsi, calc_stoch, calc_cci, calc_mfi

def generate_signals3(df):
    signals = {}

    # Run indicators
    rsi  = calc_rsi(df['close']).iloc[-1]
    stoch, _ = calc_stoc(df)
    stoch_val = stoch.iloc[-1]
    cci = calc_cci(df).iloc[-1]
    mfi = calc_mfi(df).iloc[-1]

    # Interpret
    signals["RSI"] = "Buy ✅" if rsi < 30 else "Sell ❌" if rsi > 70 else "Neutral ⚪"
    signals["Stoch"] = ("Buy ✅" if stoch_val < 20 else "Sell ❌" if stoch_val > 80 else "Neutral ⚪")
    signals["CCI"] = "Buy ✅" if cci < -100 else "Sell ❌" if cci > 100 else "Neutral ⚪"
    signals["MFI"] = "Buy ✅" if mfi < 20 else "Sell ❌" if mfi > 80 else "Neutral ⚪"

    return signals


def plot_detailed_signals4(df, signals, symbol, timeframe, final_signal, pattern_signal, highlight_idx=None):
    df_plot = df.tail(150)

    fig, (ax_price, ax_ind) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}, dpi=100
    )

    width_day = 0.03
    upcolor, downcolor = "#000000", "#ff0000"

    # 🔹 Candlesticks
    for ts, row in df_plot.iterrows():
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = upcolor if c >= o else downcolor
        ax_price.vlines(ts, l, h, color=color, linewidth=1)
        rect_bottom = min(o, c)
        rect_height = abs(c - o)
        ax_price.add_patch(
            plt.Rectangle(
                (mdates.date2num(ts)-width_day/2, rect_bottom),
                width_day, rect_height if rect_height > 0 else 0.001,
                facecolor=color, edgecolor=color, alpha=0.8
            )
        )

    # 🔹 Special highlight for Harami
    if highlight_idx is not None and highlight_idx in df_plot.index:
        curr_ts = highlight_idx
        prev_ts = df_plot.index[df_plot.index.get_loc(curr_ts) - 1]

        curr_open, curr_close = df_plot.loc[curr_ts, ["open", "close"]]
        prev_open, prev_close = df_plot.loc[prev_ts, ["open", "close"]]

        # Parent candle shaded gray
        ax_price.add_patch(
            plt.Rectangle(
                (mdates.date2num(prev_ts)-width_day/2, min(prev_open, prev_close)),
                width_day, abs(prev_close-prev_open),
                facecolor="gray", alpha=0.3, edgecolor="black"
            )
        )
        # Harami candle with yellow border
        color = "green" if curr_close > curr_open else "red"
        ax_price.add_patch(
            plt.Rectangle(
                (mdates.date2num(curr_ts)-width_day/2, min(curr_open, curr_close)),
                width_day, abs(curr_close-curr_open),
                facecolor=color, edgecolor="yellow", linewidth=2
            )
        )

    # Indicators (example: RSI panel, you can extend with CCI, MFI, etc.)
    ax_ind.plot(df_plot.index, df_plot["RSI"], label="RSI", color="purple")
    ax_ind.axhline(30, linestyle="--", color="green", alpha=0.5)
    ax_ind.axhline(70, linestyle="--", color="red", alpha=0.5)
    ax_ind.legend(loc="upper left", fontsize="x-small")

    ax_price.set_title(f"{symbol} ({timeframe}) — {final_signal} | {pattern_signal}")
    ax_price.set_ylabel("Price")

    plt.xticks(rotation=20)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    plt.close(fig)

    return buf

def plot_meeting_lines_chart(df, signals, symbol, timeframe, final_signal, meeting_signal, highlight_idx):
    import mplfinance as mpf
    import io

    df_plot = df.copy()

    # Colors
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s  = mpf.make_mpf_style(marketcolors=mc)

    # Candle highlights: hollow for meeting line candles
    add_plots = []
    hlines = []
    for idx in highlight_idx:
        if idx in df_plot.index:
            hlines.append((df_plot.loc[idx, "close"],))

    fig, ax = mpf.plot(
        df_plot,
        type='candle',
        style=s,
        title=f"{symbol} ({timeframe}) - {final_signal}\n{meeting_signal}",
        ylabel='Price',
        returnfig=True
    )

    for idx in highlight_idx:
        if idx in df_plot.index:
            row = df_plot.loc[idx]
            color = 'g' if row['close'] > row['open'] else 'r'
            ax[0].add_patch(
                plt.Rectangle(
                    (idx, min(row['open'], row['close'])),
                    width=0.5,
                    height=abs(row['close'] - row['open']),
                    fill=False, edgecolor=color, linewidth=2
                )
            )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


# -------------------- Events --------------------
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"❌ Missing argument: `{error.param.name}`\nUsage: !sell SYMBOL LOT sl=... tp=...")
    elif isinstance(error, commands.BadArgument):
        await ctx.send("❌ Invalid argument type. Make sure numbers are correct.")
    else:
        # fallback for other errors
        await ctx.send(f"⚠️ Error: {error}")
        raise error  # re-raise so you still see traceback in terminal

# -------------------- Commands --------------------

@bot.command(name="mirror")
async def mirror(ctx, timeframe: str = "1h"):
    """
    Run combined multi-indicator + Meeting Lines strategy across all pairs in WATCHLIST.
    Shows detailed indicator charts only for pairs where >=3 signals agree.
    Meeting line candles are shown hollowed to highlight them.
    """
    for symbol in WATCHLIST:
        try:
            # 1️⃣ Fetch historical data
            df = fetch_history_mt5(symbol, timeframe=timeframe, period="10d")
            if df is None or df.empty:
                await ctx.send(f"⚠️ No data for {symbol}")
                continue

            # 2️⃣ Indicators
            df["RSI"] = calc_rsi(df["close"])
            df["CCI"] = calc_cci(df)
            stoch_k, stoch_d = calc_stoc(df)
            df["%K"], df["%D"] = stoch_k, stoch_d
            df["MFI"] = calc_mfi(df)

            # 3️⃣ Detect Meeting Lines
            meeting_signal, direction = "Neutral ⚪", None
            meeting_idx = []

            if len(df) >= 2:
                prev, curr = df.iloc[-2], df.iloc[-1]
                prev_mid = (prev["open"] + prev["close"]) / 2

                # Bullish Meeting Line
                if prev["close"] < prev["open"] and curr["close"] > curr["open"]:
                    if abs(curr["close"] - prev_mid) <= (prev["Body"] * 0.1):  # within 10% of body midpoint
                        meeting_signal = "Bullish Meeting Line ✅"
                        direction = "Buy"
                        meeting_idx = [df.index[-2], df.index[-1]]

                # Bearish Meeting Line
                elif prev["close"] > prev["open"] and curr["close"] < curr["open"]:
                    if abs(curr["close"] - prev_mid) <= (prev["Body"] * 0.1):
                        meeting_signal = "Bearish Meeting Line ❌"
                        direction = "Sell"
                        meeting_idx = [df.index[-2], df.index[-1]]

            # 4️⃣ Signals
            signals = generate_signals3_from_series(df)
            buy_count = sum(1 for s in signals.values() if "Buy" in s)
            sell_count = sum(1 for s in signals.values() if "Sell" in s)

            # 5️⃣ RSI Confirmation
            if direction == "Buy" and df["RSI"].iloc[-1] < 30:
                meeting_signal += " + RSI ✅"
            elif direction == "Sell" and df["RSI"].iloc[-1] > 70:
                meeting_signal += " + RSI ✅"
            else:
                meeting_signal += " + RSI ⚠️ Not confirmed"

            # 6️⃣ Final Signal
            if buy_count >= 3 or "Buy" in meeting_signal:
                final_signal = "STRONG BUY"
            elif sell_count >= 3 or "Sell" in meeting_signal:
                final_signal = "STRONG SELL"
            elif buy_count == 1 or sell_count == 1:
                final_signal = "Weak/Single Signal"
            else:
                final_signal = "Mixed/Neutral"

            # 7️⃣ Chart plotting with hollow meeting line candles
            if (buy_count >= 3 or sell_count >= 3 or "Meeting Line" in meeting_signal) and meeting_idx:
                buf = plot_meeting_lines_chart(
                    df, signals, symbol, timeframe,
                    final_signal, meeting_signal, highlight_idx=meeting_idx
                )
                file = discord.File(buf, filename=f"{symbol}_mirror.png")
                await ctx.send(
                    content=f"📊 {symbol} ({timeframe}) detailed breakdown:\n{final_signal}\nMeeting Line: {meeting_signal}\nSignals: {signals}",
                    file=file
                )
            else:
                await ctx.send(
                    f"📊 {symbol} ({timeframe}) — {final_signal}\nMeeting Line: {meeting_signal}\nSignals: {signals}"
                )

        except Exception as e:
            await ctx.send(f"❌ Error in mirror for {symbol}: {e}")


@bot.command(name="parasite")
async def parasite(ctx, timeframe: str = "1h"):
    """
    Run combined multi-indicator + Bullish/Bearish Harami strategy across all pairs in WATCHLIST.
    Shows detailed indicator charts only for pairs where >=3 signals agree.
    """
    for symbol in WATCHLIST:
        try:
            # 1️⃣ Fetch historical data
            df = fetch_history_mt5(symbol, timeframe=timeframe, period="10d")
            if df is None or df.empty:
                await ctx.send(f"⚠️ No data for {symbol}")
                continue

            # 2️⃣ Calculate indicators
            df["RSI"] = calc_rsi(df["close"])
            df["CCI"] = calc_cci(df)
            stoch_k, stoch_d = calc_stoc(df)
            df["%K"] = stoch_k
            df["%D"] = stoch_d
            df["MFI"] = calc_mfi(df)

            # 3️⃣ Detect Harami
            harami_signal = "Neutral ⚪"
            direction = None
            harami_idx = None

            if len(df) >= 2:
                prev = df.iloc[-2]
                curr = df.iloc[-1]

                # Bullish Harami
                if prev["close"] < prev["open"] and \
                   min(curr["open"], curr["close"]) > min(prev["open"], prev["close"]) and \
                   max(curr["open"], curr["close"]) < max(prev["open"], prev["close"]):
                    harami_signal = "Bullish Harami ✅"
                    direction = "Buy"
                    harami_idx = df.index[-1]

                # Bearish Harami
                elif prev["close"] > prev["open"] and \
                     min(curr["open"], curr["close"]) > min(prev["open"], prev["close"]) and \
                     max(curr["open"], curr["close"]) < max(prev["open"], prev["close"]):
                    harami_signal = "Bearish Harami ❌"
                    direction = "Sell"
                    harami_idx = df.index[-1]

            # 4️⃣ Multi-indicator signals
            signals = generate_signals3_from_series(df)

            buy_count = sum(1 for s in signals.values() if "Buy" in s)
            sell_count = sum(1 for s in signals.values() if "Sell" in s)

            # 5️⃣ Confirm with RSI
            if direction == "Buy" and df["RSI"].iloc[-1] < 30:
                harami_signal += " + RSI ✅"
            elif direction == "Sell" and df["RSI"].iloc[-1] > 70:
                harami_signal += " + RSI ✅"
            else:
                harami_signal += " + RSI ⚠️ Not confirmed"

            # 6️⃣ Final decision
            if buy_count >= 3 or "Buy" in harami_signal:
                final_signal = "STRONG BUY"
            elif sell_count >= 3 or "Sell" in harami_signal:
                final_signal = "STRONG SELL"
            elif buy_count == 1 or sell_count == 1:
                final_signal = "Weak/Single Signal"
            else:
                final_signal = "Mixed/Neutral"

            # 7️⃣ Plot only if strong signal
            if (buy_count >= 3 or sell_count >= 3 or "Harami" in harami_signal) and harami_idx is not None:
                buf = plot_detailed_signals4(
                    df, signals, symbol, timeframe,
                    final_signal, harami_signal, highlight_idx=harami_idx
                )
                file = discord.File(buf, filename=f"{symbol}_parasite.png")
                await ctx.send(
                    content=f"📊 {symbol} ({timeframe}) detailed breakdown:\n{final_signal}\nHarami: {harami_signal}\nSignals: {signals}",
                    file=file
                )
            else:
                await ctx.send(
                    f"📊 {symbol} ({timeframe}) — {final_signal}\nHarami: {harami_signal}\nSignals: {signals}"
                )

        except Exception as e:
            await ctx.send(f"❌ Error for {symbol}: {e}")


@bot.command(name="minotaur")
async def minotaur(ctx, timeframe: str = "1h"):
    """
    Run combined multi-indicator + Bullish/Bearish Engulfing strategy across all pairs in WATCHLIST.
    Shows detailed indicator charts only for pairs where >=3 signals agree.
    """
    for symbol in WATCHLIST:
        try:
            # 1️⃣ Fetch historical data
            df = fetch_history_mt5(symbol, timeframe=timeframe, period="10d")
            if df is None or df.empty:
                await ctx.send(f"⚠️ No data for {symbol}")
                continue

            # 2️⃣ Calculate indicators
            df["RSI"] = calc_rsi(df["close"])
            df["CCI"] = calc_cci(df)
            stoch_k, stoch_d = calc_stoc(df)
            df["%K"] = stoch_k
            df["%D"] = stoch_d
            df["MFI"] = calc_mfi(df)

            # 3️⃣ Detect Engulfing patterns
            df['Body'] = abs(df['close'] - df['open'])
            engulf_signal = "Neutral ⚪"
            direction = None
            info = "No Engulfing pattern detected"

            if len(df) >= 2:
                prev = df.iloc[-2]
                curr = df.iloc[-1]

                # Bullish Engulfing
                if prev['close'] < prev['open'] and curr['close'] > curr['open'] and curr['close'] > prev['open'] and curr['open'] < prev['close']:
                    engulf_signal = "Bullish Engulfing ✅"
                    direction = "Buy"
                    info = "Bullish Engulfing detected"
                # Bearish Engulfing
                elif prev['close'] > prev['open'] and curr['close'] < curr['open'] and curr['close'] < prev['open'] and curr['open'] > prev['close']:
                    engulf_signal = "Bearish Engulfing ❌"
                    direction = "Sell"
                    info = "Bearish Engulfing detected"

            # 4️⃣ Generate multi-indicator signals
            signals = generate_signals3_from_series(df)

            # Count number of buy/sell signals
            buy_count = sum(1 for s in signals.values() if "Buy" in s)
            sell_count = sum(1 for s in signals.values() if "Sell" in s)

            # 5️⃣ Combine final decision with engulfing confirmation
            if direction == "Buy" and stoch_k.iloc[-1] < 20:
                engulf_signal += " + Stoch ✅"
            elif direction == "Sell" and stoch_k.iloc[-1] > 80:
                engulf_signal += " + Stoch ✅"
            elif direction:
                engulf_signal += " + Stoch ⚠️ Not confirmed"

            # 6️⃣ Decide final signal strength
            if buy_count >= 3 or "Buy" in engulf_signal:
                final_signal = "STRONG BUY"
            elif sell_count >= 3 or "Sell" in engulf_signal:
                final_signal = "STRONG SELL"
            elif buy_count == 1 or sell_count == 1:
                final_signal = "Weak/Single Signal"
            else:
                final_signal = "Mixed/Neutral"

            # 7️⃣ Only plot chart if strong signal
            if buy_count >= 3 or sell_count >= 3 or "Engulfing" in engulf_signal:
                buf = plot_detailed_signals3(df, signals, symbol, timeframe, final_signal, engulf_signal)
                file = discord.File(buf, filename=f"{symbol}_minotaur.png")
                await ctx.send(
                    content=f"📊 {symbol} ({timeframe}) detailed breakdown:\n{final_signal}\nEngulfing: {engulf_signal}\nSignals: {signals}",
                    file=file
                )
            else:
                # Text-only report
                await ctx.send(
                    f"📊 {symbol} ({timeframe}) — {final_signal}\nEngulfing: {engulf_signal}\nSignals: {signals}"
                )

        except Exception as e:
            await ctx.send(f"❌ Error for {symbol}: {e}")



@bot.command(name="raven")
async def crows_detailed(ctx, timeframe: str = "1h"):
    """
    Run combined strategy across all pairs in WATCHLIST.
    Show detailed indicator charts (candlestick + RSI/Stoch/CCI/MFI) 
    only for pairs where >=3 signals agree.
    """
    for symbol in WATCHLIST:
        # 1️⃣ Fetch historical data
        df = fetch_history_mt5(symbol, timeframe=timeframe)
        if df is None or df.empty:
            await ctx.send(f"⚠️ No data for {symbol}")
            continue

        # 2️⃣ Compute indicators
        df["RSI"] = calc_rsi(df["close"])
        df["CCI"] = calc_cci(df)
        stoch_k, stoch_d = calc_stoc(df)
        df["%K"] = stoch_k
        df["%D"] = stoch_d
        df["MFI"] = calc_mfi(df)

        # 3️⃣ Generate signals
        signals = generate_signals3(df)

        buy_count = sum(1 for s in signals.values() if "Buy" in s)
        sell_count = sum(1 for s in signals.values() if "Sell" in s)

        # 4️⃣ Decide final signal
        if buy_count >= 3:
            final_signal = " STRONG BUY"
        elif sell_count >= 3:
            final_signal = " STRONG SELL"
        elif buy_count == 1 or sell_count == 1:
            final_signal = " Weak/Single Signal"
        else:
            final_signal = " Mixed/Neutral"

        # 5️⃣ Only plot if 3+ indicators agree
        if buy_count >= 3 or sell_count >= 3:
            buf = plot_detailed_signals2(df, signals, symbol, timeframe, final_signal)
            file = discord.File(buf, filename=f"{symbol}_detailed.png")
            await ctx.send(
                content=f"📊 {symbol} ({timeframe}) detailed breakdown:\n{final_signal}\n\n**Signals:** {signals}",
                file=file
            )
        else:
            # Text-only report
            await ctx.send(
                f"📊 {symbol} ({timeframe}) — {final_signal}\nSignals: {signals}"
            )


'''@bot.command(name="crowssd")
async def crows_detailed(ctx, timeframe: str = "1h"):
    for symbol in WATCHLIST:
        try:
            df = fetch_history_mt5(symbol, timeframe=timeframe)
            
            ''''''df_proc, df_plot = _standardize_and_index(df)
            print("columns:", df_proc.columns.tolist())
            print("index type:", type(df_proc.index), df_proc.index[:3])
            print(df_plot[['Open','High','Low','Close']].head().to_string())''''''
        except Exception as e:
            await ctx.send(f"❌ Error fetching {symbol}: {e}")
            continue

        if df is None or len(df) == 0:
            await ctx.send(f"❌ No data for {symbol}")
            continue

        # standardize dataframe and prepare mplfinance dataframe
        try:
            df_proc, df_plot = _standardize_and_index(df)
        except KeyError as e:
            await ctx.send(f"❌ Data format error for {symbol}: {e}")
            # optional: dump head for debugging
            try:
                head = df.head(3).to_dict()
                await ctx.send(f"Data sample: ```{head}```")
            except Exception:
                pass
            continue

        # require enough bars to compute indicators
        if len(df_proc) < 20:
            await ctx.send(f"ℹ️ {symbol} ({timeframe}) — not enough bars ({len(df_proc)}) for indicators.")
            continue

        # compute indicators (aligned with df_proc index)
        df_proc['RSI'] = calc_rsi(df_proc['close'])
        df_proc['%K'], df_proc['%D'] = calc_stoc(df_proc)
        df_proc['CCI'] = calc_cci(df_proc)
        df_proc['MFI'] = calc_mfi(df_proc)
        macd_line, macd_sig, macd_hist = calc_macd(df_proc['close'])
        df_proc['MACD'] = macd_line
        df_proc['MACD_SIG'] = macd_sig
        df_proc['MACD_HIST'] = macd_hist

        signals = generate_signals3_from_series(df_proc)

        buy_count  = sum(1 for s in signals.values() if "Buy" in s)
        sell_count = sum(1 for s in signals.values() if "Sell" in s)

        if buy_count >= 3:
            final = " STRONG BUY"
        elif sell_count >= 3:
            final = " STRONG SELL"
        else:
            # only report signals, do not create heavy chart
            await ctx.send(f"ℹ️ {symbol} ({timeframe}) → **Signals:** {signals} → ⚖️ No chart (not enough confirmations).")
            continue

        # ------------------ PLOT ------------------
        # keep recent N bars for readability
        N = 150
        df_plot_section = df_plot.tail(N)
        df_proc_section = df_proc.tail(N)

        # create figure with panels: candles + RSI + Stoch + CCI + MFI
        fig, axs = plt.subplots(5, 1, figsize=(14, 12), sharex=True,
                                 gridspec_kw={"height_ratios":[3, 1, 1, 1, 1]})

        # Candles via mplfinance on axs[0]
        try:
            # use mpf.plot with ax - mplfinance will draw candles on the provided Axes
            #mpf.plot(df_plot_section, type='candle', ax=axs[0], volume=False, style='charles', show_nontrading=False)
            #fig, ax = 
            mpf.plot(
            df_plot_section,
            type="candle",
            ax=axs[0],
            style="charles",
            ylabel="Price")
            #axs[0].set_title(f"{symbol} ({timeframe}) - {final}")
        except Exception as e:
            # fallback: send debug message and skip
            await ctx.send(f"❌ mplfinance plot error for {symbol}: {e}")
            plt.close(fig)
            continue

        #axs[0].set_title(f"{symbol} ({timeframe}) — {final}")
        #axs[0].set_ylabel("Price")

        # RSI (small markers only, small size so they don't affect autoscale)
        axs[1].plot(df_proc_section.index, df_proc_section['RSI'], linewidth=1)
        axs[1].axhline(70, linestyle="--", alpha=0.5)
        axs[1].axhline(30, linestyle="--", alpha=0.5)
        # small marker for last point (clip_on False avoids changing axis limits)
        axs[1].scatter(df_proc_section.index[-1], df_proc_section['RSI'].iloc[-1],
                       marker="^" if "Buy" in signals["RSI"] else ("v" if "Sell" in signals["RSI"] else "o"),
                       s=40, clip_on=False)
        axs[1].set_ylabel("RSI")

        # Stochastic
        axs[2].plot(df_proc_section.index, df_proc_section['%K'], linewidth=1, label="%K")
        axs[2].plot(df_proc_section.index, df_proc_section['%D'], linewidth=1, label="%D")
        axs[2].axhline(80, linestyle="--", alpha=0.5)
        axs[2].axhline(20, linestyle="--", alpha=0.5)
        axs[2].legend(loc="upper left")
        axs[2].set_ylabel("Stoch")

        # CCI
        axs[3].plot(df_proc_section.index, df_proc_section['CCI'], linewidth=1)
        axs[3].axhline(100, linestyle="--", alpha=0.5)
        axs[3].axhline(-100, linestyle="--", alpha=0.5)
        axs[3].set_ylabel("CCI")

        # MFI
        axs[4].plot(df_proc_section.index, df_proc_section['MFI'], linewidth=1)
        axs[4].axhline(80, linestyle="--", alpha=0.5)
        axs[4].axhline(20, linestyle="--", alpha=0.5)
        axs[4].set_ylabel("MFI")

        plt.tight_layout()

        # save & send
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        plt.close(fig)

        file = discord.File(buf, filename=f"{symbol}_detailed.png")
        await ctx.send(content=f"📊 {symbol} ({timeframe}) detailed breakdown:\n{final}\n**Signals:** {signals}", file=file)
'''

'''@bot.command()
async def crows(ctx, timeframe: str = "1h"):
    """
    Run combined strategy across all pairs in watchlist.
    """
    report = ""
    for symbol in WATCHLIST:
        df = fetch_history_mt5(symbol, timeframe=timeframe)

        signals = generate_signals3(df)
        buy_count = sum(1 for s in signals.values() if "Buy" in s)
        sell_count = sum(1 for s in signals.values() if "Sell" in s)

        # Strength of signal
        if buy_count >= 3:
            final = "📈 STRONG BUY"
        elif sell_count >= 3:
            final = "📉 STRONG SELL"
        elif buy_count == 1 or sell_count == 1:
            final = "🤔 Weak/Single Signal"
        else:
            final = "⚖️ Mixed/Neutral"

        # Add to text report
        report += f"**{symbol} {timeframe}** → {final}\n"
        for k, v in signals.items():
            report += f"   • {k}: {v}\n"
        report += "\n"

        # Generate chart with signals
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(df['close'], label="Price", color="black")

        # Plot markers where signals triggered
        ''''''if "Buy" in signals["RSI"]:
            ax.scatter(len(df)-1, df['close'].iloc[-1], color="green", marker="^", s=100, label="RSI Buy")
        if "Sell" in signals["RSI"]:
            ax.scatter(len(df)-1, df['close'].iloc[-1], color="red", marker="v", s=100, label="RSI Sell")''''''
        # (You can repeat for Stoch, CCI, MFI with different colors/shapes)

        ax.legend()
        ax.set_title(f"{symbol} Signals ({timeframe})")


        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        #fig = plot_signals(df, sigs)
        if fig:
            file = discord.File(buf, filename=f"{symbol}.png")
            await ctx.send(content=f"📊 {symbol} results:\n{final}", file=file)
            #img_bytes = pio.to_image(fig, format="png")
            #with io.BytesIO(img_bytes) as image_binary:
                #await ctx.send(file=discord.File(fp=image_binary, filename=f"{symbol}_signals.png"))

        # Send chart + text
        #file = discord.File(buf, filename=f"{symbol}.png")
        #await ctx.send(content=f"📊 {symbol} results:\n{final}", file=file)

    # Final summary
    #await ctx.send(f"✅ Combined Report:\n{report}")'''
            

'''@bot.command(name="crowssd")
async def crows_detailed(ctx, timeframe: str = "1h"):
    for symbol in WATCHLIST:
        df = fetch_history_mt5(symbol, timeframe=timeframe)

        # ensure datetime index for mplfinance
        #df["time"] = pd.to_datetime(df["time"])
        #df.set_index("time", inplace=True)

        # calculate indicators
        df["RSI"] = calc_rsi(df["close"])
        df["%K"], df["%D"] = calc_stoc(df)
        df["CCI"] = calc_cci(df)
        df["MFI"] = calc_mfi(df)

        signals = generate_signals3(df)

        buy_count  = sum(1 for s in signals.values() if "Buy" in s)
        sell_count = sum(1 for s in signals.values() if "Sell" in s)

        if buy_count >= 3:
            final = "📈 STRONG BUY"
        elif sell_count >= 3:
            final = "📉 STRONG SELL"
        else:
            # just report signals, no chart
            await ctx.send(
                content=f"ℹ️ {symbol} ({timeframe}) → {signals} → ⚖️ Neutral (no chart)"
            )
            continue

        # build subplots (candles + indicators)
        fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

        # Ensure df has proper format for mplfinance
        df_plot = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low", "close": "Close", "tick_volume": "Volume"
        })
        df_plot.set_index(pd.to_datetime(df_plot.index), inplace=True)

        # Candlestick chart
        mpf.plot(
            df_plot.tail(150),
            type="candle",
            ax=axs[0],
            style="charles",
            ylabel="Price"
        )
        axs[0].set_title(f"{symbol} ({timeframe}) → {final}")

        # RSI
        axs[1].plot(df.index, df["RSI"], label="RSI", color="purple")
        axs[1].axhline(70, linestyle="--", color="red", alpha=0.6)
        axs[1].axhline(30, linestyle="--", color="green", alpha=0.6)
        axs[1].set_ylabel("RSI")

        # Stochastic
        axs[2].plot(df.index, df["%K"], label="%K", color="blue")
        axs[2].plot(df.index, df["%D"], label="%D", color="orange")
        axs[2].axhline(80, linestyle="--", color="red", alpha=0.6)
        axs[2].axhline(20, linestyle="--", color="green", alpha=0.6)
        axs[2].set_ylabel("Stoch")
        axs[2].legend()

        # CCI
        axs[3].plot(df.index, df["CCI"], color="brown")
        axs[3].axhline(100, linestyle="--", color="red", alpha=0.6)
        axs[3].axhline(-100, linestyle="--", color="green", alpha=0.6)
        axs[3].set_ylabel("CCI")

        # MFI
        axs[4].plot(df.index, df["MFI"], color="teal")
        axs[4].axhline(80, linestyle="--", color="red", alpha=0.6)
        axs[4].axhline(20, linestyle="--", color="green", alpha=0.6)
        axs[4].set_ylabel("MFI")

        plt.tight_layout()

        # send to discord
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        file = discord.File(buf, filename=f"{symbol}_detailed.png")

        await ctx.send(
            content=f"📊 {symbol} ({timeframe}) detailed breakdown:\n{final}\n\n"
                    f"**Signals:** {signals}",
            file=file
        )'''

@bot.command()
async def crows(ctx, timeframe: str = "1h"):
    """
    Run combined strategy across all pairs in watchlist.
    Shows charts only for STRONG BUY/SELL.
    """
    report = ""
    for symbol in WATCHLIST:
        df = fetch_history_mt5(symbol, timeframe=timeframe)
        if df is None or df.empty:
            await ctx.send(f"⚠️ No data for {symbol}")
            continue

        # Generate indicator signals with explanations
        signals = generate_signals3(df)

        # Count buy/sell
        buy_count = sum(1 for s in signals.values() if "Buy" in s)
        sell_count = sum(1 for s in signals.values() if "Sell" in s)

        # Strength of signal
        if buy_count >= 3:
            final = " STRONG BUY"
        elif sell_count >= 3:
            final = " STRONG SELL"
        elif buy_count == 1 or sell_count == 1:
            final = " Weak/Single Signal"
        else:
            final = " Mixed/Neutral"

        # Build text with reasoning
        report += f"**{symbol} {timeframe}** → {final}\n"
        for k, v in signals.items():
            report += f"   • {k}: {v}\n"
        report += "\n"

        # Show chart only if strong
        if "STRONG" in final:
            # Candlestick chart
            mc = mpf.make_marketcolors(
                up='green', down='red',
                edge='black', wick='yellow',
                volume='yellow'
            )
            s  = mpf.make_mpf_style(marketcolors=mc, base_mpf_style="nightclouds")

            fig, ax = mpf.plot(
                df.tail(80),  # last ~80 candles
                type='candle',
                style=s,
                returnfig=True,
                title=f"{symbol} ({timeframe}) - {final}",
                ylabel="Price"
            )

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            buf.seek(0)
            plt.close(fig)

            file = discord.File(buf, filename=f"{symbol}.png")
            await ctx.send(content=f"📊 {symbol} results:\n{final}", file=file)

    # Send summary after loop
    if report:
        await ctx.send(f"📝 Combined Report:\n{report}")


def plot_detailed_signals(df, signals):
    """Return matplotlib fig with candlestick + indicator subplots."""

    import mplfinance as mpf
    
    # Compute and attach indicators
    '''df["RSI"] = calc_rsi(df)
    df["Stoch"] = detect_pattern3(df)
    df["CCI"] = detect_pattern1(df)
    df["MFI"] = calc_mfi(df)'''

    # Build figure layout
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(5, 1, figure=fig, height_ratios=[3, 1, 1, 1, 1])

    # --- Candlestick chart (top)
    ax1 = fig.add_subplot(gs[0])
    mc = mpf.make_marketcolors(up='white', down='black', edge='black', wick='black')
    s  = mpf.make_mpf_style(marketcolors=mc, base_mpf_style="nightclouds")
    mpf.plot(df.tail(100), type='candle', style=s, ax=ax1, volume=False)

    # --- RSI
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    rsi = calc_rsi(df)
    ax2.plot(rsi, label="RSI", color="blue")
    ax2.axhline(30, linestyle="--", color="green")
    ax2.axhline(70, linestyle="--", color="red")
    ax2.set_ylabel("RSI")
    if "Buy" in signals["RSI"]:
        ax2.scatter(rsi.index[-1], rsi.iloc[-1], color="green", marker="^", s=80)
    elif "Sell" in signals["RSI"]:
        ax2.scatter(rsi.index[-1], rsi.iloc[-1], color="red", marker="v", s=80)

    # --- Stochastic
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    stoch = calc_stoc(df)
    ax3.plot(stoch["%K"], label="%K", color="orange")
    ax3.plot(stoch["%D"], label="%D", color="purple")
    ax3.axhline(20, linestyle="--", color="green")
    ax3.axhline(80, linestyle="--", color="red")
    ax3.set_ylabel("Stoc")
    ax3.legend()

    # --- CCI
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    cci = calc_cci(df)
    ax4.plot(cci, label="CCI", color="brown")
    ax4.axhline(-100, linestyle="--", color="green")
    ax4.axhline(100, linestyle="--", color="red")
    ax4.set_ylabel("CCI")

    # --- MFI
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    mfi = calc_mfi(df)
    ax5.plot(mfi, label="MFI", color="teal")
    ax5.axhline(20, linestyle="--", color="green")
    ax5.axhline(80, linestyle="--", color="red")
    ax5.set_ylabel("MFI")

    fig.tight_layout()
    return fig

@bot.command(name="crowsd")
async def crows_detailed(ctx, timeframe: str = "1h"):
    """
    Run combined strategy across all pairs in watchlist.
    Show detailed indicator charts (RSI, MACD, Stoch, CCI) to explain *why* a signal was triggered.
    """
    for symbol in WATCHLIST:
        df = fetch_history_mt5(symbol, timeframe=timeframe)
        # --- Compute indicators & attach as columns ---
        df["RSI"] = calc_rsi(df["close"])
        #macd, signal = macd(df)   # <-- make sure you have a calc_macd() function
        df["MACD"] = macd
        #df["Signal"] = signal
        stoch_k,stoch_d = calc_stoc(df)          # <-- must return a DataFrame with "%K" and "%D"
        df["%K"] = stoch_k
        df["%D"] = stoch_d
        df["CCI"] = calc_cci(df)
        df["MFI"] = calc_mfi(df)
        signals = generate_signals3(df)


        buy_count = sum(1 for s in signals.values() if "Buy" in s)
        sell_count = sum(1 for s in signals.values() if "Sell" in s)

        # Signal strength
        if buy_count >= 3:
            final = "📈 STRONG BUY"
        elif sell_count >= 3:
            final = "📉 STRONG SELL"
        elif buy_count == 1 or sell_count == 1:
            final = "🤔 Weak/Single Signal"
        else:
            final = "⚖️ Mixed/Neutral"

        # -------------------- DETAILED CHART --------------------
        fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

        # 1. Candlestick price chart
        
        
        
        #fig, ax = 
        mpf.plot(
                df.tail(200),  # last ~80 candles
                type='candle',
                style="charles",
                returnfig=True,
                ax=axs[0],
                title=f"{symbol} ({timeframe}) - {final}",
                #ylabel="Price"
            )
        #plot_df = df.tail(200)
        '''mpf.plot(
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True),
            df.tail(200),
            type="candle",
            ax=axs[0],
            style="charles",
            #tick_volume=True,
            #show_nontrading=True
        )
        axs[0].set_title(f"{symbol} Price + Signals ({timeframe}) → {final}")'''

        # 2. RSI
        axs[1].plot(df.index, df["RSI"], label="RSI", color="purple")
        axs[1].axhline(70, linestyle="--", color="red", alpha=0.5)
        axs[1].axhline(30, linestyle="--", color="green", alpha=0.5)
        axs[1].set_ylabel("RSI")
        '''' if "Buy" in signals.get("RSI", ""):
            axs[1].scatter(df.index[-1], df["RSI"].iloc[-1], color="green", marker="^", s=100, label="RSI Buy")
        if "Sell" in signals.get("RSI", ""):
            axs[1].scatter(df.index[-1], df["RSI"].iloc[-1], color="red", marker="v", s=100, label="RSI Sell")'''
        #axs[1].legend()

        # 3. MACD
        #axs[2].plot(df.index, df["MACD"], label="MACD", color="blue")
        #axs[2].plot(df.index, df["Signal"], label="Signal", color="orange")
        #axs[2].axhline(0, color="black", linewidth=0.8)
        #if "Buy" in signals.get("MACD", ""):
        #    axs[2].scatter(df.index[-1], df["MACD"].iloc[-1], color="green", marker="^", s=100, label="MACD Buy")
        #if "Sell" in signals.get("MACD", ""):
        #    axs[2].scatter(df.index[-1], df["MACD"].iloc[-1], color="red", marker="v", s=100, label="MACD Sell")
        #axs[2].set_ylabel("MACD")
        #axs[2].legend()

        # 4. Stochastic
        axs[3].plot(df.index, df["%K"], label="%K", color="blue")
        axs[3].plot(df.index, df["%D"], label="%D", color="orange")
        axs[3].axhline(80, linestyle="--", color="red", alpha=0.5)
        axs[3].axhline(20, linestyle="--", color="green", alpha=0.5)
        '''if "Buy" in signals.get("Stochastic", ""):
            axs[3].scatter(df.index[-1], df["%K"].iloc[-1], color="green", marker="^", s=100, label="Stoc Buy")
        if "Sell" in signals.get("Stochastic", ""):
            axs[3].scatter(df.index[-1], df["%K"].iloc[-1], color="red", marker="v", s=100, label="Stoc Sell")'''
        #axs[3].set_ylabel("Stoc")
        #axs[3].legend()

        # 5. CCI
        axs[4].plot(df.index, df["CCI"], label="CCI", color="brown")
        axs[4].axhline(100, linestyle="--", color="red", alpha=0.5)
        axs[4].axhline(-100, linestyle="--", color="green", alpha=0.5)
        '''if "Buy" in signals.get("CCI", ""):
            axs[4].scatter(df.index[-1], df["CCI"].iloc[-1], color="green", marker="^", s=100, label="CCI Buy")
        if "Sell" in signals.get("CCI", ""):
            axs[4].scatter(df.index[-1], df["CCI"].iloc[-1], color="red", marker="v", s=100, label="CCI Sell")'''
        #axs[4].set_ylabel("CCI")
        #axs[4].legend()

        plt.tight_layout()

        # Send chart to Discord
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        file = discord.File(buf, filename=f"{symbol}_detailed.png")

        await ctx.send(
            content=f"📊 {symbol} ({timeframe}) detailed breakdown:\n{final}\n\n"
                    f"**Signals:** {signals}",
            file=file
        )




@bot.command()
async def rsi(ctx, symbol: str = "EURUSDm", timeframe: str = "1h"):
    """
    Example: !rsi EURUSD=X 1h
    """
    try:
        # Map timeframe
        tf_map = {"1h": "1h", "4h": "4h", "1d": "1d"}
        interval = tf_map.get(timeframe, "1h")

        #df = yf.download(symbol, period="3mo", interval=interval)
        df = fetch_history_mt5(symbol, timeframe=timeframe, period="10d")
        if df.empty:
            await ctx.send(f"⚠️ Could not fetch data for {symbol} {timeframe}")
            return

        signal, info = detect_pattern4(df)
        if not signal:
            await ctx.send(f"🔎 {symbol} {timeframe}: {info}")
            return

        confirmed, cinfo = confirm_with_rsi(df, signal)
        if confirmed:
            await ctx.send(
                f"✅ {symbol} {timeframe}: {info}\n📊 {cinfo}\n➡️ Trade Signal: **{signal}**"
            )
        else:
            await ctx.send(
                f"⚠️ {symbol} {timeframe}: {info}\n📊 {cinfo}\n(No trade)"
            )
    except Exception as e:
        await ctx.send(f"❌ Error: {e}")

@bot.command()
async def stoc(ctx, symbol: str = "EURUSDm", timeframe: str = "1h"):
    """
    Example: !stoc EURUSDm 1h
    Checks 3 Soldiers / Crows + Stochastic confirmation
    """
    try:
        # 🔹 Replace with your MT5 fetch
        df = fetch_history_mt5(symbol, timeframe=timeframe, period="10d")
        if df is None or df.empty:
            await ctx.send(f"❌ No data for {symbol} {timeframe}")
            return
        signal, info, direction = detect_pattern3(df)
        if signal == 0:
            await ctx.send(f"🔎 {symbol} {timeframe}: {info}")
            return
        confirmed, stoch_info = confirm_signal(df, signal)
        if confirmed:
            await ctx.send(
                f"✅ {symbol} {timeframe}: {info}\n📊 {stoch_info}\n➡️ Action: **{direction}**"
            )
        else:
            await ctx.send(
                f"⚠️ {symbol} {timeframe}: {info}\n📊 {stoch_info}\n(No trade)"
            )
    except Exception as e:
        await ctx.send(f"❌ Error: {e}")

@bot.command()
async def mfi(ctx, symbol: str = "EURUSDm", interval: str = "1h"):
    """Check candlestick + MFI pattern for a given symbol"""
    try:
        #data = yf.download(symbol, , period="5d")
        data = fetch_history_mt5(symbol,timeframe=interval, period="5d")
        data['MFI'] = calc_mfi(data, MFI_PERIOD)

        signal, info, direction = detect_pattern2(data)

        if signal == 0:
            await ctx.send(f"🔍 {symbol} ({interval}): {info}")
        else:
            last_mfi = data['MFI'].iloc[-2]
            confirmed = (signal == 1 and last_mfi < 40) or (signal == -1 and last_mfi > 60)

            if confirmed:
                await ctx.send(f"📊 {symbol} ({interval})\nℹ️ {info}\n✅ Confirmed with MFI ({last_mfi:.2f}) → **{direction}**")
            else:
                await ctx.send(f"📊 {symbol} ({interval})\nℹ️ {info}\n⚠️ Pattern found but not confirmed by MFI ({last_mfi:.2f})")
    except Exception as e:
        await ctx.send(f"❌ Error: {e}")


@bot.command()
async def cci(ctx, symbol: str = "EURUSDm", timeframe: str = "1h"):
    """
    Example: !cci EURUSDm 1h
    """
    try:
        # Use yfinance to fetch data
        df = fetch_history_mt5(symbol,  timeframe=timeframe)#'''period="1d",'''
        #df = df.dropna()
        # Detect candlestick pattern
        signal, info = detect_pattern1(df)
        if not signal:
            await ctx.send(f"🔎 {symbol} {timeframe}: {info}")
            return
        # Confirm with CCI
        confirmed, cci_info = confirm_signal(df, signal)
        if confirmed:
            await ctx.send(f"✅ {symbol} {timeframe}: {info}\n📊 {cci_info}\n➡️ Signal: **{signal}**")
        else:
            await ctx.send(f"⚠️ {symbol} {timeframe}: {info}\n📊 {cci_info}\n(No trade)")
    except Exception as e:
        await ctx.send(f"❌ Error: {e}")


@bot.command(name="freq")
async def cmd_freq(ctx, symbol: str = "EURUSDm", period: str = "1d", timeframe: str = "1m"):
    """
    Frequency analysis of market using FFT (like AC power analysis).
    Shows dominant market frequency, amplitude, and volatility.
    """
    try:
        df = fetch_history_mt5(symbol, period=period, timeframe=timeframe)
        if df is None or df.empty:
            await ctx.send("No data available.")
            return

        # Use closing prices
        closes = df["close"].values
        n = len(closes)

        # Remove mean to center signal
        closes_detrended = closes - np.mean(closes)

        # FFT
        freqs = np.fft.rfftfreq(n, d=1)  # frequency bins
        fft_vals = np.fft.rfft(closes_detrended)
        amplitudes = np.abs(fft_vals)

        # Find dominant frequency
        idx = np.argmax(amplitudes[1:]) + 1  # skip DC component
        dominant_freq = freqs[idx]
        dominant_amp = amplitudes[idx]

        # Cycle length in candles
        cycle_length = int(1 / dominant_freq) if dominant_freq > 0 else None

        # Plot spectrum
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(freqs, amplitudes, color="blue")
        ax.set_title(f"Frequency Spectrum — {symbol}")
        ax.set_xlabel("Frequency (cycles per candle)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)

        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        plt.close(fig)

        file = discord.File(buf, filename=f"{symbol}_freq.png")

        # Embed summary
        embed = discord.Embed(title=f"Frequency Analysis — {symbol}", color=discord.Color.purple())
        embed.add_field(name="Dominant Frequency", value=f"{dominant_freq:.4f} cycles/candle", inline=False)
        if cycle_length:
            embed.add_field(name="Cycle Length", value=f"~{cycle_length} candles per wave", inline=True)
        embed.add_field(name="Amplitude", value=f"{dominant_amp:.2f}", inline=True)
        embed.add_field(name="Volatility Estimate", value=f"{np.std(closes):.5f}", inline=True)

        await ctx.send(embed=embed, file=file)

    except Exception as e:
        await ctx.send(f"Error in frequency analysis: {e}")



# --------------------- heartbeat with snake overlay ---------------------
@bot.command(name="heartbeat")
async def cmd_heartbeat(ctx,
                        symbol: str = "EURUSDm",
                        period: str = "7d",
                        timeframe: str = "15m"):
    """
    Show candlestick chart with:
    - Red line = cumulative averaged heartbeat
    - Blue line = snake wiggle precise waveform
    """
    try:
        df = fetch_history_mt5(symbol, period=period, timeframe=timeframe)
        if df is None or df.empty:
            await ctx.send("No data available for the requested symbol/timeframe.")
            return

        df_plot = df.copy()
        rename_map = {
            "open": "Open", "high": "High",
            "low": "Low", "close": "Close",
            "tick_volume": "Volume"
        }
        df_plot = df_plot.rename(columns={k: v for k, v in rename_map.items() if k in df_plot.columns})

        # Compute both waveforms
        osc_cum, dom_freq_cum = compute_heartbeat_cumulative(df_plot, keep_freqs=8)
        osc_precise, dom_freq_precise = compute_heartbeat_precise(df_plot)

        # Addplots: baseline (red) + wiggle overlay (blue)
        ap = [
            mpf.make_addplot(osc_cum, panel=1, color='tab:red', ylabel='Heartbeat'),
            mpf.make_addplot(osc_precise, panel=1, color='tab:blue', secondary_y=False)
        ]

        fig, axes = mpf.plot(df_plot,
                             type='candle',
                             style='charles',
                             addplot=ap,
                             volume=False,
                             figsize=(12, 8),
                             returnfig=True,
                             panel_ratios=(3, 1),
                             tight_layout=True)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        embed = discord.Embed(
            title=f"Heartbeat Oscillator — {symbol}",
            description="Red = cumulative baseline, Blue = snake waveform",
            color=discord.Color.teal()
        )
        embed.add_field(name="Dominant freq (cumulative)", value=f"{dom_freq_cum:.6f}", inline=True)
        embed.add_field(name="Dominant freq (precise)", value=f"{dom_freq_precise:.6f}", inline=True)
        embed.add_field(name="Strength (std)", value=f"{np.std(osc_cum):.4f}", inline=True)
        embed.set_footer(text=f"Timeframe={timeframe} | Period={period}")

        file = discord.File(buf, filename=f"{symbol}_heartbeat.png")
        await ctx.send(embed=embed, file=file)

    except Exception as e:
        await ctx.send(f"Error in heartbeat: {e}")
        raise


@bot.command(name="price")
async def cmd_price(ctx, symbol: str = "EURUSDm"):
    try:
        info = ensure_symbol_selected(symbol)
        tick = mt5.symbol_info_tick(symbol)

        embed = discord.Embed(title=f"Price — {symbol}", color=discord.Color.blue())

        if tick:
            # Example: ±200 points (adjust as needed)
            buy_sl = tick.bid - 200 * info.point
            buy_tp = tick.bid + 200 * info.point
            sell_sl = tick.ask + 200 * info.point
            sell_tp = tick.ask - 200 * info.point

            embed.add_field(name="Bid (Buy)", value=f"{tick.bid:.5f}", inline=True)
            embed.add_field(
                name="SL/TP (Buy)", 
                value=f"SL: {buy_sl:.5f}\nTP: {buy_tp:.5f}", 
                inline=True
            )
            embed.add_field(name="Ask (Sell)", value=f"{tick.ask:.5f}", inline=True)
            embed.add_field(
                name="SL/TP (Sell)", 
                value=f"SL: {sell_sl:.5f}\nTP: {sell_tp:.5f}", 
                inline=True
            )
            embed.add_field(name="Spread (points)", value=str(info.spread), inline=True)
        else:
            embed.add_field(name="Error", value="No tick data", inline=False)

        await ctx.send(embed=embed)

    except Exception as e:
        await ctx.send(f"Error fetching price: {e}")


@bot.command(name="buy")
async def cmd_buy(ctx, symbol: str, lot: float, *, rest: str = ""):
    """
    Usage: !buy SYMBOL LOT sl=1.0830 tp=1.0950
    """
    try:
        sl = None; tp = None
        parts = rest.split()
        for p in parts:
            if p.startswith("sl="): sl = float(p.split("=",1)[1])
            if p.startswith("tp="): tp = float(p.split("=",1)[1])
        res = place_order(symbol, lot, "BUY", sl=sl, tp=tp, comment="Discord BUY")
        await ctx.send(f"Order result: {res}")
    except Exception as e:
        await ctx.send(f"Buy failed: {e}")



@bot.command(name="sell")
async def cmd_sell(ctx, symbol: str, lot: float, *, rest: str = ""):
    """
    Usage: !sell SYMBOL LOT sl=1.0830 tp=1.0950
    """
    try:
        sl = None; tp = None
        parts = rest.split()
        for p in parts:
            if p.startswith("sl="): sl = float(p.split("=",1)[1])
            if p.startswith("tp="): tp = float(p.split("=",1)[1])
        res = place_order( symbol, lot, "SELL", sl=sl, tp=tp, comment="Discord SELL")
        await ctx.send(f"Order result: {res}")
    except Exception as e:
        logger.exception("Error in sell command")
        await ctx.reply(f"Sell failed: {e}")

@bot.command(name="orders")
async def cmd_orders(ctx):
    try:
        pos = list_open_positions()
        if not pos:
            await ctx.send("No open positions.")
            return
        desc = []
        for p in pos:
            desc.append(f"Ticket: {p['ticket']} | {p['symbol']} {p['type']} {p['volume']} @ {p['price_open']} SL:{p['sl']} TP:{p['tp']} PnL:{p['profit']}")
        # send in chunks if long
        await ctx.send("Open positions:\n" + "\n".join(desc))
    except Exception as e:
        await ctx.send(f"Error listing orders: {e}")

@bot.command(name="account")
async def cmd_account(ctx):
    try:
        # Basic account info
        info = mt5.account_info()
        if info is None:
            await ctx.send("Failed to fetch account info.")
            return
        bal = info.balance; equity = info.equity; margin = info.margin; free = info.margin_free
        embed = discord.Embed(title="Account Info", color=discord.Color.teal())
        embed.add_field(name="Balance", value=str(bal), inline=True)
        embed.add_field(name="Equity", value=str(equity), inline=True)
        embed.add_field(name="Margin", value=str(margin), inline=True)
        embed.add_field(name="Free Margin", value=str(free), inline=True)
        # closed trades summary (last 3650 days)
        deals = closed_trades_since(days=3650)
        total_profit = sum(d["profit"] for d in deals)
        embed.add_field(name="Closed trades (count)", value=str(len(deals)), inline=False)
        embed.add_field(name="Closed trades total P/L", value=str(round(total_profit,2)), inline=False)
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"Account error: {e}")

@bot.command(name="update_order")
async def cmd_update_order(ctx, ticket: int, *, rest: str = ""):
    """
    Usage: !update_order <ticket> sl=1.0830 tp=1.0950
    Optionally pass side=BUY/SELL to check mismatch; warns if mismatch.
    """
    try:
        sl = None; tp = None; side = None
        parts = rest.split()
        for p in parts:
            if p.startswith("sl="): sl = float(p.split("=",1)[1])
            if p.startswith("tp="): tp = float(p.split("=",1)[1])
            if p.startswith("side="): side = p.split("=",1)[1].upper()
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            await ctx.send("Position not found.")
            return
        p = positions[0]
        pos_type = "BUY" if p.type == 0 else "SELL"
        if side and side != pos_type:
            await ctx.send(f"Warning: provided side {side} does not match actual position type {pos_type}. Aborting.")
            return
        res = modify_position_sl_tp(ticket, sl=sl, tp=tp)
        await ctx.send(f"Modify result: {res}")
    except Exception as e:
        await ctx.send(f"Update order failed: {e}")


@bot.command(name="auto_buy")
async def auto_buy(ctx, symbol: str, volume: float, rr: str = "1:3", buffer: float = None):
    try:
        rr_val = float(rr.split(":")[1]) / float(rr.split(":")[0])
        order_id, msg = place_order(symbol, volume, "BUY", rr_val, buffer)
        await ctx.send(msg)
    except Exception as e:
        await ctx.send(f"Error in auto_buy: {e}")

@bot.command(name="auto_sell")
async def auto_sell(ctx, symbol: str, volume: float, rr: str = "1:3", buffer: float = None):
    try:
        rr_val = float(rr.split(":")[1]) / float(rr.split(":")[0])
        order_id, msg = place_order(symbol, volume, "SELL", rr_val, buffer)
        await ctx.send(msg)
    except Exception as e:
        await ctx.send(f"Error in auto_sell: {e}")

# ===============================
# 📒 History Command
# ===============================

TRADE_LOG_FILE = "trades_history.csv"

# 🔹 Utility: log closed trade to CSV
def log_trade_to_csv(deal):
    # Map MT5 reason code -> human readable
    reason_map = {
        mt5.DEAL_REASON_SL: "SL",
        mt5.DEAL_REASON_TP: "TP",
        mt5.DEAL_REASON_EXPERT: "EA",
        mt5.DEAL_REASON_CLIENT: "Manual",
        mt5.DEAL_REASON_MARGINCALL: "Margin Call",
    }
    reason = reason_map.get(deal.reason, "Other")

    row = [
        deal.time,
        deal.symbol,
        deal.volume,
        deal.price,
        deal.profit,
        deal.swap,
        deal.commission,
        reason
    ]

    # Write header only if file is new
    write_header = not os.path.exists(TRADE_LOG_FILE)
    with open(TRADE_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["time", "symbol", "volume", "price", "profit", "swap", "commission", "reason"])
        writer.writerow(row)


@bot.command()
async def history(ctx, limit: int = 10):
    """Show last closed trades with PnL, save to CSV, and compare to open positions."""
    try:
        # --- Fetch closed trades ---
        closed = mt5.history_deals_get()
        if not closed:
            await ctx.send("⚠️ No closed trades found.")
            return

        # Log each closed deal to CSV
        for deal in closed:
            log_trade_to_csv(deal)

        # --- Read from CSV ---
        df_closed = pd.read_csv(TRADE_LOG_FILE)
        df_closed["PnL"] = df_closed["profit"] + df_closed["swap"] + df_closed["commission"]
        df_show = df_closed.tail(limit)

        # --- Open positions ---
        positions = mt5.positions_get()
        open_pnl = sum(p.profit for p in positions) if positions else 0

        # --- Build Embed ---
        embed = discord.Embed(title="📒 Trade History", color=discord.Color.blue())
        for _, row in df_show.iterrows():
            embed.add_field(
                name=f"{row['symbol']} @ {row['time']}",
                value=f"PnL: {row['PnL']:.2f} | Reason: {row['reason']}",
                inline=False
            )

        embed.add_field(name="💰 Current Open PnL", value=f"{open_pnl:.2f}", inline=False)
        embed.add_field(name="📊 Net Total", value=f"{df_closed['PnL'].sum() + open_pnl:.2f}", inline=False)

        await ctx.send(embed=embed)

    except Exception as e:
        await ctx.send(f"⚠️ Error fetching history: {e}")





@bot.command(name="auto_update_chart")
async def cmd_auto_update_chart(ctx, symbol: str = "EURUSDm", theme: str = "dark"):
    """
    Starts dual charts:
      • Static chart with indicators (sent once)
      • Lightweight auto-updating chart (last ~50 candles, no indicators, bull/bear)
    Use !close_chart to stop (admin or chart owner).
    """
    try:
        channel = ctx.channel
        sent = await AutoChartManager.start_chart(channel, symbol, ctx.author.id, theme=theme)
        if sent:
            await ctx.send("📈 Auto-updating live chart started.", delete_after=5)
    except Exception as e:
        await ctx.send("Failed to start charts: " + str(e))

@bot.command(name="close_chart")
async def cmd_close_chart(ctx, message_id: int = None):
    """
    Close a running chart. If message_id omitted, tries to close all charts the user started (owner/admin).
    """
    try:
        if message_id:
            # only owner/admin can close
            running = AutoChartManager.running.get(message_id)
            if not running:
                await ctx.send("No running chart with that message id.")
                return
            author = running.get("author_id")
            if ctx.author.id != author and not ctx.author.guild_permissions.administrator:
                await ctx.send("Only chart owner or admin can close.")
                return
            AutoChartManager.stop_chart(message_id)
            await ctx.send(f"Stopped chart {message_id}.")
        else:
            # stop all charts for which user is owner or admin
            stopped = 0
            to_stop = []
            for mid, info in list(AutoChartManager.running.items()):
                if ctx.author.id == info.get("author_id") or ctx.author.guild_permissions.administrator:
                    to_stop.append(mid)
            for mid in to_stop:
                AutoChartManager.stop_chart(mid); stopped += 1
            await ctx.send(f"Stopped {stopped} charts.")
    except Exception as e:
        await ctx.send("close_chart error: " + str(e))

last_signals = {}  # keep track to avoid duplicate alerts

@bot.command(name="signal")
async def cmd_signal(ctx, symbol: str, timeframe: str = "5m", periods: int = 200):
    #await ctx.trigger_typing()
    try:
        # same period logic as chart
        if timeframe.endswith("m"):
            period = f"{max(1, int(periods/60))}d"
        elif timeframe.endswith("h"):
            hours = int(timeframe[:-1])
            days = max(1, int((periods * hours) / 24))
            period = f"{min(365, days)}d"
        else:
            period = f"{min(730, int(periods/24))}d"

        df = fetch_history_mt5(symbol, timeframe = timeframe, period = period)
        df = compute_indicators(df)
        sig = generate_signal(df)
        embed = discord.Embed(title=f"Signal — {symbol}", color=discord.Color.red())
        embed.add_field(name="Signal", value=sig["signal"], inline=True)
        embed.add_field(name="Confidence", value=str(sig["confidence"]), inline=True)
        embed.add_field(name="Price", value=str(sig["price"]), inline=True)
        embed.add_field(name="Time", value=sig["time"], inline=False)
        embed.add_field(name="Reasons", value="\n".join(sig["reasons"]) or "-", inline=False)
        await ctx.send(embed=embed)
    except Exception as e:
        logger.exception("Error in signal command")
        await ctx.send(f"Error computing signal: {e}")




@bot.command()
async def volume(ctx, symbol: str, limit: int = 200, timeframe: str = "15m", theme: str = "light"):
    """
    Show candlestick chart + per-candle volume histogram
    with SMA and EMA overlays. Supports dark/light theme.
    Usage: !volume EURUSDm 200 15m dark
    """
    try:
        tf_min = _parse_timeframe_minutes(timeframe)
        minutes_needed = tf_min * limit
        days_needed = max(1, math.ceil(minutes_needed / (60 * 24)))
        period_str = f"{days_needed}d"

        df = fetch_history_mt5(symbol, period=period_str, timeframe=timeframe)
        if df is None or df.empty:
            await ctx.send(f"No data for {symbol} {timeframe}")
            return

        df = df.tail(limit).copy()
        df = df.rename(columns={
            "open":"Open", "high":"High", "low":"Low",
            "close":"Close", "tick_volume":"Volume"
        })

        # Indicators
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()

        apds = [
            mpf.make_addplot(df["SMA20"], color="blue", width=1.0),
            mpf.make_addplot(df["EMA50"], color="orange", width=1.0),
        ]
        
        # Theme
        if theme.lower() == "dark":
            #mpf_style = mpf.make_mpf_style(base_mpf_style=my_dark, y_on_right=True) #binance,starsandstripes,nightclouds,mike,kenan,checkers,blueskies
            #mpf.plot(df, type="candle", style=my_dark )    
            mpf_style = mpf.make_mpf_style(base_mpf_style="nightclouds",marketcolors=mpf.make_marketcolors(up="lime", down="red", inherit=True),facecolor="black", edgecolor="black", gridcolor="gray")    
        else:
            mpf_style = mpf.make_mpf_style(base_mpf_style="charles", y_on_right=True) #charles,sas,ibd,brasil

        # Plot
        fig, axes = mpf.plot(
            df,
            type="candle",
            volume=True,
            addplot=apds,
            style=mpf_style,
            returnfig=True,
            figsize=(12,7),
            datetime_format="%m-%d %H:%M"
        )

        # --- Add black borders to volume bars ---
        vol_ax = axes[2]  # axes[2] = volume subplot
        for bar in vol_ax.patches:
            bar.set_edgecolor("black")
            bar.set_linewidth(0.5)

        # Save and send
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close(fig)
        await ctx.send(file=discord.File(buf, filename=f"{symbol}_volume.png"))

    except Exception as e:
        await ctx.send(f"Error generating volume chart: {e}")


# ---------- WATCHER (integrated heartbeat + volume baselines; uses ticks properly) ----------
@tasks.loop(minutes=POLL_INTERVAL_MIN)
async def watcher_loop():
    channel_id = 1413201090102689792
    channel = bot.get_channel(channel_id)
    if not channel:
        logger.warning("No alert channel found; skipping automatic alerts this iteration.")
        return

    for pair in WATCH_LIST:
        try:
            # use a reasonably short timeframe for baseline calc (you can modify)
            timeframe = "1m"  # use 1m for heartbeat accuracy; you can change per pair if desired
            df = fetch_history_mt5(pair, period="7d", timeframe="15m")  # still used by other signals
            if df is None or df.empty:
                continue

            # Existing trading signals (unchanged)
            try:
                df_ind = compute_indicators(df.copy())
                sig = generate_signal(df_ind)
                last = last_signals.get(pair)
                if (last is None) or (last["signal"] != sig["signal"] and sig["confidence"] >= 0.4):
                    last_signals[pair] = sig
                    embed = discord.Embed(title=f"Auto Alert — {pair}", color=discord.Color.red())
                    embed.add_field(name="Signal", value=sig["signal"], inline=True)
                    embed.add_field(name="Confidence", value=str(sig["confidence"]), inline=True)
                    embed.add_field(name="Price", value=str(sig["price"]), inline=True)
                    embed.add_field(name="Time", value=sig["time"], inline=False)
                    embed.add_field(name="Reasons", value="\n".join(sig["reasons"]) or "-", inline=False)
                    img_bytes = plot_candles_with_indicators(df_ind.tail(200), pair)
                    file = discord.File(io.BytesIO(img_bytes), filename=f"{pair}_alert.png")
                    await channel.send(embed=embed, file=file)
            except Exception:
                logger.exception("Signal generation error (non-fatal); continuing watcher.")

            # HEARTBEAT: use tick_lookback_seconds derived from poll interval (sensible)
            tick_lookback_seconds = max(60, POLL_INTERVAL_MIN * 60)  # e.g. 60s or poll-interval
            hb = compute_heartbeat(pair,  tick_lookback_seconds=tick_lookback_seconds)

            base = HEARTBEAT_BASELINES.get(pair)
            gmin = HEARTBEAT_GLOBAL_MINS.get(pair, hb)

            if base is None:
                HEARTBEAT_BASELINES[pair] = hb
                HEARTBEAT_GLOBAL_MINS[pair] = hb
                logger.info(f"[{pair}] Init heartbeat baseline: {hb:.2f} bpm")
            else:
                # Spike detection (only send alerts for spikes)
                if hb > base * HEARTBEAT_SPIKE_FACTOR:
                    msg = f"⚡ Heartbeat spike on {pair}! baseline {base:.1f} → now {hb:.1f} bpm"
                    await channel.send(msg)
                    HEARTBEAT_BASELINES[pair] = hb  # update baseline upward
                elif hb < base * HEARTBEAT_DROP_FACTOR:
                    # adjust baseline downward silently
                    logger.info(f"[{pair}] Heartbeat dropped: {base:.2f} → {hb:.2f} (new baseline)")
                    HEARTBEAT_BASELINES[pair] = hb

                # update global minima
                if gmin is None or hb < gmin:
                    HEARTBEAT_GLOBAL_MINS[pair] = hb

            # VOLUME baseline: use the last N candles' tick_volume average
            try:
                vol_now = float(np.mean(df["tick_volume"].values[-3:]))
            except Exception:
                vol_now = float(np.mean(df["tick_volume"].values)) if "tick_volume" in df.columns else 0.0

            vbase = VOLUME_BASELINES.get(pair)
            if vbase is None:
                VOLUME_BASELINES[pair] = vol_now
                logger.info(f"[{pair}] Init volume baseline: {vol_now:.1f}")
            else:
                if vol_now > vbase * VOLUME_SPIKE_FACTOR:
                    await channel.send(f"📈 Volume spike on {pair}! baseline {vbase:.1f} → now {vol_now:.1f}")
                    VOLUME_BASELINES[pair] = vol_now
                elif vol_now < vbase * VOLUME_DROP_FACTOR:
                    logger.info(f"[{pair}] Volume dropped: {vbase:.1f} → {vol_now:.1f} (new baseline)")
                    VOLUME_BASELINES[pair] = vol_now

            # debug console output (always print)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {pair} hb={hb:.2f} bpm | base={HEARTBEAT_BASELINES[pair]:.2f} | vol_now={vol_now:.1f} | vbase={VOLUME_BASELINES[pair]:.1f}")

        except Exception as e:
            logger.exception(f"Error in watcher loop for {pair}: {e}")




CSV_FILE = "market_log.csv"

# --- Ensure CSV headers exist once ---
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "pair",
            "heartbeat", "hb_baseline", "hb_min",
            "volume", "vol_baseline"
        ])

def log_to_csv(pair, hb, hbase, hmin, vol_now, vbase):
    """Append one row of market data to CSV"""
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            pair,
            f"{hb:.2f}" if hb is not None else "",
            f"{hbase:.2f}" if hbase is not None else "",
            f"{hmin:.2f}" if hmin is not None else "",
            f"{vol_now:.2f}" if vol_now is not None else "",
            f"{vbase:.2f}" if vbase is not None else "",
        ])


@tasks.loop(seconds=1)  # proactive, tick-by-tick
async def realtime_watcher():
    for pair in WATCH_LIST:
        log_to_csv(pair, hb, hbase, hmin, vol_now, vbase)
        try:
            # ----- Heartbeat -----
            hb = compute_heartbeat(pair, tick_lookback_seconds=10)
            if hb is None: 
                continue

            hbase = BASELINES["heartbeat"].get(pair, {}).get("baseline")
            hmin = BASELINES["heartbeat"].get(pair, {}).get("min", hb)

            if hbase is None:
                BASELINES["heartbeat"][pair] = {"baseline": hb, "min": hb}
                save_baselines(BASELINES)
            else:
                if hb > hbase * 1.5:  # Spike
                    channel = bot.get_channel(ALERT_CHANNEL_ID)
                    if channel:
                        await channel.send(f"⚡ **Heartbeat spike** {pair}: {hbase:.1f} → {hb:.1f} bpm")
                    BASELINES["heartbeat"][pair]["baseline"] = hb
                    save_baselines(BASELINES)
                elif hb < hbase * 0.5:  # Drop
                    BASELINES["heartbeat"][pair]["baseline"] = hb
                    save_baselines(BASELINES)

                if hb < hmin:
                    BASELINES["heartbeat"][pair]["min"] = hb
                    save_baselines(BASELINES)

            # ----- Volume -----
            df = fetch_history_mt5(pair, period="1d", timeframe="1m")
            if df is None or df.empty: 
                continue
            vol_now = float(df["tick_volume"].values[-1])
            vbase = BASELINES["volume"].get(pair, {}).get("baseline")

            if vbase is None:
                BASELINES["volume"][pair] = {"baseline": vol_now}
                save_baselines(BASELINES)
            else:
                if vol_now > vbase * 1.5:  # Spike
                    channel = bot.get_channel(ALERT_CHANNEL_ID)
                    if channel:
                        await channel.send(f"📈 **Volume spike** {pair}: {vbase:.1f} → {vol_now:.1f}")
                    BASELINES["volume"][pair]["baseline"] = vol_now
                    save_baselines(BASELINES)
                elif vol_now < vbase * 0.5:  # Drop
                    BASELINES["volume"][pair]["baseline"] = vol_now
                    save_baselines(BASELINES)

            # ----- CSV Logging -----
            log_to_csv(pair, hb, hbase, hmin, vol_now, vbase)

        except Exception as e:
            logger.exception(f"Error realtime_watcher {pair}: {e}")


@realtime_watcher.before_loop
async def before_watcher():
    await bot.wait_until_ready()

# -------------------- Run --------------------
async def main():
    async with bot:
        await bot.start(TOKEN)

asyncio.run(main())
