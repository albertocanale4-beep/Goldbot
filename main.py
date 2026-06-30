"""
XAU/USD Trading Bot
- Dati real-time via Twelve Data (polling ottimizzato)
- Indicatori: RSI, MACD, Supporti/Resistenze, Fibonacci, Liquidità, Pattern candele, Multi-TF
- Alert Telegram con BUY/SELL + SL + TP
"""

import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIGURAZIONE — inserisci le tue chiavi qui
# ─────────────────────────────────────────────
TWELVE_API_KEY = "614d51e21d9b48ec9640ee12af072be0"
TELEGRAM_TOKEN = "8945641015:AAGA3ePn0s79JPJu_x9knPgv34Kgd6kxjE4"
TELEGRAM_CHAT_ID = "8559615194"

SYMBOL = "XAU/USD"
CHECK_INTERVAL = 60  # secondi tra ogni controllo (1 minuto)

# Soglia minima score per inviare alert
MIN_SCORE_BUY  = 7.5
MIN_SCORE_SELL = 7.5

# ─────────────────────────────────────────────
#  TELEGRAM
# ─────────────────────────────────────────────
def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")

# ─────────────────────────────────────────────
#  FETCH CANDELE
# ─────────────────────────────────────────────
def fetch_candles(interval: str, outputsize: int = 100) -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_API_KEY
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if "values" not in data:
            print(f"Errore fetch {interval}: {data.get('message', 'unknown')}")
            return pd.DataFrame()
        df = pd.DataFrame(data["values"])
        df = df.rename(columns={"datetime": "time", "open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"})
        for col in ["o", "h", "l", "c"]:
            df[col] = pd.to_numeric(df[col])
        if "v" in df.columns:
            df["v"] = pd.to_numeric(df["v"])
        df = df.sort_values("time").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Fetch error {interval}: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────
#  INDICATORI
# ─────────────────────────────────────────────
def calc_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)

def calc_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal
    return macd_line.iloc[-1], signal.iloc[-1], histogram.iloc[-1]

def calc_ema(series: pd.Series, period: int) -> float:
    return series.ewm(span=period, adjust=False).mean().iloc[-1]

def calc_pivot_points(df: pd.DataFrame):
    """Calcola supporti e resistenze tramite pivot points classici."""
    h = df["h"].iloc[-2]
    l = df["l"].iloc[-2]
    c = df["c"].iloc[-2]
    pivot = (h + l + c) / 3
    r1 = 2 * pivot - l
    r2 = pivot + (h - l)
    s1 = 2 * pivot - h
    s2 = pivot - (h - l)
    return {"pivot": pivot, "r1": r1, "r2": r2, "s1": s1, "s2": s2}

def calc_fibonacci(df: pd.DataFrame, lookback: int = 50):
    """Calcola livelli Fibonacci sul range degli ultimi N periodi."""
    high = df["h"].tail(lookback).max()
    low  = df["l"].tail(lookback).min()
    diff = high - low
    levels = {
        "0%":    high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50%":   high - 0.500 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
        "100%":  low
    }
    return levels, high, low

def detect_liquidity_sweep(df: pd.DataFrame) -> str:
    """
    Rileva sweep di liquidità (SMC/ICT):
    - Sweep high: la candela buca il max precedente ma chiude sotto → bearish
    - Sweep low:  la candela buca il min precedente ma chiude sopra → bullish
    """
    if len(df) < 3:
        return "none"
    prev_high = df["h"].iloc[-3]
    prev_low  = df["l"].iloc[-3]
    curr = df.iloc[-1]
    if curr["h"] > prev_high and curr["c"] < prev_high:
        return "bearish_sweep"
    if curr["l"] < prev_low and curr["c"] > prev_low:
        return "bullish_sweep"
    return "none"

def detect_pattern(df: pd.DataFrame) -> str:
    """Rileva pattern candele: Engulfing, Hammer, Shooting Star, Pin Bar, Doji."""
    if len(df) < 2:
        return "none"
    c1 = df.iloc[-2]
    c2 = df.iloc[-1]
    body2 = abs(c2["c"] - c2["o"])
    range2 = c2["h"] - c2["l"]
    upper_wick = c2["h"] - max(c2["c"], c2["o"])
    lower_wick = min(c2["c"], c2["o"]) - c2["l"]

    # Doji
    if range2 > 0 and body2 / range2 < 0.1:
        return "doji"

    # Hammer (bullish)
    if lower_wick > 2 * body2 and upper_wick < body2 and c2["c"] > c2["o"]:
        return "hammer"

    # Shooting Star (bearish)
    if upper_wick > 2 * body2 and lower_wick < body2 and c2["c"] < c2["o"]:
        return "shooting_star"

    # Pin Bar bullish
    if lower_wick > 2 * body2 and lower_wick > upper_wick:
        return "pin_bar_bullish"

    # Pin Bar bearish
    if upper_wick > 2 * body2 and upper_wick > lower_wick:
        return "pin_bar_bearish"

    # Bullish Engulfing
    if (c1["c"] < c1["o"] and c2["c"] > c2["o"] and
            c2["o"] < c1["c"] and c2["c"] > c1["o"]):
        return "bullish_engulfing"

    # Bearish Engulfing
    if (c1["c"] > c1["o"] and c2["c"] < c2["o"] and
            c2["o"] > c1["c"] and c2["c"] < c1["o"]):
        return "bearish_engulfing"

    return "none"

# ─────────────────────────────────────────────
#  CALCOLO SCORE
# ─────────────────────────────────────────────
def compute_score(df_m1, df_m5, df_m15, df_h1, df_h4):
    """
    Calcola score BUY e SELL in base ai pesi definiti.
    Ritorna (score_buy, score_sell, dettaglio)
    """
    score_buy  = 0.0
    score_sell = 0.0
    details    = []
    price      = df_m5["c"].iloc[-1]

    # ── RSI (peso alto = 2.0) ──────────────────
    rsi = calc_rsi(df_m5["c"])
    if rsi < 35:
        score_buy += 2.0
        details.append(f"RSI {rsi} ipervenduto ✅ BUY +2")
    elif rsi > 65:
        score_sell += 2.0
        details.append(f"RSI {rsi} ipercomprato ✅ SELL +2")
    else:
        details.append(f"RSI {rsi} neutro")

    # ── MACD (peso alto = 2.0) ─────────────────
    macd, signal, hist = calc_macd(df_m5["c"])
    _, _, prev_hist    = calc_macd(df_m5["c"].iloc[:-1])
    if hist > 0 and prev_hist <= 0:
        score_buy += 2.0
        details.append(f"MACD crossover bullish ✅ BUY +2")
    elif hist < 0 and prev_hist >= 0:
        score_sell += 2.0
        details.append(f"MACD crossover bearish ✅ SELL +2")
    else:
        details.append(f"MACD hist {round(hist,2)} no crossover")

    # ── SUPPORTI & RESISTENZE (peso alto = 2.0) ─
    pivots = calc_pivot_points(df_h1)
    near_support    = abs(price - pivots["s1"]) / price < 0.003 or abs(price - pivots["s2"]) / price < 0.003
    near_resistance = abs(price - pivots["r1"]) / price < 0.003 or abs(price - pivots["r2"]) / price < 0.003
    if near_support:
        score_buy += 2.0
        details.append(f"Prezzo vicino a supporto (S1={round(pivots['s1'],2)}) ✅ BUY +2")
    elif near_resistance:
        score_sell += 2.0
        details.append(f"Prezzo vicino a resistenza (R1={round(pivots['r1'],2)}) ✅ SELL +2")
    else:
        details.append(f"Pivot: P={round(pivots['pivot'],2)} S1={round(pivots['s1'],2)} R1={round(pivots['r1'],2)}")

    # ── FIBONACCI (peso medio = 1.0) ───────────
    fib_levels, fib_high, fib_low = calc_fibonacci(df_h4)
    for label in ["61.8%", "78.6%", "100%"]:
        level = fib_levels[label]
        if abs(price - level) / price < 0.002:
            if price < (fib_high + fib_low) / 2:
                score_buy += 1.0
                details.append(f"Fibonacci {label} ({round(level,2)}) supporto ✅ BUY +1")
            else:
                score_sell += 1.0
                details.append(f"Fibonacci {label} ({round(level,2)}) resistenza ✅ SELL +1")
            break

    # ── PRESE DI LIQUIDITÀ (peso molto alto = 3.0) ─
    sweep = detect_liquidity_sweep(df_m5)
    if sweep == "bullish_sweep":
        score_buy += 3.0
        details.append(f"Sweep LOW (SMC) → inversione bullish ✅ BUY +3")
    elif sweep == "bearish_sweep":
        score_sell += 3.0
        details.append(f"Sweep HIGH (SMC) → inversione bearish ✅ SELL +3")
    else:
        details.append("Nessun sweep di liquidità")

    # ── PATTERN CANDELE M5 (peso medio = 1.5) ──
    pattern = detect_pattern(df_m5)
    bullish_patterns = ["hammer", "pin_bar_bullish", "bullish_engulfing"]
    bearish_patterns = ["shooting_star", "pin_bar_bearish", "bearish_engulfing"]
    if pattern in bullish_patterns:
        score_buy += 1.5
        details.append(f"Pattern {pattern} ✅ BUY +1.5")
    elif pattern in bearish_patterns:
        score_sell += 1.5
        details.append(f"Pattern {pattern} ✅ SELL +1.5")
    elif pattern == "doji":
        details.append("Doji (indecisione)")
    else:
        details.append("Nessun pattern rilevante")

    # ── MULTI-TIMEFRAME EMA20/50 (peso alto = 2.0) ─
    mtf_score_buy  = 0
    mtf_score_sell = 0
    for label, df_tf in [("M15", df_m15), ("H1", df_h1), ("H4", df_h4)]:
        if df_tf.empty:
            continue
        ema20 = calc_ema(df_tf["c"], 20)
        ema50 = calc_ema(df_tf["c"], 50)
        if ema20 > ema50:
            mtf_score_buy += 1
        elif ema20 < ema50:
            mtf_score_sell += 1
    if mtf_score_buy >= 2:
        score_buy += 2.0
        details.append(f"Multi-TF bullish ({mtf_score_buy}/3 timeframe) ✅ BUY +2")
    elif mtf_score_sell >= 2:
        score_sell += 2.0
        details.append(f"Multi-TF bearish ({mtf_score_sell}/3 timeframe) ✅ SELL +2")
    else:
        details.append(f"Multi-TF misto (buy={mtf_score_buy} sell={mtf_score_sell})")

    return score_buy, score_sell, details, price, pivots, rsi, mtf_score_buy, mtf_score_sell

# ─────────────────────────────────────────────
#  CALCOLO SL / TP
# ─────────────────────────────────────────────
def calc_sl_tp(direction: str, price: float, df_m5: pd.DataFrame, atr_mult_sl=1.5, rr=2.0):
    """
    SL basato su ATR (14 periodi).
    TP calcolato con risk/reward 1:2.
    """
    high_low = df_m5["h"] - df_m5["l"]
    atr = high_low.rolling(14).mean().iloc[-1]
    sl_dist = atr * atr_mult_sl
    tp_dist = sl_dist * rr

    if direction == "BUY":
        sl = round(price - sl_dist, 2)
        tp = round(price + tp_dist, 2)
    else:
        sl = round(price + sl_dist, 2)
        tp = round(price - tp_dist, 2)
    return sl, tp, round(atr, 2)

# ─────────────────────────────────────────────
#  FORMATTA MESSAGGIO TELEGRAM
# ─────────────────────────────────────────────
def format_message(direction: str, score: float, price: float, sl: float, tp: float,
                   atr: float, rsi: float, details: list) -> str:
    emoji = "🟢" if direction == "BUY" else "🔴"
    now = datetime.now().strftime("%H:%M UTC")
    detail_text = "\n".join([f"  • {d}" for d in details])
    # Prezzo massimo di ingresso (0.5 ATR dall'entry)
    if direction == "BUY":
        entry_limit = round(price + atr * 0.5, 2)
        entry_label = f"Max `{entry_limit}` — sopra questo salta il trade"
    else:
        entry_limit = round(price - atr * 0.5, 2)
        entry_label = f"Min `{entry_limit}` — sotto questo salta il trade"

    return (
        f"{emoji} *{direction} XAU/USD* — {now}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *Prezzo:* `{price}`\n"
        f"⏱ *Entra entro:* {entry_label}\n"
        f"🎯 *Score:* `{score:.1f} / 14.5`\n"
        f"🛑 *Stop Loss:* `{sl}`\n"
        f"✅ *Take Profit:* `{tp}`\n"
        f"📊 *ATR:* `{atr}` | *RSI:* `{rsi}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*Dettaglio segnali:*\n{detail_text}"
    )

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
last_signal = {"direction": None, "time": 0}
SIGNAL_COOLDOWN = 300  # non ripetere lo stesso segnale per 5 minuti

# Trailing stop tracking
active_trade = {
    "direction": None,
    "entry": None,
    "sl": None,
    "tp": None,
    "atr": None,
    "trail_notified": []  # livelli già notificati
}



def check_trailing_stop(price: float):
    """
    Controlla se il prezzo si è mosso abbastanza da spostare lo SL.
    Notifica su Telegram i livelli di trailing.
    """
    if not active_trade["direction"]:
        return

    direction = active_trade["direction"]
    entry     = active_trade["entry"]
    sl        = active_trade["sl"]
    tp        = active_trade["tp"]
    atr       = active_trade["atr"]

    if direction == "BUY":
        profit_pts = price - entry
        # Livelli di trailing: ogni 0.5 ATR di profitto
        trail_levels = [
            (atr * 0.5,  entry,        "Breakeven raggiunto! Sposta SL a entry"),
            (atr * 1.0,  entry + atr * 0.5, "Sposta SL — profitto parziale garantito"),
            (atr * 1.5,  entry + atr * 1.0, "Sposta SL — ottima posizione!"),
        ]
        for threshold, new_sl, msg in trail_levels:
            key = f"buy_{threshold}"
            if profit_pts >= threshold and key not in active_trade["trail_notified"]:
                active_trade["sl"] = new_sl
                active_trade["trail_notified"].append(key)
                new_sl_r = round(new_sl, 2)
                text = "📈 *TRAILING STOP BUY*\n"
                text += f"Prezzo: `{round(price,2)}` | Entry: `{round(entry,2)}`\n"
                text += f"✅ {msg}\n"
                text += f"🛑 *Nuovo SL: `{new_sl_r}`*"
                send_telegram(text)

    elif direction == "SELL":
        profit_pts = entry - price
        trail_levels = [
            (atr * 0.5,  entry,        "Breakeven raggiunto! Sposta SL a entry"),
            (atr * 1.0,  entry - atr * 0.5, "Sposta SL — profitto parziale garantito"),
            (atr * 1.5,  entry - atr * 1.0, "Sposta SL — ottima posizione!"),
        ]
        for threshold, new_sl, msg in trail_levels:
            key = f"sell_{threshold}"
            if profit_pts >= threshold and key not in active_trade["trail_notified"]:
                active_trade["sl"] = new_sl
                active_trade["trail_notified"].append(key)
                new_sl_r = round(new_sl, 2)
                text = "📉 *TRAILING STOP SELL*\n"
                text += f"Prezzo: `{round(price,2)}` | Entry: `{round(entry,2)}`\n"
                text += f"✅ {msg}\n"
                text += f"🛑 *Nuovo SL: `{new_sl_r}`*"
                send_telegram(text)

    # Controlla se TP raggiunto
    if direction == "BUY" and price >= tp:
        send_telegram("🎯 *TARGET RAGGIUNTO! BUY XAU/USD*\nEntry: `" + str(round(entry,2)) + "` TP: `" + str(round(tp,2)) + "`\n+" + str(round(tp-entry,2)) + " punti")
        active_trade["direction"] = None
        active_trade["trail_notified"] = []

    elif direction == "SELL" and price <= tp:
        send_telegram("🎯 *TARGET RAGGIUNTO! SELL XAU/USD*\nEntry: `" + str(round(entry,2)) + "` TP: `" + str(round(tp,2)) + "`\n+" + str(round(entry-tp,2)) + " punti")
        active_trade["direction"] = None
        active_trade["trail_notified"] = []

    # Controlla se SL raggiunto
    if direction == "BUY" and price <= active_trade["sl"]:
        send_telegram('🛑 *STOP LOSS BUY XAU/USD*\nEntry: `' + str(round(entry,2)) + '` SL: `' + str(round(active_trade["sl"],2)) + '`\n' + str(round(active_trade["sl"]-entry,2)) + ' punti')
        active_trade["direction"] = None
        active_trade["trail_notified"] = []

    elif direction == "SELL" and price >= active_trade["sl"]:
        send_telegram('🛑 *STOP LOSS SELL XAU/USD*\nEntry: `' + str(round(entry,2)) + '` SL: `' + str(round(active_trade["sl"],2)) + '`\n' + str(round(entry-active_trade["sl"],2)) + ' punti')
        active_trade["direction"] = None
        active_trade["trail_notified"] = []

def is_trading_session() -> bool:
    """Blocca segnali fuori dalle sessioni attive (07:00-22:00 UTC)."""
    hour = datetime.now().hour  # UTC approximation on Railway
    return 7 <= hour < 22

def run():
    print(f"🚀 Bot XAU/USD avviato — controllo ogni {CHECK_INTERVAL}s")
    send_telegram("🚀 *Bot XAU/USD avviato!*\nMonitoraggio in corso su M1/M5/M15/H1/H4...")

    while True:
        try:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Fetch candele...")

            df_m1  = fetch_candles("1min",  outputsize=50)
            time.sleep(1)
            df_m5  = fetch_candles("5min",  outputsize=100)
            time.sleep(1)
            df_m15 = fetch_candles("15min", outputsize=100)
            time.sleep(1)
            df_h1  = fetch_candles("1h",    outputsize=100)
            time.sleep(1)
            df_h4  = fetch_candles("4h",    outputsize=100)

            if df_m5.empty:
                print("Dati M5 non disponibili, riprovo...")
                time.sleep(CHECK_INTERVAL)
                continue

            score_buy, score_sell, details, price, pivots, rsi, mtf_score_buy, mtf_score_sell = compute_score(
                df_m1, df_m5, df_m15, df_h1, df_h4
            )

            print(f"  Price: {price} | Score BUY: {score_buy:.1f} | Score SELL: {score_sell:.1f}")

            now_ts = time.time()
            direction = None

            if score_buy >= MIN_SCORE_BUY and score_buy > score_sell:
                direction = "BUY"
            elif score_sell >= MIN_SCORE_SELL and score_sell > score_buy:
                direction = "SELL"
            
            # Richiedi conferma MTF: blocca se trend opposto su tutti e 3 i TF
            if direction == "BUY" and mtf_score_sell == 3:
                print(f"  BUY bloccato: Multi-TF bearish 3/3")
                direction = None
            elif direction == "SELL" and mtf_score_buy == 3:
                print(f"  SELL bloccato: Multi-TF bullish 3/3")
                direction = None

            # Blocca segnali fuori sessione (22:00-07:00 UTC)
            if direction and not is_trading_session():
                print(f"  Fuori sessione di trading, segnale {direction} ignorato")
                direction = None

            if direction:
                # Evita segnali duplicati ravvicinati
                same_dir   = last_signal["direction"] == direction
                too_recent = (now_ts - last_signal["time"]) < SIGNAL_COOLDOWN
                if not (same_dir and too_recent):
                    score = score_buy if direction == "BUY" else score_sell
                    sl, tp, atr = calc_sl_tp(direction, price, df_m5)
                    msg = format_message(direction, score, price, sl, tp, atr, rsi, details)
                    send_telegram(msg)
                    last_signal["direction"] = direction
                    last_signal["time"]      = now_ts
                    # Attiva trailing stop per questo trade
                    active_trade["direction"] = direction
                    active_trade["entry"]     = price
                    active_trade["sl"]        = sl
                    active_trade["tp"]        = tp
                    active_trade["atr"]       = atr
                    active_trade["trail_notified"] = []
                    print(f"  ➜ Alert {direction} inviato! Score={score:.1f}")
                else:
                    print(f"  Segnale {direction} già inviato di recente, skip.")
            else:
                print(f"  Nessun segnale (score sotto soglia)")

            # Controlla trailing stop se c'è un trade attivo
            if active_trade["direction"] and not df_m1.empty:
                current_price = df_m1["c"].iloc[-1]
                check_trailing_stop(current_price)

        except Exception as e:
            print(f"Errore loop: {e}")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    run()