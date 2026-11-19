"""Pakistan Finance Predictor - single-file Flask app for Vercel.

Loads the trained RNN / LSTM / GRU models from your (private) Hugging Face repo,
pulls recent prices, rebuilds the training features and predicts the next
trading day's return.

Environment variables (set in Vercel -> Project -> Settings -> Environment Variables):
    HF_TOKEN   Hugging Face READ token (needed because the repo is private)
    HF_REPO    optional, default "khalilahmed0062/pakistan-finance-prediction-lab"
"""
import io
import json
import os
import sys
import time
import types
import zipfile
from html.parser import HTMLParser

import h5py
import numpy as np
import pandas as pd
import requests
from flask import Flask, Response, jsonify, request

# --- joblib can unpickle the saved StandardScalers without installing scikit-learn
try:  # pragma: no cover
    import sklearn  # noqa: F401
except ImportError:
    class _StandardScaler:
        def __setstate__(self, state):
            self.__dict__.update(state)

    for _name in ("sklearn", "sklearn.preprocessing", "sklearn.preprocessing._data"):
        sys.modules[_name] = types.ModuleType(_name)
    sys.modules["sklearn.preprocessing._data"].StandardScaler = _StandardScaler
    sys.modules["sklearn.preprocessing"].StandardScaler = _StandardScaler
import joblib  # noqa: E402

app = Flask(__name__)

HF_REPO = os.environ.get("HF_REPO", "khalilahmed0062/pakistan-finance-prediction-lab")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
LOCAL_ARTIFACTS = os.environ.get("LOCAL_ARTIFACTS", "")  # only used for local testing
CACHE_DIR = "/tmp/pf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

ASSETS = ["OGDC", "HBL", "MEBL", "ENGRO", "KSE100"]
MODELS = ["RNN", "LSTM", "GRU"]
KSE_URL = (
    "https://raw.githubusercontent.com/MuHamzaSheikh/PSXpert/main/data/"
    "kse100_index_history_2002_2026.csv"
)
PSX_URL = "https://dps.psx.com.pk/historical"
PSX_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Referer": "https://dps.psx.com.pk/",
    "X-Requested-With": "XMLHttpRequest",
}

_models = {}      # (asset, model) -> layers
_assets = {}      # asset -> {"config":..., "scaler":...}
_prices = {}      # asset -> (timestamp, DataFrame)
PRICE_TTL = 6 * 3600


# ----------------------------------------------------------------------------
# Hugging Face artifact loading
# ----------------------------------------------------------------------------
def get_artifact(asset, filename):
    """Return bytes of deployment_artifacts/<asset>/<filename> (cached in /tmp)."""
    if LOCAL_ARTIFACTS:
        with open(os.path.join(LOCAL_ARTIFACTS, asset, filename), "rb") as f:
            return f.read()
    local = os.path.join(CACHE_DIR, asset, filename)
    if os.path.exists(local):
        with open(local, "rb") as f:
            return f.read()
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/deployment_artifacts/{asset}/{filename}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code in (401, 403, 404):
        raise RuntimeError(
            f"Could not download {asset}/{filename} from Hugging Face "
            f"(HTTP {r.status_code}). Check HF_TOKEN / HF_REPO."
        )
    r.raise_for_status()
    os.makedirs(os.path.dirname(local), exist_ok=True)
    with open(local, "wb") as f:
        f.write(r.content)
    return r.content


def load_keras_weights(raw):
    """Read layer specs + weights out of a .keras file without TensorFlow."""
    z = zipfile.ZipFile(io.BytesIO(raw))
    cfg = json.loads(z.read("config.json"))
    h5 = h5py.File(io.BytesIO(z.read("model.weights.h5")), "r")
    layers = []
    seen = {}  # Keras 3 stores weights under layers/<type>, layers/<type>_1, ... by order
    snake = {"SimpleRNN": "simple_rnn", "LSTM": "lstm", "GRU": "gru", "Dense": "dense",
             "Dropout": "dropout"}
    for layer in cfg["config"]["layers"]:
        kind = layer["class_name"]
        if kind == "InputLayer":
            continue
        if kind not in snake:
            raise ValueError(f"Unsupported layer: {kind}")
        idx = seen.get(kind, 0)
        seen[kind] = idx + 1
        if kind == "Dropout":
            continue
        name = snake[kind] if idx == 0 else f"{snake[kind]}_{idx}"
        group = h5[f"layers/{name}"]
        vars_group = group["cell/vars"] if kind != "Dense" else group["vars"]
        weights = [np.array(vars_group[str(i)], dtype=np.float32) for i in range(len(vars_group))]
        layers.append((kind, layer["config"].get("activation"), weights))
    return layers


def get_asset(asset):
    if asset not in _assets:
        config = json.loads(get_artifact(asset, "config.json"))
        fs = joblib.load(io.BytesIO(get_artifact(asset, "feature_scaler.pkl")))
        ts = joblib.load(io.BytesIO(get_artifact(asset, "target_scaler.pkl")))
        _assets[asset] = {
            "config": config,
            "f_mean": np.asarray(fs.mean_, dtype=np.float64),
            "f_scale": np.asarray(fs.scale_, dtype=np.float64),
            "t_mean": float(np.ravel(ts.mean_)[0]),
            "t_scale": float(np.ravel(ts.scale_)[0]),
        }
    return _assets[asset]


def get_model(asset, model):
    key = (asset, model)
    if key not in _models:
        _models[key] = load_keras_weights(get_artifact(asset, f"{model}.keras"))
    return _models[key]


# ----------------------------------------------------------------------------
# NumPy forward pass (matches Keras SimpleRNN / LSTM / GRU(reset_after) / Dense)
# ----------------------------------------------------------------------------
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _simple_rnn(x, w):
    k, r, b = w
    h = np.zeros((x.shape[0], r.shape[0]), np.float32)
    for t in range(x.shape[1]):
        h = np.tanh(x[:, t] @ k + h @ r + b)
    return h


def _lstm(x, w):
    k, r, b = w
    u = r.shape[0]
    h = np.zeros((x.shape[0], u), np.float32)
    c = np.zeros_like(h)
    for t in range(x.shape[1]):
        z = x[:, t] @ k + h @ r + b
        i, f, g, o = z[:, :u], z[:, u:2*u], z[:, 2*u:3*u], z[:, 3*u:]
        c = _sig(f) * c + _sig(i) * np.tanh(g)
        h = _sig(o) * np.tanh(c)
    return h


def _gru(x, w):
    k, r, b = w
    u = r.shape[0]
    h = np.zeros((x.shape[0], u), np.float32)
    for t in range(x.shape[1]):
        xz = x[:, t] @ k + b[0]
        hz = h @ r + b[1]
        z = _sig(xz[:, :u] + hz[:, :u])
        rg = _sig(xz[:, u:2*u] + hz[:, u:2*u])
        hh = np.tanh(xz[:, 2*u:] + rg * hz[:, 2*u:])
        h = z * h + (1.0 - z) * hh
    return h


def forward(layers, x):
    out = x.astype(np.float32)
    for kind, act, w in layers:
        if kind == "SimpleRNN":
            out = _simple_rnn(out, w)
        elif kind == "LSTM":
            out = _lstm(out, w)
        elif kind == "GRU":
            out = _gru(out, w)
        else:
            out = out @ w[0] + w[1]
            if act == "relu":
                out = np.maximum(out, 0.0)
    return out


# ----------------------------------------------------------------------------
# Price data
# ----------------------------------------------------------------------------
class _TableParser(HTMLParser):
    """Parse the first <table> into (headers, rows) using only the stdlib."""

    def __init__(self):
        super().__init__()
        self.headers, self.rows = [], []
        self._in_table = self._done = False
        self._cell = None
        self._row = None

    def handle_starttag(self, tag, attrs):
        if self._done:
            return
        if tag == "table":
            self._in_table = True
        elif self._in_table and tag == "tr":
            self._row = []
        elif self._in_table and tag in ("th", "td"):
            self._cell = []

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if self._done:
            return
        if tag in ("th", "td") and self._cell is not None:
            text = "".join(self._cell).strip()
            if tag == "th":
                self.headers.append(text)
            elif self._row is not None:
                self._row.append(text)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None
        elif tag == "table":
            self._done = True


def parse_psx_html(html):
    p = _TableParser()
    p.feed(html)
    names = [h.strip().lower() for h in p.headers]
    df = pd.DataFrame([r[:len(names)] for r in p.rows], columns=names[:max(len(r) for r in p.rows)]
                      if p.rows else names)
    if df.empty:
        raise RuntimeError("PSX returned no price table")
    out = pd.DataFrame({"date": pd.to_datetime(df["date"], errors="coerce", format="mixed")})
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
    out = out.dropna().drop_duplicates("date", keep="last").sort_values("date")
    bad = (
        (out["high"] < out[["open", "close"]].max(axis=1))
        | (out["low"] > out[["open", "close"]].min(axis=1))
        | (out["high"] < out["low"])
        | (out["close"] <= 0)
    )
    return out[~bad].reset_index(drop=True)


def fetch_prices(asset):
    cached = _prices.get(asset)
    if cached and time.time() - cached[0] < PRICE_TTL:
        return cached[1]
    if asset == "KSE100":
        df = pd.read_csv(KSE_URL, usecols=["date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna().drop_duplicates("date", keep="last").sort_values("date")
        df = df[df["close"] > 0].reset_index(drop=True)
    else:
        r = requests.post(PSX_URL, data={"symbol": asset}, headers=PSX_HEADERS, timeout=30)
        r.raise_for_status()
        df = parse_psx_html(r.text)
    df = df.tail(400).reset_index(drop=True)
    _prices[asset] = (time.time(), df)
    return df


# ----------------------------------------------------------------------------
# Features (identical to the training notebook)
# ----------------------------------------------------------------------------
def _rsi(close):
    ch = close.diff()
    gain = ch.clip(lower=0).rolling(14).mean()
    loss = (-ch.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi[(loss == 0) & (gain > 0)] = 100
    rsi[(loss == 0) & (gain == 0)] = 50
    return rsi


def add_features(df, asset):
    df = df.copy()
    df["daily_return"] = df["close"].pct_change()
    df["sma20_distance"] = df["close"] / df["close"].rolling(20).mean() - 1
    df["sma50_distance"] = df["close"] / df["close"].rolling(50).mean() - 1
    df["volatility20"] = df["daily_return"].rolling(20).std()
    df["rsi14"] = _rsi(df["close"])
    if asset == "KSE100":
        df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
        df["momentum_10"] = df["close"] / df["close"].shift(10) - 1
        df["volatility60"] = df["daily_return"].rolling(60).std()
    else:
        prev = df["close"].shift(1)
        df["overnight_gap"] = df["open"] / prev - 1
        df["intraday_return"] = df["close"] / df["open"] - 1
        df["high_low_range"] = (df["high"] - df["low"]) / prev
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    return df


def predict(asset, model_name):
    meta = get_asset(asset)
    cfg = meta["config"]
    cols, lookback = cfg["feature_columns"], int(cfg["lookback"])

    prices = fetch_prices(asset)
    feats = add_features(prices, asset).replace([np.inf, -np.inf], np.nan)
    feats = feats.dropna(subset=cols).reset_index(drop=True)
    if len(feats) < lookback:
        raise RuntimeError(f"Not enough price history for {asset} ({len(feats)} rows).")

    window = feats[cols].values[-lookback:]
    x = ((window - meta["f_mean"]) / meta["f_scale"]).astype(np.float32)[None]

    preds = {}
    for m in MODELS:
        y = float(forward(get_model(asset, m), x).reshape(-1)[0])
        preds[m] = y * meta["t_scale"] + meta["t_mean"]

    ret = preds[model_name]
    last_close = float(feats["close"].iloc[-1])
    return {
        "asset": asset,
        "model": model_name,
        "as_of_date": feats["date"].iloc[-1].strftime("%Y-%m-%d"),
        "last_close": round(last_close, 2),
        "predicted_return_pct": round(ret * 100, 4),
        "predicted_next_close": round(last_close * (1 + ret), 2),
        "direction": "UP" if ret > 0 else "DOWN",
        "all_models_pct": {m: round(v * 100, 4) for m, v in preds.items()},
        "recommended_model": cfg.get("selected_model"),
    }


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.route("/api/predict")
def api_predict():
    asset = request.args.get("asset", "OGDC").upper()
    model = request.args.get("model", "LSTM").upper()
    if asset not in ASSETS:
        return jsonify(error=f"asset must be one of {ASSETS}"), 400
    if model not in MODELS:
        return jsonify(error=f"model must be one of {MODELS}"), 400
    try:
        return jsonify(predict(asset, model))
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pakistan Finance Predictor</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--text:#1c2430;--muted:#6b7686;--accent:#0b6e4f;--up:#0b8a4a;--down:#c0392b;--border:#e1e5ea}
@media(prefers-color-scheme:dark){:root{--bg:#12161c;--card:#1b212a;--text:#e8ecf1;--muted:#97a3b3;--accent:#3ecf8e;--up:#3ecf8e;--down:#ff6b5e;--border:#2a323d}}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--text)}
main{max-width:560px;margin:0 auto;padding:24px 16px 48px}h1{font-size:1.4rem;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 20px;font-size:.92rem}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px}
label{display:block;font-size:.8rem;color:var(--muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:.04em}
select{width:100%;padding:11px;font-size:1rem;border-radius:10px;border:1px solid var(--border);background:var(--bg);color:var(--text)}
.models{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:16px}
.models button{padding:11px 0;font-size:.95rem;border-radius:10px;border:1px solid var(--border);background:var(--bg);color:var(--text);cursor:pointer}
.models button.active{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
.go{width:100%;margin-top:16px;padding:13px;font-size:1rem;font-weight:600;border:0;border-radius:10px;background:var(--accent);color:#fff;cursor:pointer}
.go:disabled{opacity:.6;cursor:wait}.big{font-size:2.2rem;font-weight:700;margin:4px 0}
.up{color:var(--up)}.down{color:var(--down)}
.row{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border);font-size:.95rem}
.row:last-child{border-bottom:0}.muted{color:var(--muted)}.note{font-size:.8rem;color:var(--muted);line-height:1.5}.err{color:var(--down)}
</style></head><body><main>
<h1>Pakistan Finance Predictor</h1>
<p class="sub">Next-trading-day return forecast for PSX stocks and the KSE-100 using RNN, LSTM and GRU models.</p>
<div class="card">
<label for="asset">Stock / Index</label>
<select id="asset">
<option value="OGDC">OGDC - Oil &amp; Gas Development Co.</option>
<option value="HBL">HBL - Habib Bank Ltd.</option>
<option value="MEBL">MEBL - Meezan Bank Ltd.</option>
<option value="ENGRO">ENGRO - Engro Corporation</option>
<option value="KSE100">KSE-100 Index</option></select>
<div class="models" id="models">
<button data-model="RNN">RNN</button><button data-model="LSTM" class="active">LSTM</button><button data-model="GRU">GRU</button></div>
<button class="go" id="go">Predict</button></div>
<div class="card" id="result" hidden></div>
<p class="note">Educational project, not financial advice. The first request after a pause can take a few seconds while models load.</p>
</main>
<script>
let model="LSTM";const $=i=>document.getElementById(i);
document.querySelectorAll("#models button").forEach(b=>{b.onclick=()=>{
document.querySelectorAll("#models button").forEach(x=>x.classList.remove("active"));
b.classList.add("active");model=b.dataset.model}});
$("go").onclick=async()=>{const btn=$("go"),box=$("result");
btn.disabled=true;btn.textContent="Predicting...";box.hidden=false;box.textContent="Loading...";
try{const r=await fetch(`/api/predict?asset=${$("asset").value}&model=${model}`);const d=await r.json();
if(!r.ok)throw new Error(d.error||"Request failed");
const up=d.direction==="UP",cls=up?"up":"down",sign=d.predicted_return_pct>0?"+":"";
const others=Object.entries(d.all_models_pct).map(([m,v])=>`<div class="row"><span>${m}${m===d.recommended_model?" (best on validation)":""}</span><span>${v>0?"+":""}${v}%</span></div>`).join("");
box.innerHTML=`<div class="muted">${d.asset} - ${d.model} model</div>
<div class="big ${cls}">${up?"&#9650;":"&#9660;"} ${sign}${d.predicted_return_pct}%</div>
<div class="row"><span class="muted">Data as of</span><span>${d.as_of_date}</span></div>
<div class="row"><span class="muted">Last close</span><span>${d.last_close.toLocaleString()}</span></div>
<div class="row"><span class="muted">Implied next close</span><span>${d.predicted_next_close.toLocaleString()}</span></div>
<div style="margin-top:14px" class="muted">All three models</div>${others}`;
}catch(e){box.innerHTML=`<span class="err">Error: ${e.message}</span>`}
finally{btn.disabled=false;btn.textContent="Predict"}};
</script></body></html>"""


if __name__ == "__main__":
    app.run(debug=True)
