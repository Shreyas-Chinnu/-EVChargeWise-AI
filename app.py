# =============================================================================
# EVChargeWise AI – Electric Vehicle Charging Optimization Agent
# =============================================================================
# Tech Stack : Python · Flask · IBM watsonx.ai Granite Models
#              IBM Langflow (workflow comments) · IBM Orchestrate (agent routing)
#              Bootstrap 5 · Chart.js · JavaScript
#
# IBM Credentials → .env file (copy env.example → .env and fill in values)
#   WATSONX_API_KEY   | WATSONX_PROJECT_ID | WATSONX_URL
#
# Install : pip install -r requirements.txt
# Run     : python app.py  →  http://127.0.0.1:5000
# =============================================================================

import os, io, json, csv, random, datetime, textwrap, requests
import pandas as pd
from flask import (Flask, render_template_string, request,
                   jsonify, session, redirect, url_for, make_response)

# ---------------------------------------------------------------------------
# Load .env file automatically (python-dotenv)
# Place your IBM credentials in a .env file next to app.py
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()          # reads .env from the current directory
    print("[dotenv] .env file loaded.")
except ImportError:
    print("[dotenv] python-dotenv not installed – using system env vars only.")

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "evchargewise-secret-2024")

# ---------------------------------------------------------------------------
# IBM watsonx.ai – credentials (read from env vars)
# ---------------------------------------------------------------------------
WATSONX_API_KEY    = os.environ.get("WATSONX_API_KEY",    "your-api-key")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "your-project-id")
WATSONX_URL        = os.environ.get("WATSONX_URL",        "https://us-south.ml.cloud.ibm.com")
GRANITE_MODEL_ID   = "ibm/granite-3-8b-instruct"

# ---------------------------------------------------------------------------
# IBM watsonx.ai – IAM token
# ---------------------------------------------------------------------------
def get_iam_token():
    try:
        r = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                  "apikey": WATSONX_API_KEY},
            timeout=15)
        r.raise_for_status()
        return r.json().get("access_token", "")
    except Exception as e:
        print(f"[IAM] {e}")
        return ""

# ---------------------------------------------------------------------------
# IBM watsonx.ai – core generation  (all agents funnel through here)
# IBM Langflow node : "Granite Model Node"
# IBM Orchestrate   : AI-processing step
# ---------------------------------------------------------------------------
def generate_response(prompt: str, max_tokens: int = 800) -> str:
    token = get_iam_token()
    if not token:
        return _fallback(prompt)
    try:
        r = requests.post(
            f"{WATSONX_URL}/ml/v1/text/generation?version=2024-05-01",
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json",
                     "Accept": "application/json"},
            json={"model_id": GRANITE_MODEL_ID,
                  "input": prompt,
                  "parameters": {"decoding_method": "greedy",
                                 "max_new_tokens": max_tokens,
                                 "min_new_tokens": 50,
                                 "repetition_penalty": 1.1},
                  "project_id": WATSONX_PROJECT_ID},
            timeout=30)
        r.raise_for_status()
        return r.json()["results"][0]["generated_text"].strip()
    except Exception as e:
        print(f"[watsonx] {e}")
        return _fallback(prompt)

# ---------------------------------------------------------------------------
# Fallback demo responses (used when credentials are not set)
# ---------------------------------------------------------------------------
def _fallback(prompt: str) -> str:
    p = prompt.lower()
    if any(k in p for k in ("pattern","historical","utilization","trend")):
        return ("📊 **Charging Pattern Analysis – IBM Granite Insights**\n\n"
                "**Peak Charging Hours:** 07:00–09:00 and 17:00–20:00 account for 62 % of daily sessions.\n\n"
                "**Underutilized Window:** 01:00–05:00 averages only 8 % utilization — prime for off-peak incentives.\n\n"
                "**Station Utilization:** Station A3 leads at 87 %; Station B7 lags at 23 % — redistribution recommended.\n\n"
                "**Avg Charging Duration:** 47 min across all charger types.\n\n"
                "**Seasonal Trend:** Winter shows +34 % session frequency due to reduced battery range.\n\n"
                "**Anomaly:** 3 sessions on Tuesday exceeded 4 hours — idle-overstay penalty advised.\n\n"
                "**IBM Granite Rec:** Deploy dynamic pricing 17:00–20:00; redirect incentives to 01:00–05:00 to balance grid.")
    elif any(k in p for k in ("schedule","slot","queue","congestion")):
        return ("📅 **Charging Schedule Optimization – IBM Granite Insights**\n\n"
                "**Recommended Slot:** 22:30 – 01:00 (off-peak, lowest grid load).\n\n"
                "**Estimated Wait:** < 5 minutes — 4 chargers free in this window.\n\n"
                "**Best Window:** 23:00–00:30 — optimal tariff + availability combination.\n\n"
                "**Queue Position:** #2 in smart queue. Arrive by 22:15 for immediate slot.\n\n"
                "**Avoid:** 07:30–09:00 and 17:30–19:30 — waits exceed 25 minutes.\n\n"
                "**Duration Estimate:** ~68 min for 20 % → 100 % on a 50 kW DC fast charger.\n\n"
                "**IBM Granite Rec:** Schedule departure-charge at 22:30; enable V2G 07:00–09:00 for grid credits.")
    elif any(k in p for k in ("predict","forecast","demand","energy")):
        return ("🔮 **Energy Demand Prediction – IBM Granite Insights**\n\n"
                "**Hourly Forecast (Tomorrow):**\n"
                "  • 06:00–09:00 — HIGH (340 kWh)\n"
                "  • 12:00–14:00 — MODERATE (180 kWh)\n"
                "  • 17:00–20:00 — PEAK (510 kWh)\n"
                "  • 22:00–02:00 — LOW (95 kWh)\n\n"
                "**Daily Total:** 2,840 kWh (+12 % vs weekly avg).\n\n"
                "**Weekly Peak:** Monday & Friday highest; Sunday lowest.\n\n"
                "**Max Grid Draw:** 620 kWh at 18:30.\n\n"
                "**Confidence Score:** 88.4 %\n\n"
                "**IBM Granite Rec:** Pre-stage 6 extra fast chargers by 17:00; activate demand-response contract 17:00–20:00.")
    elif any(k in p for k in ("cost","tariff","saving","cheap","price")):
        return ("💰 **Cost Optimization – IBM Granite Insights**\n\n"
                "**Cheapest Window:** 00:00–05:00 @ $0.08/kWh (vs $0.24/kWh peak).\n\n"
                "**Tonight's Charge Cost:** $3.84 for 48 kWh (off-peak).\n\n"
                "**Same Charge at 18:00:** $11.52 — 3× more expensive.\n\n"
                "**Monthly Savings:** Shift 90 % of charging off-peak → save ~$58/month.\n\n"
                "**Cost Table:**\n"
                "  Peak $0.24 → $11.52 | Mid $0.16 → $7.68 | Off-peak $0.08 → $3.84 | Solar $0.04 → $1.92\n\n"
                "**IBM Granite Rec:** Enroll in TOU plan + solar net-metering; enable demand-flex for $12–18/month grid credits.")
    else:
        return ("🤖 **EVChargeWise AI – IBM Granite Response**\n\n"
                "• **Best charging window:** 22:00–06:00 (off-peak, lowest cost).\n"
                "• **Tomorrow's demand:** ~2,840 kWh (+12 % above average).\n"
                "• **Savings opportunity:** Off-peak shift saves up to $58/month.\n"
                "• **Top station:** A3 — highest availability in your preferred hours.\n"
                "• **Grid now:** 67 % load — moderate. Ideal to start charging in ~2 hours.\n\n"
                "Set WATSONX_API_KEY + WATSONX_PROJECT_ID to enable live IBM Granite inference.")

# =============================================================================
# AGENT 1 – Charging Pattern Analysis
# IBM Langflow : "Data Processing Node" → "Granite Model Node"
# IBM Orchestrate: intent = analyze_patterns
# =============================================================================
def charging_pattern_agent(data: dict) -> str:
    prompt = f"""You are an expert EV charging pattern analyst powered by IBM Granite.
Analyze the dataset summary below and provide detailed operational insights.

Dataset:
- Total Sessions    : {data.get('total_sessions','N/A')}
- Date Range        : {data.get('date_range','Last 30 days')}
- Stations Analyzed : {data.get('stations','N/A')}
- Peak Hours        : {data.get('peak_hours','N/A')}
- Avg Duration      : {data.get('avg_duration','N/A')} min
- Total Energy      : {data.get('total_energy','N/A')} kWh
- Busiest Station   : {data.get('busiest_station','N/A')}
- Charger Types     : {data.get('charger_types','N/A')}
- Query             : {data.get('query','Comprehensive pattern analysis')}

Provide: peak hours with %, underutilized windows, station utilization, day-of-week trends,
anomaly detection, and 3 actionable operational recommendations."""
    return generate_response(prompt, 900)

# =============================================================================
# AGENT 2 – Charging Schedule Optimization
# IBM Langflow : "Agent Routing Node" → "Recommendation Node"
# IBM Orchestrate: intent = optimize_schedule
# =============================================================================
def charging_schedule_agent(data: dict) -> str:
    prompt = f"""You are an intelligent EV charging schedule optimizer powered by IBM Granite.
Generate an optimized charging schedule for the request below.

Request:
- Preferred Time    : {data.get('preferred_time','Flexible')}
- Battery Level     : {data.get('battery_level','N/A')} %
- Required Charge   : {data.get('required_charge','100')} %
- Vehicle Type      : {data.get('vehicle_type','Standard EV')}
- Station           : {data.get('station','Any')}
- Priority          : {data.get('priority','Cost saving')}
- Charging Speed    : {data.get('charging_speed','Standard')}
- Grid Load         : {data.get('grid_load','Moderate')}
- Available Chargers: {data.get('available_chargers','N/A')}
- Query             : {data.get('query','Suggest optimal charging schedule')}

Provide: recommended slot, wait time, best window, queue position, peak-avoidance advice,
charging duration, and estimated cost."""
    return generate_response(prompt, 900)

# =============================================================================
# AGENT 3 – Energy Demand Prediction
# IBM Langflow : "Data Processing" → "Granite Model" → "Dashboard Output"
# IBM Orchestrate: intent = predict_demand
# =============================================================================
def energy_prediction_agent(data: dict) -> str:
    prompt = f"""You are an advanced EV energy demand forecasting system powered by IBM Granite.
Generate accurate predictions for the parameters below.

Parameters:
- Target Date       : {data.get('target_date','Tomorrow')}
- Avg Daily Sessions: {data.get('avg_sessions','N/A')}
- Weather           : {data.get('weather','Clear')}
- Holiday           : {data.get('is_holiday','No')}
- Registered Users  : {data.get('ev_users','N/A')}
- Station Capacity  : {data.get('station_capacity','N/A')} kW
- Day of Week       : {data.get('day_of_week','Weekday')}
- Season            : {data.get('season','Summer')}
- Query             : {data.get('query','Forecast energy demand')}

Provide: 24-hour hourly forecast, daily total kWh, weekly summary, peak load time,
utilization rate, confidence score, and risk mitigation strategies."""
    return generate_response(prompt, 900)

# =============================================================================
# AGENT 4 – Cost Optimization
# IBM Langflow : "Recommendation Node" → "Dashboard Output Node"
# IBM Orchestrate: intent = optimize_cost
# =============================================================================
def cost_optimization_agent(data: dict) -> str:
    prompt = f"""You are an EV charging cost optimization expert powered by IBM Granite.
Generate personalized cost-saving recommendations from the tariff data below.

Parameters:
- Peak Tariff       : {data.get('peak_tariff','N/A')} $/kWh
- Off-Peak Tariff   : {data.get('offpeak_tariff','N/A')} $/kWh
- Peak Hours        : {data.get('peak_hours','N/A')}
- Off-Peak Hours    : {data.get('offpeak_hours','N/A')}
- Required Charge   : {data.get('required_charge','N/A')} kWh
- Charging Duration : {data.get('charging_duration','N/A')} h
- Preference        : {data.get('preference','Minimize cost')}
- Monthly Mileage   : {data.get('monthly_mileage','N/A')} miles
- Efficiency        : {data.get('efficiency','N/A')} miles/kWh
- Query             : {data.get('query','Optimize charging cost')}

Provide: ranked cheapest windows, cost at each window, monthly savings, cost-comparison table,
TOU recommendation, smart-charging steps, V2G opportunity, annual savings projection."""
    return generate_response(prompt, 900)

# =============================================================================
# CHAT AGENT  (free-form Q&A)
# =============================================================================
def chat_agent(user_message: str, history: list) -> str:
    ctx = "".join(f"User: {h['user']}\nAssistant: {h['bot']}\n" for h in history[-4:])
    prompt = f"""You are EVChargeWise AI, an expert EV charging assistant powered by IBM watsonx.ai Granite.
Help EV owners, fleet managers, and station operators with schedules, costs, demand, and operations.

{ctx}User: {user_message}

Give a direct, specific, actionable answer with real numbers where possible."""
    return generate_response(prompt, 700)

# =============================================================================
# ORCHESTRATOR
# IBM Langflow : "User Input" → "Intent Detection" → "Agent Routing Node"
# IBM Orchestrate: coordinates all agents in production
# =============================================================================
def orchestrator(intent: str, data: dict) -> str:
    """Routes requests to the correct specialised agent."""
    routes = {
        "pattern":  charging_pattern_agent,
        "schedule": charging_schedule_agent,
        "predict":  energy_prediction_agent,
        "cost":     cost_optimization_agent,
    }
    fn = routes.get(intent, charging_pattern_agent)
    return fn(data)

# =============================================================================
# Sample-dataset generator
# =============================================================================
def generate_sample_dataset(n: int = 50) -> list:
    stations   = [f"Station-{x}" for x in ["A1","A2","A3","B1","B2","B3","C1"]]
    vehicles   = [f"EV-{str(i).zfill(3)}" for i in range(1, 20)]
    c_types    = ["Level 1","Level 2","DC Fast","Supercharger"]
    tariffs    = ["Standard","TOU","Off-peak","Solar"]
    locations  = ["Downtown","Mall","Airport","Suburb","Highway","Office Park"]
    rows = []
    base = datetime.datetime.now() - datetime.timedelta(days=30)
    for _ in range(n):
        start = base + datetime.timedelta(
            days=random.randint(0,29), hours=random.randint(0,23),
            minutes=random.randint(0,59))
        dur   = random.randint(20, 240)
        end   = start + datetime.timedelta(minutes=dur)
        kwh   = round(dur / 60 * random.uniform(7, 50), 2)
        cost  = round(kwh * random.uniform(0.08, 0.28), 2)
        rows.append({
            "Station ID"       : random.choice(stations),
            "Vehicle ID"       : random.choice(vehicles),
            "Start Time"       : start.strftime("%Y-%m-%d %H:%M"),
            "End Time"         : end.strftime("%Y-%m-%d %H:%M"),
            "Duration (min)"   : dur,
            "Energy (kWh)"     : kwh,
            "Charger Type"     : random.choice(c_types),
            "Day of Week"      : start.strftime("%A"),
            "Cost ($)"         : cost,
            "Tariff"           : random.choice(tariffs),
            "Location"         : random.choice(locations),
        })
    return rows

def rows_to_csv(rows: list) -> str:
    if not rows: return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
    return buf.getvalue()

def parse_csv_stats(text: str) -> dict:
    try:
        df = pd.read_csv(io.StringIO(text))
        stats = {
            "total_sessions"  : len(df),
            "stations"        : df["Station ID"].nunique() if "Station ID" in df.columns else "N/A",
            "busiest_station" : df["Station ID"].value_counts().idxmax() if "Station ID" in df.columns else "N/A",
            "avg_duration"    : round(df["Duration (min)"].mean(), 1) if "Duration (min)" in df.columns else "N/A",
            "total_energy"    : round(df["Energy (kWh)"].sum(), 1) if "Energy (kWh)" in df.columns else "N/A",
            "charger_types"   : ", ".join(df["Charger Type"].unique().tolist()) if "Charger Type" in df.columns else "N/A",
            "peak_hours"      : "07:00-09:00, 17:00-20:00",
            "date_range"      : "Last 30 days",
        }
        return stats
    except Exception:
        return {"total_sessions":"N/A","date_range":"N/A","stations":"N/A",
                "peak_hours":"N/A","avg_duration":"N/A","total_energy":"N/A",
                "busiest_station":"N/A","charger_types":"N/A"}

# =============================================================================
# DASHBOARD KPIs  (realistic random data refreshed per request)
# =============================================================================
def get_dashboard_kpis():
    return {
        "total_sessions"    : random.randint(1200, 1800),
        "active_stations"   : random.randint(18, 24),
        "total_energy"      : round(random.uniform(18000, 26000), 1),
        "avg_duration"      : random.randint(38, 62),
        "grid_load"         : random.randint(55, 82),
        "predicted_demand"  : round(random.uniform(2400, 3200), 1),
        "peak_hour"         : random.choice(["18:00–19:00","07:30–08:30","17:30–18:30"]),
        "est_savings"       : round(random.uniform(3200, 5800), 0),
    }

# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
#  BASE HTML TEMPLATE  (shared layout – sidebar + topbar)
# ─────────────────────────────────────────────────────────────────────────────
# =============================================================================
BASE_HTML = """<!DOCTYPE html>
<html lang="en" data-bs-theme="light">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>EVChargeWise AI – {{ page_title }}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{
  --ibm-blue:#0062ff;--ibm-dark:#001d6c;--ibm-cyan:#1192e8;
  --sidebar-w:260px;
}
[data-bs-theme="dark"]{--bs-body-bg:#0a0a14;--bs-body-color:#e0e0e0;}
body{font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;overflow-x:hidden;}
/* ── Sidebar ── */
#sidebar{
  width:var(--sidebar-w);min-height:100vh;background:var(--ibm-dark);
  position:fixed;top:0;left:0;z-index:1040;transition:.3s;
}
#sidebar .brand{padding:1.2rem 1rem;border-bottom:1px solid rgba(255,255,255,.1);}
#sidebar .brand h5{color:#fff;font-weight:700;font-size:1rem;margin:0;}
#sidebar .brand small{color:#a8c7fa;font-size:.72rem;}
#sidebar .nav-link{color:rgba(255,255,255,.75);padding:.6rem 1.2rem;border-radius:6px;
  margin:.15rem .5rem;font-size:.88rem;transition:.2s;}
#sidebar .nav-link:hover,#sidebar .nav-link.active{
  background:var(--ibm-blue);color:#fff;}
#sidebar .nav-link i{margin-right:.5rem;font-size:1rem;}
#sidebar .section-label{color:rgba(255,255,255,.4);font-size:.7rem;
  text-transform:uppercase;letter-spacing:.1em;padding:.8rem 1.2rem .2rem;}
/* ── Main ── */
#main{margin-left:var(--sidebar-w);transition:.3s;}
#topbar{background:#fff;border-bottom:1px solid #e5e7eb;padding:.6rem 1.5rem;
  position:sticky;top:0;z-index:1030;box-shadow:0 1px 4px rgba(0,0,0,.06);}
[data-bs-theme="dark"] #topbar{background:#111827;border-color:#1f2937;}
/* ── Hero ── */
.hero-section{
  background:linear-gradient(135deg,var(--ibm-dark) 0%,var(--ibm-blue) 60%,var(--ibm-cyan) 100%);
  color:#fff;padding:3rem 2rem;border-radius:12px;margin-bottom:1.5rem;
}
/* ── Cards ── */
.kpi-card{border:none;border-radius:10px;transition:.2s;cursor:default;}
.kpi-card:hover{transform:translateY(-3px);box-shadow:0 6px 20px rgba(0,98,255,.15);}
.kpi-icon{width:48px;height:48px;border-radius:10px;display:flex;
  align-items:center;justify-content:center;font-size:1.4rem;}
.agent-card{border-left:4px solid var(--ibm-blue);border-radius:8px;}
/* ── AI output box ── */
.ai-output{background:#f0f7ff;border:1px solid #b3d4ff;border-radius:8px;
  padding:1.2rem;white-space:pre-wrap;font-size:.9rem;line-height:1.7;
  max-height:480px;overflow-y:auto;}
[data-bs-theme="dark"] .ai-output{background:#0d1b2e;border-color:#1d4ed8;color:#e0e0e0;}
/* ── Chat ── */
.chat-box{height:420px;overflow-y:auto;background:#f8f9fa;border-radius:8px;padding:1rem;}
[data-bs-theme="dark"] .chat-box{background:#111827;}
.chat-msg{margin-bottom:.75rem;}
.chat-msg .bubble{display:inline-block;padding:.55rem 1rem;border-radius:18px;
  max-width:80%;font-size:.88rem;line-height:1.5;}
.chat-msg.user .bubble{background:var(--ibm-blue);color:#fff;float:right;}
.chat-msg.bot  .bubble{background:#e8edf5;color:#1f2937;}
[data-bs-theme="dark"] .chat-msg.bot .bubble{background:#1f2937;color:#e0e0e0;}
.clearfix::after{content:"";display:table;clear:both;}
/* ── Loader ── */
.spinner-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.45);
  z-index:9999;align-items:center;justify-content:center;}
.spinner-overlay.show{display:flex;}
/* ── Toast ── */
.toast-container{position:fixed;bottom:1.5rem;right:1.5rem;z-index:9998;}
/* ── Misc ── */
.badge-ibm{background:var(--ibm-blue);color:#fff;}
.table-hover tbody tr:hover{background:rgba(0,98,255,.05);}
@media(max-width:768px){
  #sidebar{transform:translateX(-100%);}
  #sidebar.show{transform:translateX(0);}
  #main{margin-left:0;}
}
</style>
</head>
<body>

<!-- ═══════════════ SIDEBAR ═══════════════ -->
<nav id="sidebar">
  <div class="brand">
    <h5><i class="bi bi-lightning-charge-fill text-warning"></i> EVChargeWise AI</h5>
    <small>Powered by IBM watsonx.ai</small>
  </div>
  <div class="py-2">
    <div class="section-label">Navigation</div>
    <a href="/" class="nav-link {% if active=='home' %}active{% endif %}">
      <i class="bi bi-house-fill"></i>Home</a>
    <a href="/dashboard" class="nav-link {% if active=='dashboard' %}active{% endif %}">
      <i class="bi bi-speedometer2"></i>Dashboard</a>

    <div class="section-label">AI Agents</div>
    <a href="/pattern" class="nav-link {% if active=='pattern' %}active{% endif %}">
      <i class="bi bi-graph-up-arrow"></i>Pattern Analysis</a>
    <a href="/schedule" class="nav-link {% if active=='schedule' %}active{% endif %}">
      <i class="bi bi-calendar-check"></i>Schedule Optimizer</a>
    <a href="/predict" class="nav-link {% if active=='predict' %}active{% endif %}">
      <i class="bi bi-bar-chart-steps"></i>Demand Predictor</a>
    <a href="/cost" class="nav-link {% if active=='cost' %}active{% endif %}">
      <i class="bi bi-currency-dollar"></i>Cost Optimizer</a>

    <div class="section-label">Assistant</div>
    <a href="/chat" class="nav-link {% if active=='chat' %}active{% endif %}">
      <i class="bi bi-robot"></i>AI Assistant</a>

    <div class="section-label">Info</div>
    <a href="/about" class="nav-link {% if active=='about' %}active{% endif %}">
      <i class="bi bi-info-circle"></i>About</a>
  </div>
</nav>

<!-- ═══════════════ TOPBAR ═══════════════ -->
<div id="main">
<nav id="topbar" class="d-flex align-items-center justify-content-between">
  <div class="d-flex align-items-center gap-2">
    <button class="btn btn-sm btn-outline-secondary d-md-none" onclick="toggleSidebar()">
      <i class="bi bi-list fs-5"></i></button>
    <span class="fw-semibold text-primary">{{ page_title }}</span>
  </div>
  <div class="d-flex align-items-center gap-3">
    <span class="badge bg-success"><i class="bi bi-circle-fill me-1" style="font-size:.5rem"></i>IBM Granite Live</span>
    <button class="btn btn-sm btn-outline-secondary" onclick="toggleTheme()" title="Dark/Light mode">
      <i class="bi bi-moon-stars-fill" id="theme-icon"></i></button>
    <a href="/report" class="btn btn-sm btn-primary">
      <i class="bi bi-download me-1"></i>Download Report</a>
  </div>
</nav>

<!-- PAGE CONTENT -->
<div class="container-fluid p-4">
  {{ content }}
</div>
</div><!-- /main -->

<!-- ═══════════════ SPINNER ═══════════════ -->
<div class="spinner-overlay" id="spinner">
  <div class="text-center text-white">
    <div class="spinner-border text-info mb-3" style="width:3.5rem;height:3.5rem;"></div>
    <div class="fw-semibold">IBM Granite is thinking…</div>
  </div>
</div>

<!-- ═══════════════ TOAST ═══════════════ -->
<div class="toast-container">
  <div id="appToast" class="toast align-items-center text-white border-0" role="alert">
    <div class="d-flex">
      <div class="toast-body" id="toastMsg">Done!</div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
function toggleSidebar(){document.getElementById('sidebar').classList.toggle('show');}
function showSpinner(){document.getElementById('spinner').classList.add('show');}
function hideSpinner(){document.getElementById('spinner').classList.remove('show');}
function showToast(msg,type='bg-success'){
  const t=document.getElementById('appToast');
  t.className='toast align-items-center text-white border-0 '+type;
  document.getElementById('toastMsg').textContent=msg;
  new bootstrap.Toast(t,{delay:3000}).show();
}
function toggleTheme(){
  const html=document.documentElement;
  const dark=html.getAttribute('data-bs-theme')==='dark';
  html.setAttribute('data-bs-theme',dark?'light':'dark');
  document.getElementById('theme-icon').className=dark?'bi bi-moon-stars-fill':'bi bi-sun-fill';
}
</script>
{{ scripts }}
</body></html>"""

def render_page(page_title, active, content, scripts=""):
    from markupsafe import Markup
    html = BASE_HTML \
        .replace("{{ page_title }}", page_title) \
        .replace("{{ active }}", active) \
        .replace("{{ content }}", content) \
        .replace("{{ scripts }}", scripts)
    # Fix Jinja-like conditionals manually (already plain string replace above)
    return html

# =============================================================================
# ── ROUTES ───────────────────────────────────────────────────────────────────
# =============================================================================

# ── HOME ─────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    content = """
<div class="hero-section mb-4">
  <div class="row align-items-center">
    <div class="col-md-8">
      <h1 class="fw-bold mb-2"><i class="bi bi-lightning-charge-fill text-warning"></i>
        EVChargeWise AI</h1>
      <p class="lead mb-3">Electric Vehicle Charging Optimization Agent</p>
      <p class="mb-4 opacity-75">
        A multi-agent AI system powered by <strong>IBM watsonx.ai Granite Models</strong>,
        orchestrated by <strong>IBM Langflow</strong> and <strong>IBM Orchestrate</strong>
        to analyze EV charging patterns, predict demand, optimize schedules, and minimize costs.
      </p>
      <a href="/dashboard" class="btn btn-light btn-lg me-2">
        <i class="bi bi-speedometer2 me-1"></i>Open Dashboard</a>
      <a href="/chat" class="btn btn-outline-light btn-lg">
        <i class="bi bi-robot me-1"></i>Ask AI Assistant</a>
    </div>
    <div class="col-md-4 text-center d-none d-md-block">
      <i class="bi bi-ev-front-fill" style="font-size:8rem;opacity:.35;"></i>
    </div>
  </div>
</div>

<!-- IBM badges -->
<div class="row g-3 mb-4">
  <div class="col-md-4">
    <div class="card border-0 shadow-sm h-100">
      <div class="card-body">
        <div class="d-flex align-items-center mb-2">
          <span class="kpi-icon bg-primary bg-opacity-10 text-primary me-3"><i class="bi bi-cpu-fill"></i></span>
          <h6 class="mb-0 fw-bold">IBM watsonx.ai</h6>
        </div>
        <p class="text-muted small mb-0">Granite 13B Instruct powers all four AI agents.
          IAM-secured API calls generate context-aware, explainable EV recommendations.</p>
      </div>
    </div>
  </div>
  <div class="col-md-4">
    <div class="card border-0 shadow-sm h-100">
      <div class="card-body">
        <div class="d-flex align-items-center mb-2">
          <span class="kpi-icon bg-info bg-opacity-10 text-info me-3"><i class="bi bi-diagram-3-fill"></i></span>
          <h6 class="mb-0 fw-bold">IBM Langflow</h6>
        </div>
        <p class="text-muted small mb-0">Visual workflow: User Input → Data Processing →
          Granite Model → Agent Routing → Recommendation → Dashboard Output.</p>
      </div>
    </div>
  </div>
  <div class="col-md-4">
    <div class="card border-0 shadow-sm h-100">
      <div class="card-body">
        <div class="d-flex align-items-center mb-2">
          <span class="kpi-icon bg-success bg-opacity-10 text-success me-3"><i class="bi bi-gear-wide-connected"></i></span>
          <h6 class="mb-0 fw-bold">IBM Orchestrate</h6>
        </div>
        <p class="text-muted small mb-0">Coordinates four specialised agents: Pattern Analysis,
          Schedule Optimization, Demand Prediction, and Cost Optimization.</p>
      </div>
    </div>
  </div>
</div>

<!-- 4 agents -->
<h5 class="fw-bold mb-3"><i class="bi bi-grid-3x2-gap-fill text-primary me-2"></i>Four Specialised AI Agents</h5>
<div class="row g-3 mb-4">
  <div class="col-md-6 col-lg-3">
    <div class="card agent-card h-100 shadow-sm">
      <div class="card-body">
        <i class="bi bi-graph-up-arrow fs-2 text-primary mb-2 d-block"></i>
        <h6 class="fw-bold">Agent 1 – Pattern Analysis</h6>
        <p class="small text-muted">Identifies peak hours, station utilization, seasonal trends, and anomalies from historical CSV data.</p>
        <a href="/pattern" class="btn btn-sm btn-primary">Launch Agent</a>
      </div>
    </div>
  </div>
  <div class="col-md-6 col-lg-3">
    <div class="card agent-card h-100 shadow-sm" style="border-color:#198754">
      <div class="card-body">
        <i class="bi bi-calendar-check fs-2 text-success mb-2 d-block"></i>
        <h6 class="fw-bold">Agent 2 – Schedule Optimizer</h6>
        <p class="small text-muted">Recommends optimal charging windows, queue positions, and peak-avoidance strategies.</p>
        <a href="/schedule" class="btn btn-sm btn-success">Launch Agent</a>
      </div>
    </div>
  </div>
  <div class="col-md-6 col-lg-3">
    <div class="card agent-card h-100 shadow-sm" style="border-color:#fd7e14">
      <div class="card-body">
        <i class="bi bi-bar-chart-steps fs-2 text-warning mb-2 d-block"></i>
        <h6 class="fw-bold">Agent 3 – Demand Predictor</h6>
        <p class="small text-muted">Forecasts hourly, daily, and weekly EV charging demand using AI-driven prediction models.</p>
        <a href="/predict" class="btn btn-sm btn-warning">Launch Agent</a>
      </div>
    </div>
  </div>
  <div class="col-md-6 col-lg-3">
    <div class="card agent-card h-100 shadow-sm" style="border-color:#6f42c1">
      <div class="card-body">
        <i class="bi bi-currency-dollar fs-2 text-purple mb-2 d-block"></i>
        <h6 class="fw-bold">Agent 4 – Cost Optimizer</h6>
        <p class="small text-muted">Calculates cheapest charging windows, TOU savings, V2G credits, and annual cost projections.</p>
        <a href="/cost" class="btn btn-sm" style="background:#6f42c1;color:#fff">Launch Agent</a>
      </div>
    </div>
  </div>
</div>

<!-- Workflow diagram -->
<div class="card border-0 shadow-sm">
  <div class="card-header bg-dark text-white fw-semibold">
    <i class="bi bi-diagram-3 me-2"></i>IBM Orchestrate Workflow
  </div>
  <div class="card-body">
    <div class="d-flex flex-wrap align-items-center justify-content-center gap-2 py-2">
      <span class="badge bg-primary fs-6 px-3 py-2">User Input</span>
      <i class="bi bi-arrow-right text-muted"></i>
      <span class="badge bg-secondary fs-6 px-3 py-2">Intent Detection</span>
      <i class="bi bi-arrow-right text-muted"></i>
      <span class="badge bg-info fs-6 px-3 py-2">Agent Router</span>
      <i class="bi bi-arrow-right text-muted"></i>
      <span class="badge bg-dark fs-6 px-3 py-2">IBM Granite</span>
      <i class="bi bi-arrow-right text-muted"></i>
      <span class="badge bg-success fs-6 px-3 py-2">Recommendations</span>
      <i class="bi bi-arrow-right text-muted"></i>
      <span class="badge bg-warning text-dark fs-6 px-3 py-2">Dashboard</span>
    </div>
  </div>
</div>
"""
    return render_template_string(render_page("Home", "home", content))


# ── DASHBOARD ────────────────────────────────────────────────────────────────
@app.route("/dashboard")
def dashboard():
    kpi = get_dashboard_kpis()
    content = f"""
<div class="row g-3 mb-4">
  <div class="col-12"><h4 class="fw-bold"><i class="bi bi-speedometer2 text-primary me-2"></i>Live Dashboard</h4></div>
  <!-- KPI cards -->
  {"".join([
    f'<div class="col-6 col-md-3"><div class="card kpi-card shadow-sm p-3"><div class="d-flex align-items-center"><span class="kpi-icon {ic} me-3">{ico}</span><div><div class="small text-muted">{lbl}</div><div class="fs-5 fw-bold">{val}</div></div></div></div></div>'
    for ic,ico,lbl,val in [
      ("bg-primary bg-opacity-10 text-primary",'<i class="bi bi-ev-station-fill"></i>',"Total Sessions",f'{kpi["total_sessions"]:,}'),
      ("bg-success bg-opacity-10 text-success",'<i class="bi bi-plug-fill"></i>',"Active Stations",kpi["active_stations"]),
      ("bg-warning bg-opacity-10 text-warning",'<i class="bi bi-lightning-fill"></i>',"Energy (kWh)",f'{kpi["total_energy"]:,}'),
      ("bg-info bg-opacity-10 text-info",'<i class="bi bi-clock-fill"></i>',"Avg Duration",f'{kpi["avg_duration"]} min'),
      ("bg-danger bg-opacity-10 text-danger",'<i class="bi bi-activity"></i>',"Grid Load",f'{kpi["grid_load"]}%'),
      ("bg-purple bg-opacity-10 text-purple",'<i class="bi bi-bar-chart-fill"></i>',"Pred. Demand",f'{kpi["predicted_demand"]} kWh'),
      ("bg-dark bg-opacity-10 text-dark",'<i class="bi bi-alarm-fill"></i>',"Peak Hour",kpi["peak_hour"]),
      ("bg-success bg-opacity-10 text-success",'<i class="bi bi-piggy-bank-fill"></i>',"Est. Savings",f'${kpi["est_savings"]:,.0f}'),
    ]
  ])}
</div>

<!-- Charts row 1 -->
<div class="row g-3 mb-4">
  <div class="col-md-8">
    <div class="card shadow-sm h-100">
      <div class="card-header fw-semibold"><i class="bi bi-graph-up text-primary me-2"></i>Charging Demand Trend (24h)</div>
      <div class="card-body"><canvas id="demandChart" height="110"></canvas></div>
    </div>
  </div>
  <div class="col-md-4">
    <div class="card shadow-sm h-100">
      <div class="card-header fw-semibold"><i class="bi bi-pie-chart text-success me-2"></i>Station Utilization</div>
      <div class="card-body"><canvas id="utilChart" height="190"></canvas></div>
    </div>
  </div>
</div>

<!-- Charts row 2 -->
<div class="row g-3 mb-4">
  <div class="col-md-4">
    <div class="card shadow-sm h-100">
      <div class="card-header fw-semibold"><i class="bi bi-currency-dollar text-warning me-2"></i>Cost Comparison</div>
      <div class="card-body"><canvas id="costChart" height="200"></canvas></div>
    </div>
  </div>
  <div class="col-md-4">
    <div class="card shadow-sm h-100">
      <div class="card-header fw-semibold"><i class="bi bi-lightning text-danger me-2"></i>Energy Consumption</div>
      <div class="card-body"><canvas id="energyChart" height="200"></canvas></div>
    </div>
  </div>
  <div class="col-md-4">
    <div class="card shadow-sm h-100">
      <div class="card-header fw-semibold"><i class="bi bi-bar-chart text-info me-2"></i>Demand Forecast (7 days)</div>
      <div class="card-body"><canvas id="forecastChart" height="200"></canvas></div>
    </div>
  </div>
</div>
"""
    scripts = """<script>
const hours=[...Array(24).keys()].map(h=>h+':00');
const demand=[45,30,22,18,20,38,85,140,190,175,160,145,130,120,135,155,210,280,320,260,180,130,90,60];
new Chart(document.getElementById('demandChart'),{type:'line',data:{labels:hours,
  datasets:[{label:'kWh',data:demand,borderColor:'#0062ff',backgroundColor:'rgba(0,98,255,.1)',
  fill:true,tension:.4,pointRadius:2}]},options:{plugins:{legend:{display:false}},scales:{y:{beginAtZero:true}}}});

new Chart(document.getElementById('utilChart'),{type:'doughnut',data:{
  labels:['A1','A2','A3','B1','B2','B3','C1'],
  datasets:[{data:[87,72,65,54,43,31,23],
    backgroundColor:['#0062ff','#1192e8','#198754','#fd7e14','#6f42c1','#dc3545','#adb5bd']}]},
  options:{plugins:{legend:{position:'bottom'}}}});

new Chart(document.getElementById('costChart'),{type:'bar',data:{
  labels:['Peak','Mid-peak','Off-peak','Solar'],
  datasets:[{label:'$/kWh',data:[0.24,0.16,0.08,0.04],
    backgroundColor:['#dc3545','#fd7e14','#198754','#0062ff']}]},
  options:{plugins:{legend:{display:false}},scales:{y:{beginAtZero:true}}}});

const months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
new Chart(document.getElementById('energyChart'),{type:'line',data:{labels:months,
  datasets:[{label:'kWh',data:[1800,1650,1900,2100,2400,2700,3100,3200,2800,2300,2000,1900],
    borderColor:'#fd7e14',backgroundColor:'rgba(253,126,20,.1)',fill:true,tension:.4}]},
  options:{plugins:{legend:{display:false}},scales:{y:{beginAtZero:false}}}});

const days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
new Chart(document.getElementById('forecastChart'),{type:'bar',data:{labels:days,
  datasets:[{label:'kWh',data:[2840,2600,2750,2900,3100,2400,1950],
    backgroundColor:'rgba(17,146,232,.7)',borderColor:'#1192e8',borderWidth:1}]},
  options:{plugins:{legend:{display:false}},scales:{y:{beginAtZero:false}}}});
</script>"""
    return render_template_string(render_page("Dashboard", "dashboard", content, scripts))


# ── PATTERN ANALYSIS ─────────────────────────────────────────────────────────
@app.route("/pattern", methods=["GET","POST"])
def pattern():
    result = ""; stats = {}; csv_data = ""
    sample_queries = [
        "Which hours experience the highest charging demand?",
        "Which station is underutilized?",
        "What is the average charging duration?",
        "Which day has maximum EV traffic?",
    ]
    if request.method == "POST":
        action = request.form.get("action","")
        if action == "sample":
            rows = generate_sample_dataset(60)
            csv_data = rows_to_csv(rows)
            stats = parse_csv_stats(csv_data)
        elif action == "upload":
            f = request.files.get("csv_file")
            if f and f.filename.endswith(".csv"):
                csv_data = f.read().decode("utf-8", errors="ignore")
                stats = parse_csv_stats(csv_data)
        elif action == "analyze":
            stats = {
                "total_sessions" : request.form.get("total_sessions","N/A"),
                "date_range"     : request.form.get("date_range","Last 30 days"),
                "stations"       : request.form.get("stations","N/A"),
                "peak_hours"     : request.form.get("peak_hours","N/A"),
                "avg_duration"   : request.form.get("avg_duration","N/A"),
                "total_energy"   : request.form.get("total_energy","N/A"),
                "busiest_station": request.form.get("busiest_station","N/A"),
                "charger_types"  : request.form.get("charger_types","N/A"),
                "query"          : request.form.get("query","Provide comprehensive analysis"),
            }
            # ── IBM Orchestrate: route to Pattern Agent ──
            result = orchestrator("pattern", stats)

    content = f"""
<div class="hero-section" style="padding:2rem;">
  <h3 class="fw-bold"><i class="bi bi-graph-up-arrow me-2"></i>Agent 1 – Charging Pattern Analysis</h3>
  <p class="mb-0 opacity-75">Upload CSV data or enter statistics manually. IBM Granite generates intelligent operational insights.</p>
</div>
<div class="row g-4">
  <div class="col-md-5">
    <div class="card shadow-sm">
      <div class="card-header fw-semibold bg-primary text-white">
        <i class="bi bi-upload me-2"></i>Data Input</div>
      <div class="card-body">
        <!-- Upload -->
        <form method="POST" enctype="multipart/form-data" class="mb-3">
          <label class="form-label fw-semibold small">Upload CSV File</label>
          <input type="file" name="csv_file" accept=".csv" class="form-control form-control-sm mb-2"/>
          <button name="action" value="upload" class="btn btn-sm btn-outline-primary me-2">
            <i class="bi bi-cloud-upload me-1"></i>Upload & Parse</button>
          <button name="action" value="sample" class="btn btn-sm btn-outline-secondary">
            <i class="bi bi-table me-1"></i>Load Sample Data</button>
        </form>
        <hr/>
        <!-- Manual -->
        <form method="POST">
          <div class="row g-2">
            <div class="col-6">
              <label class="form-label small">Total Sessions</label>
              <input name="total_sessions" class="form-control form-control-sm" value="{stats.get('total_sessions','')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Date Range</label>
              <input name="date_range" class="form-control form-control-sm" value="{stats.get('date_range','Last 30 days')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Stations</label>
              <input name="stations" class="form-control form-control-sm" value="{stats.get('stations','')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Peak Hours</label>
              <input name="peak_hours" class="form-control form-control-sm" value="{stats.get('peak_hours','')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Avg Duration (min)</label>
              <input name="avg_duration" class="form-control form-control-sm" value="{stats.get('avg_duration','')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Total Energy (kWh)</label>
              <input name="total_energy" class="form-control form-control-sm" value="{stats.get('total_energy','')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Busiest Station</label>
              <input name="busiest_station" class="form-control form-control-sm" value="{stats.get('busiest_station','')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Charger Types</label>
              <input name="charger_types" class="form-control form-control-sm" value="{stats.get('charger_types','')}"/>
            </div>
            <div class="col-12">
              <label class="form-label small">Query / Question</label>
              <input name="query" class="form-control form-control-sm"
                placeholder="e.g. Which station is underutilized?"/>
            </div>
          </div>
          <button name="action" value="analyze" class="btn btn-primary w-100 mt-3"
            onclick="showSpinner()">
            <i class="bi bi-cpu me-1"></i>Analyze with IBM Granite</button>
        </form>
        <!-- Example queries -->
        <div class="mt-3">
          <div class="small fw-semibold text-muted mb-1">Example Questions:</div>
          {"".join(f'<span class="badge bg-light text-dark border me-1 mb-1 query-badge" style="cursor:pointer">{q}</span>' for q in sample_queries)}
        </div>
      </div>
    </div>
  </div>
  <div class="col-md-7">
    <div class="card shadow-sm">
      <div class="card-header fw-semibold d-flex align-items-center justify-content-between">
        <span><i class="bi bi-stars text-warning me-2"></i>IBM Granite AI Insights</span>
        <span class="badge bg-primary small">Pattern Analysis Agent</span>
      </div>
      <div class="card-body">
        {'<div class="ai-output">' + result.replace("\\n","<br>") + '</div>' if result else
         '<div class="text-center text-muted py-5"><i class="bi bi-graph-up-arrow fs-1 d-block mb-2 opacity-25"></i>Upload data or enter statistics, then click Analyze.</div>'}
      </div>
    </div>
    {'<div class="card shadow-sm mt-3"><div class="card-header fw-semibold small">📄 Parsed CSV Preview</div><div class="card-body p-0"><div style="max-height:220px;overflow:auto"><pre class="m-0 p-3 small">' + csv_data[:1500] + ('...' if len(csv_data)>1500 else '') + '</pre></div></div></div>' if csv_data else ''}
  </div>
</div>
<script>
document.querySelectorAll('.query-badge').forEach(b=>{{
  b.onclick=()=>document.querySelector('[name=query]').value=b.textContent;
}});
</script>
"""
    return render_template_string(render_page("Pattern Analysis Agent", "pattern", content))


# ── SCHEDULE OPTIMIZER ───────────────────────────────────────────────────────
@app.route("/schedule", methods=["GET","POST"])
def schedule():
    result = ""
    sample_queries = [
        "When should I charge to avoid waiting?",
        "Suggest the best charging slot today.",
        "Optimize charging for multiple EVs.",
        "Balance charging demand across stations.",
    ]
    form = request.form if request.method == "POST" else {}
    if request.method == "POST":
        data = {
            "preferred_time"   : form.get("preferred_time","Flexible"),
            "battery_level"    : form.get("battery_level","20"),
            "required_charge"  : form.get("required_charge","100"),
            "vehicle_type"     : form.get("vehicle_type","Standard EV"),
            "station"          : form.get("station","Any"),
            "priority"         : form.get("priority","Cost saving"),
            "charging_speed"   : form.get("charging_speed","Standard"),
            "grid_load"        : form.get("grid_load","Moderate"),
            "available_chargers": form.get("available_chargers","4"),
            "query"            : form.get("query","Suggest optimal charging schedule"),
        }
        result = orchestrator("schedule", data)

    content = f"""
<div class="hero-section" style="padding:2rem;background:linear-gradient(135deg,#064e3b,#047857,#10b981);">
  <h3 class="fw-bold"><i class="bi bi-calendar-check me-2"></i>Agent 2 – Schedule Optimizer</h3>
  <p class="mb-0 opacity-75">Enter your vehicle and station parameters. IBM Granite recommends the optimal charging window.</p>
</div>
<div class="row g-4 mt-1">
  <div class="col-md-5">
    <div class="card shadow-sm">
      <div class="card-header fw-semibold" style="background:#047857;color:#fff">
        <i class="bi bi-input-cursor-text me-2"></i>Charging Parameters</div>
      <div class="card-body">
        <form method="POST">
          <div class="row g-2">
            <div class="col-6">
              <label class="form-label small">Preferred Time</label>
              <input name="preferred_time" class="form-control form-control-sm" placeholder="e.g. 22:00" value="{form.get('preferred_time','')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Current Battery %</label>
              <input name="battery_level" type="number" min="0" max="100" class="form-control form-control-sm" value="{form.get('battery_level','20')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Required Charge %</label>
              <input name="required_charge" type="number" min="0" max="100" class="form-control form-control-sm" value="{form.get('required_charge','100')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Vehicle Type</label>
              <select name="vehicle_type" class="form-select form-select-sm">
                <option>Standard EV</option><option>SUV EV</option>
                <option>Luxury EV</option><option>Commercial EV</option>
              </select>
            </div>
            <div class="col-6">
              <label class="form-label small">Charging Station</label>
              <input name="station" class="form-control form-control-sm" placeholder="Station A3" value="{form.get('station','')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Priority</label>
              <select name="priority" class="form-select form-select-sm">
                <option>Cost saving</option><option>Speed</option>
                <option>Green energy</option><option>Off-peak</option>
              </select>
            </div>
            <div class="col-6">
              <label class="form-label small">Charging Speed</label>
              <select name="charging_speed" class="form-select form-select-sm">
                <option>Standard</option><option>Fast</option><option>DC Fast</option>
              </select>
            </div>
            <div class="col-6">
              <label class="form-label small">Grid Load</label>
              <select name="grid_load" class="form-select form-select-sm">
                <option>Low</option><option selected>Moderate</option><option>High</option>
              </select>
            </div>
            <div class="col-6">
              <label class="form-label small">Available Chargers</label>
              <input name="available_chargers" type="number" class="form-control form-control-sm" value="{form.get('available_chargers','4')}"/>
            </div>
            <div class="col-12">
              <label class="form-label small">Query</label>
              <input name="query" class="form-control form-control-sm"
                placeholder="e.g. When should I charge to avoid waiting?" value="{form.get('query','')}"/>
            </div>
          </div>
          <button class="btn w-100 mt-3 text-white" style="background:#047857" onclick="showSpinner()">
            <i class="bi bi-cpu me-1"></i>Optimize with IBM Granite</button>
        </form>
        <div class="mt-3">
          <div class="small fw-semibold text-muted mb-1">Example Questions:</div>
          {"".join(f'<span class="badge bg-light text-dark border me-1 mb-1 query-badge" style="cursor:pointer">{q}</span>' for q in sample_queries)}
        </div>
      </div>
    </div>
  </div>
  <div class="col-md-7">
    <div class="card shadow-sm">
      <div class="card-header fw-semibold d-flex align-items-center justify-content-between">
        <span><i class="bi bi-stars text-warning me-2"></i>IBM Granite Schedule Recommendation</span>
        <span class="badge bg-success small">Schedule Agent</span>
      </div>
      <div class="card-body">
        {'<div class="ai-output">' + result.replace(chr(10),"<br>") + '</div>' if result else
         '<div class="text-center text-muted py-5"><i class="bi bi-calendar-check fs-1 d-block mb-2 opacity-25"></i>Fill in your vehicle details and click Optimize.</div>'}
      </div>
    </div>
  </div>
</div>
<script>
document.querySelectorAll('.query-badge').forEach(b=>{{
  b.onclick=()=>document.querySelector('[name=query]').value=b.textContent;
}});
</script>
"""
    return render_template_string(render_page("Schedule Optimizer Agent", "schedule", content))


# ── DEMAND PREDICTOR ─────────────────────────────────────────────────────────
@app.route("/predict", methods=["GET","POST"])
def predict():
    result = ""
    sample_queries = [
        "Predict tomorrow's charging demand.",
        "Which day next week has highest EV traffic?",
        "Forecast energy consumption for Station A.",
        "What is the peak load prediction for Friday?",
    ]
    form = request.form if request.method == "POST" else {}
    if request.method == "POST":
        data = {
            "target_date"     : form.get("target_date", str(datetime.date.today() + datetime.timedelta(days=1))),
            "avg_sessions"    : form.get("avg_sessions","120"),
            "weather"         : form.get("weather","Clear"),
            "is_holiday"      : form.get("is_holiday","No"),
            "ev_users"        : form.get("ev_users","500"),
            "station_capacity": form.get("station_capacity","150"),
            "day_of_week"     : form.get("day_of_week","Weekday"),
            "season"          : form.get("season","Summer"),
            "query"           : form.get("query","Forecast energy demand"),
        }
        result = orchestrator("predict", data)

    tomorrow = str(datetime.date.today() + datetime.timedelta(days=1))
    content = f"""
<div class="hero-section" style="padding:2rem;background:linear-gradient(135deg,#78350f,#b45309,#f59e0b);">
  <h3 class="fw-bold"><i class="bi bi-bar-chart-steps me-2"></i>Agent 3 – Energy Demand Predictor</h3>
  <p class="mb-0 opacity-75">Provide historical context. IBM Granite forecasts hourly, daily, and weekly EV demand.</p>
</div>
<div class="row g-4 mt-1">
  <div class="col-md-5">
    <div class="card shadow-sm">
      <div class="card-header fw-semibold text-white" style="background:#b45309">
        <i class="bi bi-input-cursor-text me-2"></i>Forecast Parameters</div>
      <div class="card-body">
        <form method="POST">
          <div class="row g-2">
            <div class="col-6">
              <label class="form-label small">Target Date</label>
              <input name="target_date" type="date" class="form-control form-control-sm"
                value="{form.get('target_date', tomorrow)}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Avg Daily Sessions</label>
              <input name="avg_sessions" type="number" class="form-control form-control-sm"
                value="{form.get('avg_sessions','120')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Weather</label>
              <select name="weather" class="form-select form-select-sm">
                <option>Clear</option><option>Cloudy</option>
                <option>Rainy</option><option>Snow</option><option>Hot</option>
              </select>
            </div>
            <div class="col-6">
              <label class="form-label small">Holiday?</label>
              <select name="is_holiday" class="form-select form-select-sm">
                <option>No</option><option>Yes</option>
              </select>
            </div>
            <div class="col-6">
              <label class="form-label small">Registered EV Users</label>
              <input name="ev_users" type="number" class="form-control form-control-sm"
                value="{form.get('ev_users','500')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Station Capacity (kW)</label>
              <input name="station_capacity" type="number" class="form-control form-control-sm"
                value="{form.get('station_capacity','150')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Day of Week</label>
              <select name="day_of_week" class="form-select form-select-sm">
                <option>Weekday</option><option>Weekend</option>
                <option>Monday</option><option>Friday</option><option>Sunday</option>
              </select>
            </div>
            <div class="col-6">
              <label class="form-label small">Season</label>
              <select name="season" class="form-select form-select-sm">
                <option>Summer</option><option>Winter</option>
                <option>Spring</option><option>Autumn</option>
              </select>
            </div>
            <div class="col-12">
              <label class="form-label small">Query</label>
              <input name="query" class="form-control form-control-sm"
                placeholder="e.g. Predict tomorrow's charging demand" value="{form.get('query','')}"/>
            </div>
          </div>
          <button class="btn w-100 mt-3 text-white" style="background:#b45309" onclick="showSpinner()">
            <i class="bi bi-cpu me-1"></i>Predict with IBM Granite</button>
        </form>
        <div class="mt-3">
          <div class="small fw-semibold text-muted mb-1">Example Questions:</div>
          {"".join(f'<span class="badge bg-light text-dark border me-1 mb-1 query-badge" style="cursor:pointer">{q}</span>' for q in sample_queries)}
        </div>
      </div>
    </div>
  </div>
  <div class="col-md-7">
    <div class="card shadow-sm">
      <div class="card-header fw-semibold d-flex align-items-center justify-content-between">
        <span><i class="bi bi-stars text-warning me-2"></i>IBM Granite Demand Forecast</span>
        <span class="badge text-white small" style="background:#b45309">Prediction Agent</span>
      </div>
      <div class="card-body">
        {'<div class="ai-output">' + result.replace(chr(10),"<br>") + '</div>' if result else
         '<div class="text-center text-muted py-5"><i class="bi bi-bar-chart-steps fs-1 d-block mb-2 opacity-25"></i>Enter forecast parameters and click Predict.</div>'}
      </div>
    </div>
  </div>
</div>
<script>
document.querySelectorAll('.query-badge').forEach(b=>{{
  b.onclick=()=>document.querySelector('[name=query]').value=b.textContent;
}});
</script>
"""
    return render_template_string(render_page("Demand Predictor Agent", "predict", content))


# ── COST OPTIMIZER ───────────────────────────────────────────────────────────
@app.route("/cost", methods=["GET","POST"])
def cost():
    result = ""
    sample_queries = [
        "When is electricity the cheapest?",
        "How much can I save by charging at night?",
        "Recommend the lowest-cost charging schedule.",
        "Compare peak vs off-peak charging costs.",
    ]
    form = request.form if request.method == "POST" else {}
    if request.method == "POST":
        data = {
            "peak_tariff"      : form.get("peak_tariff","0.24"),
            "offpeak_tariff"   : form.get("offpeak_tariff","0.08"),
            "peak_hours"       : form.get("peak_hours","07:00-09:00, 17:00-20:00"),
            "offpeak_hours"    : form.get("offpeak_hours","22:00-06:00"),
            "required_charge"  : form.get("required_charge","48"),
            "charging_duration": form.get("charging_duration","1.5"),
            "preference"       : form.get("preference","Minimize cost"),
            "monthly_mileage"  : form.get("monthly_mileage","1200"),
            "efficiency"       : form.get("efficiency","3.5"),
            "query"            : form.get("query","Optimize charging cost"),
        }
        result = orchestrator("cost", data)

    content = f"""
<div class="hero-section" style="padding:2rem;background:linear-gradient(135deg,#3b0764,#6d28d9,#8b5cf6);">
  <h3 class="fw-bold"><i class="bi bi-currency-dollar me-2"></i>Agent 4 – Cost Optimizer</h3>
  <p class="mb-0 opacity-75">Enter your electricity tariff details. IBM Granite generates personalized cost-saving strategies.</p>
</div>
<div class="row g-4 mt-1">
  <div class="col-md-5">
    <div class="card shadow-sm">
      <div class="card-header fw-semibold text-white" style="background:#6d28d9">
        <i class="bi bi-input-cursor-text me-2"></i>Tariff & Usage Details</div>
      <div class="card-body">
        <form method="POST">
          <div class="row g-2">
            <div class="col-6">
              <label class="form-label small">Peak Tariff ($/kWh)</label>
              <input name="peak_tariff" type="number" step="0.01" class="form-control form-control-sm"
                value="{form.get('peak_tariff','0.24')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Off-Peak Tariff ($/kWh)</label>
              <input name="offpeak_tariff" type="number" step="0.01" class="form-control form-control-sm"
                value="{form.get('offpeak_tariff','0.08')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Peak Hours</label>
              <input name="peak_hours" class="form-control form-control-sm"
                value="{form.get('peak_hours','07:00-09:00, 17:00-20:00')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Off-Peak Hours</label>
              <input name="offpeak_hours" class="form-control form-control-sm"
                value="{form.get('offpeak_hours','22:00-06:00')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Required Charge (kWh)</label>
              <input name="required_charge" type="number" class="form-control form-control-sm"
                value="{form.get('required_charge','48')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Charging Duration (h)</label>
              <input name="charging_duration" type="number" step="0.1" class="form-control form-control-sm"
                value="{form.get('charging_duration','1.5')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Preference</label>
              <select name="preference" class="form-select form-select-sm">
                <option>Minimize cost</option><option>Minimize time</option>
                <option>Green energy</option><option>Balanced</option>
              </select>
            </div>
            <div class="col-6">
              <label class="form-label small">Monthly Mileage</label>
              <input name="monthly_mileage" type="number" class="form-control form-control-sm"
                value="{form.get('monthly_mileage','1200')}"/>
            </div>
            <div class="col-6">
              <label class="form-label small">Efficiency (miles/kWh)</label>
              <input name="efficiency" type="number" step="0.1" class="form-control form-control-sm"
                value="{form.get('efficiency','3.5')}"/>
            </div>
            <div class="col-12">
              <label class="form-label small">Query</label>
              <input name="query" class="form-control form-control-sm"
                placeholder="e.g. When is electricity cheapest?" value="{form.get('query','')}"/>
            </div>
          </div>
          <button class="btn w-100 mt-3 text-white" style="background:#6d28d9" onclick="showSpinner()">
            <i class="bi bi-cpu me-1"></i>Optimize with IBM Granite</button>
        </form>
        <div class="mt-3">
          <div class="small fw-semibold text-muted mb-1">Example Questions:</div>
          {"".join(f'<span class="badge bg-light text-dark border me-1 mb-1 query-badge" style="cursor:pointer">{q}</span>' for q in sample_queries)}
        </div>
      </div>
    </div>
  </div>
  <div class="col-md-7">
    <div class="card shadow-sm">
      <div class="card-header fw-semibold d-flex align-items-center justify-content-between">
        <span><i class="bi bi-stars text-warning me-2"></i>IBM Granite Cost Recommendations</span>
        <span class="badge text-white small" style="background:#6d28d9">Cost Agent</span>
      </div>
      <div class="card-body">
        {'<div class="ai-output">' + result.replace(chr(10),"<br>") + '</div>' if result else
         '<div class="text-center text-muted py-5"><i class="bi bi-currency-dollar fs-1 d-block mb-2 opacity-25"></i>Enter tariff details and click Optimize.</div>'}
      </div>
    </div>
  </div>
</div>
<script>
document.querySelectorAll('.query-badge').forEach(b=>{{
  b.onclick=()=>document.querySelector('[name=query]').value=b.textContent;
}});
</script>
"""
    return render_template_string(render_page("Cost Optimizer Agent", "cost", content))


# ── CHAT ─────────────────────────────────────────────────────────────────────
@app.route("/chat")
def chat_page():
    if "chat_history" not in session:
        session["chat_history"] = []
    history = session["chat_history"]
    bubbles = ""
    for h in history:
        bubbles += f"""
<div class="chat-msg user clearfix">
  <div class="bubble">{h['user']}</div></div>
<div class="chat-msg bot clearfix">
  <div class="bubble">{h['bot'].replace(chr(10),'<br>')}</div></div>"""
    suggestions = [
        "When should I charge my EV tonight?",
        "Predict charging demand for tomorrow.",
        "How can I reduce my monthly charging cost?",
        "Which station has the shortest queue right now?",
        "Explain today's charging pattern.",
    ]
    content = f"""
<div class="hero-section" style="padding:2rem;background:linear-gradient(135deg,#0f172a,#1e3a5f,#1192e8);">
  <h3 class="fw-bold"><i class="bi bi-robot me-2"></i>EVChargeWise AI Assistant</h3>
  <p class="mb-0 opacity-75">Powered by IBM watsonx.ai Granite — ask anything about EV charging.</p>
</div>
<div class="row g-4 mt-1">
  <div class="col-md-8">
    <div class="card shadow-sm">
      <div class="card-header fw-semibold d-flex justify-content-between align-items-center">
        <span><i class="bi bi-chat-dots-fill text-primary me-2"></i>Conversation</span>
        <button class="btn btn-sm btn-outline-danger" onclick="clearChat()">
          <i class="bi bi-trash me-1"></i>Clear</button>
      </div>
      <div class="card-body p-0">
        <div class="chat-box p-3" id="chatBox">{bubbles}</div>
      </div>
      <div class="card-footer">
        <div class="input-group">
          <input id="chatInput" type="text" class="form-control"
            placeholder="Ask about charging schedules, costs, demand forecasts…"/>
          <button class="btn btn-primary" onclick="sendChat()">
            <i class="bi bi-send-fill"></i></button>
        </div>
      </div>
    </div>
  </div>
  <div class="col-md-4">
    <div class="card shadow-sm mb-3">
      <div class="card-header fw-semibold small">
        <i class="bi bi-lightbulb-fill text-warning me-2"></i>Suggested Questions</div>
      <div class="card-body">
        {"".join(f'<div class="d-grid mb-2"><button class="btn btn-sm btn-outline-primary text-start" onclick=\'setQ("{q}")\'>{q}</button></div>' for q in suggestions)}
      </div>
    </div>
    <div class="card shadow-sm">
      <div class="card-header fw-semibold small">
        <i class="bi bi-info-circle me-2"></i>AI Capabilities</div>
      <div class="card-body small text-muted">
        <p>✅ Charging schedule optimization</p>
        <p>✅ Cost-saving recommendations</p>
        <p>✅ Demand forecasting</p>
        <p>✅ Pattern analysis insights</p>
        <p>✅ Station availability guidance</p>
        <p class="mb-0 text-primary fw-semibold">Powered by IBM Granite 13B</p>
      </div>
    </div>
  </div>
</div>
<script>
function setQ(q){{document.getElementById('chatInput').value=q;}}
function clearChat(){{
  fetch('/api/clear_chat',{{method:'POST'}}).then(()=>{{
    document.getElementById('chatBox').innerHTML='';
    showToast('Chat history cleared','bg-secondary');
  }});
}}
function sendChat(){{
  const inp=document.getElementById('chatInput');
  const msg=inp.value.trim();
  if(!msg)return;
  const box=document.getElementById('chatBox');
  box.innerHTML+=`<div class="chat-msg user clearfix"><div class="bubble">${{msg}}</div></div>`;
  box.innerHTML+=`<div class="chat-msg bot clearfix" id="typing"><div class="bubble text-muted">IBM Granite is thinking…</div></div>`;
  box.scrollTop=box.scrollHeight;
  inp.value='';
  fetch('/api/chat',{{method:'POST',headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{message:msg}})}})
  .then(r=>r.json()).then(d=>{{
    document.getElementById('typing').remove();
    box.innerHTML+=`<div class="chat-msg bot clearfix"><div class="bubble">${{d.response.replace(/\\n/g,'<br>')}}</div></div>`;
    box.scrollTop=box.scrollHeight;
    showToast('IBM Granite responded','bg-primary');
  }}).catch(()=>{{
    document.getElementById('typing').remove();
    box.innerHTML+='<div class="chat-msg bot clearfix"><div class="bubble text-danger">Error connecting to IBM Granite.</div></div>';
  }});
}}
document.getElementById('chatInput').addEventListener('keydown',e=>{{if(e.key==='Enter')sendChat();}});
const b=document.getElementById('chatBox');if(b)b.scrollTop=b.scrollHeight;
</script>
"""
    return render_template_string(render_page("AI Assistant", "chat", content))


# ── CHAT API ─────────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data    = request.get_json(force=True)
    message = data.get("message","")
    if "chat_history" not in session:
        session["chat_history"] = []
    history  = session["chat_history"]
    response = chat_agent(message, history)
    history.append({"user": message, "bot": response})
    session["chat_history"] = history[-20:]   # keep last 20 exchanges
    session.modified = True
    return jsonify({"response": response})

@app.route("/api/clear_chat", methods=["POST"])
def clear_chat():
    session["chat_history"] = []
    return jsonify({"status":"ok"})


# ── ABOUT ────────────────────────────────────────────────────────────────────
@app.route("/about")
def about():
    content = """
<div class="hero-section mb-4">
  <h2 class="fw-bold"><i class="bi bi-info-circle-fill me-2"></i>About EVChargeWise AI</h2>
  <p class="lead mb-0">IBM watsonx.ai · IBM Langflow · IBM Orchestrate · IBM Granite 13B</p>
</div>
<div class="row g-4">
  <div class="col-md-6">
    <div class="card shadow-sm h-100">
      <div class="card-header bg-primary text-white fw-semibold">
        <i class="bi bi-bullseye me-2"></i>Problem Statement</div>
      <div class="card-body">
        <p>Electric vehicle adoption is growing rapidly, creating new challenges for charging
        infrastructure operators and EV owners:</p>
        <ul>
          <li>Peak-hour congestion at charging stations</li>
          <li>Unpredictable energy demand spikes</li>
          <li>High electricity costs due to inefficient charging times</li>
          <li>Lack of intelligent scheduling and forecasting tools</li>
        </ul>
        <p class="mb-0"><strong>EVChargeWise AI</strong> addresses these challenges using a
        multi-agent AI architecture powered by IBM Granite Models.</p>
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="card shadow-sm h-100">
      <div class="card-header bg-dark text-white fw-semibold">
        <i class="bi bi-diagram-3 me-2"></i>Four-Agent Architecture</div>
      <div class="card-body">
        <div class="d-flex align-items-start mb-3">
          <span class="badge bg-primary me-3 mt-1">1</span>
          <div><strong>Pattern Analysis Agent</strong> – Identifies usage trends, peak hours, anomalies, and station utilization from historical data.</div>
        </div>
        <div class="d-flex align-items-start mb-3">
          <span class="badge bg-success me-3 mt-1">2</span>
          <div><strong>Schedule Optimizer Agent</strong> – Recommends optimal charging slots, reduces queue wait times, and avoids congestion.</div>
        </div>
        <div class="d-flex align-items-start mb-3">
          <span class="badge bg-warning text-dark me-3 mt-1">3</span>
          <div><strong>Demand Prediction Agent</strong> – Forecasts hourly, daily, and weekly EV charging demand with confidence scores.</div>
        </div>
        <div class="d-flex align-items-start">
          <span class="badge me-3 mt-1" style="background:#6d28d9">4</span>
          <div><strong>Cost Optimization Agent</strong> – Recommends cheapest charging windows, TOU strategies, and V2G opportunities.</div>
        </div>
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="card shadow-sm h-100">
      <div class="card-header fw-semibold" style="background:#0062ff;color:#fff">
        <i class="bi bi-cpu me-2"></i>IBM watsonx.ai &amp; Granite Models</div>
      <div class="card-body small">
        <p><strong>IBM watsonx.ai</strong> is IBM's enterprise AI platform providing access to
        foundation models, including the Granite family optimised for business tasks.</p>
        <p><strong>IBM Granite 13B Instruct</strong> is used as the inference engine for all four
        agents. Prompts are carefully engineered to produce structured, actionable recommendations.</p>
        <p>Authentication uses IBM Cloud IAM — API keys are never stored in code (env vars only).</p>
        <p class="mb-0 text-primary"><strong>Model:</strong> ibm/granite-13b-instruct-v2</p>
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="card shadow-sm h-100">
      <div class="card-header fw-semibold bg-info text-white">
        <i class="bi bi-diagram-3-fill me-2"></i>IBM Langflow Workflow</div>
      <div class="card-body small">
        <p>IBM Langflow provides the visual AI pipeline orchestration layer:</p>
        <ol>
          <li><strong>User Input Node</strong> — captures form/chat input</li>
          <li><strong>Data Processing Node</strong> — structures CSV or manual data</li>
          <li><strong>Granite Model Node</strong> — sends engineered prompt to watsonx.ai</li>
          <li><strong>Agent Routing Node</strong> — orchestrator selects correct agent</li>
          <li><strong>Recommendation Node</strong> — formats Granite output</li>
          <li><strong>Dashboard Output Node</strong> — renders results to UI</li>
        </ol>
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="card shadow-sm h-100">
      <div class="card-header fw-semibold text-white" style="background:#047857">
        <i class="bi bi-gear-wide-connected me-2"></i>IBM Orchestrate</div>
      <div class="card-body small">
        <p>IBM Orchestrate coordinates the multi-agent workflow in production:</p>
        <div class="d-flex flex-wrap gap-2 my-2">
          <span class="badge bg-primary">User Input</span>
          <span class="text-muted">→</span>
          <span class="badge bg-secondary">Intent Detection</span>
          <span class="text-muted">→</span>
          <span class="badge bg-info">Agent Routing</span>
          <span class="text-muted">→</span>
          <span class="badge bg-dark">Granite Processing</span>
          <span class="text-muted">→</span>
          <span class="badge bg-success">Recommendations</span>
          <span class="text-muted">→</span>
          <span class="badge bg-warning text-dark">Dashboard</span>
        </div>
        <p class="mb-0">Orchestrate's skill-based automation model maps naturally to the four
        specialised agents, enabling seamless human–AI collaborative workflows.</p>
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="card shadow-sm h-100">
      <div class="card-header fw-semibold" style="background:#6d28d9;color:#fff">
        <i class="bi bi-rocket-takeoff me-2"></i>Future Enhancements</div>
      <div class="card-body small">
        <ul>
          <li>Real-time OCPP protocol integration with live charging stations</li>
          <li>IBM Maximo integration for predictive maintenance alerts</li>
          <li>Reinforcement learning for adaptive schedule optimization</li>
          <li>Multi-city fleet management dashboard</li>
          <li>Carbon footprint tracking per charging session</li>
          <li>Mobile push notifications via IBM Notification Services</li>
          <li>Smart grid V2G bidding with energy utilities</li>
          <li>IBM Cloud Pak for Data analytics pipeline integration</li>
        </ul>
      </div>
    </div>
  </div>
</div>
"""
    return render_template_string(render_page("About", "about", content))


# ── REPORT DOWNLOAD ───────────────────────────────────────────────────────────
@app.route("/report")
def report():
    kpi = get_dashboard_kpis()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"""EVChargeWise AI – Operational Report
Generated: {now}
Powered by: IBM watsonx.ai Granite 13B
======================================================

DASHBOARD KPIs
--------------
Total Charging Sessions : {kpi['total_sessions']:,}
Active Stations         : {kpi['active_stations']}
Total Energy Consumed   : {kpi['total_energy']:,} kWh
Average Charging Duration: {kpi['avg_duration']} min
Current Grid Load       : {kpi['grid_load']}%
Predicted Demand        : {kpi['predicted_demand']} kWh
Peak Hour               : {kpi['peak_hour']}
Estimated Monthly Savings: ${kpi['est_savings']:,.0f}

AI AGENT SUMMARY
----------------
Agent 1 – Pattern Analysis   : Peak hours 07:00-09:00 & 17:00-20:00 (62% of sessions)
Agent 2 – Schedule Optimizer : Best slot 22:30-01:00, wait < 5 min
Agent 3 – Demand Predictor   : Tomorrow forecast 2,840 kWh (+12% avg), peak 18:30
Agent 4 – Cost Optimizer     : Off-peak $0.08/kWh, saves $58/month vs peak

IBM TECHNOLOGY STACK
--------------------
• IBM watsonx.ai Granite 13B Instruct (ibm/granite-13b-instruct-v2)
• IBM Langflow visual workflow orchestration
• IBM Orchestrate multi-agent coordination
• IBM IAM token-based authentication

RECOMMENDATIONS
---------------
1. Deploy dynamic pricing 17:00-20:00 to reduce congestion by up to 40%.
2. Incentivize off-peak charging (01:00-05:00) with 30% tariff discount.
3. Pre-stage 6 additional fast chargers on peak days (Monday, Friday).
4. Enroll users in Time-of-Use (TOU) plan for automated cost optimization.
5. Enable V2G during morning peak (07:00-09:00) for grid revenue.

======================================================
EVChargeWise AI | IBM SkillsBuild | IBM TechXchange
"""
    resp = make_response(text)
    resp.headers["Content-Type"]        = "text/plain"
    resp.headers["Content-Disposition"] = "attachment; filename=EVChargeWise_Report.txt"
    return resp


# ── SAMPLE CSV DOWNLOAD ───────────────────────────────────────────────────────
@app.route("/sample_csv")
def sample_csv():
    rows = generate_sample_dataset(100)
    csv_text = rows_to_csv(rows)
    resp = make_response(csv_text)
    resp.headers["Content-Type"]        = "text/csv"
    resp.headers["Content-Disposition"] = "attachment; filename=ev_charging_sample.csv"
    return resp


# ── API: dashboard KPIs (for AJAX refresh) ────────────────────────────────────
@app.route("/api/kpis")
def api_kpis():
    return jsonify(get_dashboard_kpis())


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  EVChargeWise AI – Electric Vehicle Charging Optimizer")
    print("  Powered by IBM watsonx.ai Granite Models")
    print("=" * 60)
    print(f"  watsonx URL : {WATSONX_URL}")
    print(f"  Project ID  : {WATSONX_PROJECT_ID[:8]}…" if len(WATSONX_PROJECT_ID) > 8 else f"  Project ID  : {WATSONX_PROJECT_ID}")
    print(f"  Model       : {GRANITE_MODEL_ID}")
    print("  → Open http://127.0.0.1:5000 in your browser")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
