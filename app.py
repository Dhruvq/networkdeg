from flask import Flask, render_template_string
import inference
import json
import os
import time
from datetime import datetime
import pytz

app = Flask(__name__)

@app.template_filter('to_pst')
def to_pst(timestamp_str):
    """Convert UTC timestamp string to PST"""
    try:
        # Parse the timestamp (assuming it's in UTC)
        dt = datetime.fromisoformat(str(timestamp_str).replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        # Convert to PST
        pst = pytz.timezone('America/Los_Angeles')
        dt_pst = dt.astimezone(pst)
        return dt_pst.strftime('%Y-%m-%d %H:%M:%S %Z')
    except Exception:
        return timestamp_str

def _get_cached_prediction():
    """
    Read prediction from cache file updated by cron job.
    NEVER falls back to live inference - always returns cached data.
    Adds 'is_stale' flag if cache is old.
    """
    cache_file = inference.PERSISTENT_PREDICT_PATH
    max_age_seconds = 15 * 60  # 15 minutes
    stale_threshold = 30 * 60   # 30 minutes (really old)

    # Try to read cached file
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)

            cached_ts = data.get('ts', 0)
            age = int(time.time()) - cached_ts
            prediction = data.get('prediction')

            # Skip staleness metadata for error dicts
            if "error" in prediction:
                return prediction

            # Add staleness metadata (only for valid predictions)
            if age < max_age_seconds:
                print(f"[app] Serving fresh cache (age: {age}s)")
                prediction['is_stale'] = False
                prediction['cache_age_seconds'] = age
            elif age < stale_threshold:
                print(f"[app] Serving slightly stale cache (age: {age}s)")
                prediction['is_stale'] = True
                prediction['cache_age_seconds'] = age
                prediction['staleness_level'] = 'warning'
            else:
                print(f"[app] Serving very stale cache (age: {age}s)")
                prediction['is_stale'] = True
                prediction['cache_age_seconds'] = age
                prediction['staleness_level'] = 'critical'

            return prediction
    except Exception as e:
        print(f"[app] Failed to read cache: {e}")

    # If no cache exists at all, return error (don't block on live fetch)
    return {
        "error": "Prediction service starting up. Please refresh in 1 minute.",
        "is_stale": False
    }

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Network Degradation Detection</title>
    {% if prediction and prediction.is_stale %}
    <meta http-equiv="refresh" content="30">
    {% else %}
    <meta http-equiv="refresh" content="60">
    {% endif %}
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; text-align: center; padding: 20px; background-color: #1a1a2e; color: #fff; }
        .card { background: #16213e; padding: 30px; border-radius: 15px; box-shadow: 0 8px 16px rgba(0,0,0,0.3); max-width: 400px; margin: 0 auto; }
        h1 { color: #e94560; margin-bottom: 5px; }
        p.subtitle { color: #a2a8d3; margin-top: 0; }
        
        .status-box { padding: 20px; border-radius: 10px; margin: 20px 0; font-weight: bold; font-size: 24px; }
        .safe { background-color: #0f3460; color: #4cd137; border: 2px solid #4cd137; }
        .danger { background-color: #5c1a1a; color: #e84118; border: 2px solid #e84118; animation: pulse 2s infinite; }
        
        .metric-row { display: flex; justify-content: space-between; margin-top: 20px; color: #cbd5e0; }
        .metric-value { font-weight: bold; font-size: 18px; }

        .stale-warning { background: #ff9800; color: #000; padding: 10px; border-radius: 5px; margin-top: 15px; font-size: 12px; font-weight: bold; }
        .stale-critical { background: #e74c3c; color: #fff; padding: 10px; border-radius: 5px; margin-top: 15px; font-size: 12px; font-weight: bold; }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(232, 65, 24, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(232, 65, 24, 0); }
            100% { box-shadow: 0 0 0 0 rgba(232, 65, 24, 0); }
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>AI Network degradation detection</h1>
        <p class="subtitle">Real-time Anomaly Detection</p>
        
        {% if error %}
            <p style="color:#e84118">System Error: {{ error }}</p>
        {% else %}
            <div class="status-box {{ 'danger' if prediction.probability > 0.4 else 'safe' }}">
                {{ 'DEGRADATION RISK' if prediction.probability > 0.4 else '✅ SYSTEM HEALTHY' }}
            </div>

            {% if prediction.is_stale %}
                {% if prediction.staleness_level == 'critical' %}
                <div class="stale-critical">
                    Data is {{ (prediction.cache_age_seconds / 60)|round(0) }} minutes old. Background refresh may have failed.
                </div>
                {% else %}
                <div class="stale-warning">
                    Data is {{ (prediction.cache_age_seconds / 60)|round(0) }} minutes old. Refreshing in background...
                </div>
                {% endif %}
            {% endif %}

            <p style="font-size: 14px; color: #a2a8d3;">
                AI Calculated Degradation Probability: <strong>{{ (prediction.probability * 100)|round(1) }}%</strong>
            </p>
            
            <hr style="border-color: #0f3460;">
            
            <div class="metric-row">
                <span>Latency</span>
                <span class="metric-value">{{ prediction.latency|round(1) }} ms</span>
            </div>
            <div class="metric-row">
                <span>Jitter</span>
                <span class="metric-value">{{ prediction.jitter|round(1) }} ms</span>
            </div>
            
            <p style="margin-top: 30px; font-size: 12px; color: #535c68;">Last Updated: {{ prediction.timestamp|to_pst }}</p>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    try:
        data = _get_cached_prediction()
        if isinstance(data, dict) and "error" in data:
            # return friendly page and HTTP 503
            return render_template_string(HTML_TEMPLATE, error=data['error']), 503
        return render_template_string(HTML_TEMPLATE, prediction=data)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=str(e)), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# @app.route('/')
# def home():
#     try:
#         data = inference.get_live_prediction()
#         if "error" in data:
#             return render_template_string(HTML_TEMPLATE, error=data['error'])
#         return render_template_string(HTML_TEMPLATE, prediction=data)
#     except Exception as e:
#         return render_template_string(HTML_TEMPLATE, error=str(e))

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)