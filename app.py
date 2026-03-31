from flask import Flask, render_template_string, jsonify
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
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23BF00FF' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='18' cy='5' r='3'/><circle cx='6' cy='12' r='3'/><circle cx='18' cy='19' r='3'/><line x1='8.59' y1='13.51' x2='15.42' y2='17.49'/><line x1='15.41' y1='6.51' x2='8.59' y2='10.49'/></svg>">
    {% if prediction and prediction.is_stale %}
    <meta http-equiv="refresh" content="30">
    {% else %}
    <meta http-equiv="refresh" content="60">
    {% endif %}
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 20px;
            background-color: #000000;
            color: #FFFFFF;
            margin: 0;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #ffffff; text-align: center; margin-bottom: 10px; }
        .subtitle { color: #a1a1aa; text-align: center; margin: 0 0 30px 0; }

        .row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
        @media (max-width: 768px) { .row { grid-template-columns: 1fr; } }

        .card { background: #1e1e1e; padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 8px 16px rgba(0,0,0,0.5); }
        .status-box { padding: 20px; border-radius: 10px; margin: 20px 0; font-weight: bold; font-size: 24px; }
        .safe { background-color: rgba(44, 255, 5, 0.08); color: #2CFF05; border: 1px solid rgba(44,255,5,0.35); }
        .danger { background-color: rgba(191, 0, 255, 0.08); color: #BF00FF; border: 1px solid rgba(191,0,255,0.35); animation: pulse 2s infinite; }

        .metric-row { display: flex; justify-content: space-between; margin-top: 15px; color: #a1a1aa; }
        .metric-value { font-weight: bold; font-size: 18px; color: #ffffff; }

        .stale-warning { background: rgba(191,0,255,0.1); color: #a1a1aa; border: 1px solid rgba(191,0,255,0.3); padding: 10px; border-radius: 5px; margin-top: 15px; font-size: 12px; font-weight: bold; }
        .stale-critical { background: rgba(191,0,255,0.18); color: #BF00FF; border: 1px solid rgba(191,0,255,0.5); padding: 10px; border-radius: 5px; margin-top: 15px; font-size: 12px; font-weight: bold; }

        .chart-container { position: relative; height: 300px; }
        .chart-title { font-weight: bold; margin-bottom: 10px; color: #ffffff; }

        .skeleton-loader {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(90deg, rgba(255,255,255,0.03) 25%, rgba(255,255,255,0.08) 50%, rgba(255,255,255,0.03) 75%);
            background-size: 200% 100%;
            animation: skeleton-loading 1.5s infinite;
            border-radius: 8px;
        }
        @keyframes skeleton-loading {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }

        .site-footer {
            text-align: center;
            margin-top: 50px;
            padding-top: 25px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            color: #a1a1aa;
            font-size: 14px;
        }
        .site-footer a {
            color: #a1a1aa;
            text-decoration: none;
            margin: 0 5px;
            transition: color 0.2s ease;
            font-weight: 600;
        }
        .site-footer a:hover {
            color: #BF00FF;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(191, 0, 255, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(191, 0, 255, 0); }
            100% { box-shadow: 0 0 0 0 rgba(191, 0, 255, 0); }
        }
        .header-container {
            position: relative;
            text-align: center;
            margin-bottom: 10px;
            padding-top: 5px;
        }
        .badge-container {
            position: absolute;
            left: 0;
            top: 25%;
            transform: translateY(-50%);
            display: flex;
            gap: 15px;
        }
        @media (max-width: 900px) {
            .header-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 15px;
                padding-top: 0;
            }
            .badge-container {
                position: relative;
                top: auto;
                left: auto;
                transform: none;
                margin-bottom: 0;
            }
        }
        .tech-badge {
            display: inline-flex;
            align-items: center;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #a1a1aa;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-decoration: none;
            transition: all 0.2s ease;
        }
        .tech-badge:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateY(-2px);
        }
        .tech-badge svg {
            margin-right: 8px;
        }
        .aws-badge {
            background: rgba(191, 0, 255, 0.1);
            border-color: rgba(191, 0, 255, 0.4);
            color: #BF00FF;
        }
        .aws-badge:hover {
            background: rgba(191, 0, 255, 0.25);
            border-color: #BF00FF;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header-container">
            <div class="badge-container">
                <div class="tech-badge aws-badge">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M17.5 19c2.485 0 4.5-2 4.5-4.5q0-.354-.06-.694A5 5 0 0 0 17 5.093 7 7 0 0 0 4.3 9.4 4.5 4.5 0 0 0 3.5 18"></path>
                        <path d="M3.5 19h14"></path>
                    </svg>
                    HOSTED ON AWS EC2
                </div>
            </div>
            <h1 style="margin-bottom: 0;">AI Network Degradation Detection</h1>
        </div>
        <p class="subtitle">Real-time Anomaly Detection — Last 12 Hours</p>
        <p style="text-align: center; color: #a1a1aa; max-width: 800px; margin: 0 auto 30px; font-size: 14px; line-height: 1.6;">
            An XGBoost-powered system that predicts network degradation 5 minutes before it happens using real-world internet telemetry from RIPE Atlas.
            Monitor live latency, jitter, and degradation probability with 12-hour historical trends to catch issues before they impact your experience.
        </p>

        {% if error %}
            <div class="card" style="text-align: center;">
                <p style="color:#BF00FF">System Error: {{ error }}</p>
            </div>
        {% else %}
            <!-- Row 1: Status Card + Probability Chart -->
            <div class="row">
                <div class="card">
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

                    <p style="font-size: 14px; color: #A0A0A0; margin: 15px 0;">
                        AI Calculated Degradation Probability: <strong>{{ (prediction.probability * 100)|round(1) }}%</strong>
                    </p>

                    <hr style="border-color: rgba(255, 255, 255, 0.1);">

                    <div class="metric-row">
                        <span>Latency</span>
                        <span class="metric-value">{{ prediction.latency|round(1) }} ms</span>
                    </div>
                    <div class="metric-row">
                        <span>Jitter</span>
                        <span class="metric-value">{{ prediction.jitter|round(1) }} ms</span>
                    </div>

                    <p style="margin-top: 15px; font-size: 12px; color: #a1a1aa;">Last Updated: {{ prediction.timestamp|to_pst }}</p>
                </div>

                <div class="card">
                    <div class="chart-title">Degradation Probability</div>
                    <div class="chart-container">
                        <div class="skeleton-loader" id="sk-prob"></div>
                        <canvas id="probabilityChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- Row 2: Latency Chart + Jitter Chart -->
            <div class="row">
                <div class="card">
                    <div class="chart-title">Latency — Last 12 Hours</div>
                    <div class="chart-container">
                        <div class="skeleton-loader" id="sk-lat"></div>
                        <canvas id="latencyChart"></canvas>
                    </div>
                </div>

                <div class="card">
                    <div class="chart-title">Jitter — Last 12 Hours</div>
                    <div class="chart-container">
                        <div class="skeleton-loader" id="sk-jit"></div>
                        <canvas id="jitterChart"></canvas>
                    </div>
                </div>
            </div>
        {% endif %}

        <footer class="site-footer">
            Built by Dhruv | 
            <a href="https://www.linkedin.com/in/dhruv-thippasandra/" target="_blank">LinkedIn</a> | 
            <a href="https://github.com/Dhruvq" target="_blank">GitHub</a>
        </footer>
    </div>

    <script>
        const chartColors = {
            latency: '#2CFF05',
            jitter: '#FFFFFF',
            probability: '#BF00FF',
            grid: 'rgba(255, 255, 255, 0.05)'
        };

        async function initCharts() {
            try {
                const response = await fetch('/api/history');
                const history = await response.json();

                if (!history || history.length === 0) {
                    console.log('No history data available yet');
                    return;
                }

                // Prepare labels (timestamps in HH:mm PST format)
                const labels = history.map(entry => {
                    const date = new Date(entry.timestamp * 1000);
                    return date.toLocaleString('en-US', {
                        hour: '2-digit',
                        minute: '2-digit',
                        hour12: false,
                        timeZone: 'America/Los_Angeles'
                    });
                });

                const latencyData = history.map(entry => entry.latency);
                const jitterData = history.map(entry => entry.jitter);
                const probabilityData = history.map(entry => entry.probability * 100);

                ['sk-prob', 'sk-lat', 'sk-jit'].forEach(id => {
                    const el = document.getElementById(id);
                    if (el) el.style.display = 'none';
                });

                const chartOptions = {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        filler: { propagate: true }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: { color: '#a1a1aa', font: { size: 11 } },
                            grid: { color: chartColors.grid }
                        },
                        x: {
                            ticks: { color: '#a1a1aa', font: { size: 10 }, maxRotation: 45, minRotation: 0 },
                            grid: { color: chartColors.grid }
                        }
                    }
                };

                // Latency Chart
                new Chart(document.getElementById('latencyChart'), {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Latency (ms)',
                            data: latencyData,
                            borderColor: chartColors.latency,
                            backgroundColor: 'rgba(44, 255, 5, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.4,
                            pointRadius: 2,
                            pointBackgroundColor: chartColors.latency
                        }]
                    },
                    options: {
                        ...chartOptions,
                        scales: {
                            ...chartOptions.scales,
                            y: { ...chartOptions.scales.y, title: { display: true, text: 'ms' } }
                        }
                    }
                });

                // Jitter Chart
                new Chart(document.getElementById('jitterChart'), {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Jitter (ms)',
                            data: jitterData,
                            borderColor: chartColors.jitter,
                            backgroundColor: 'rgba(255, 255, 255, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.4,
                            pointRadius: 2,
                            pointBackgroundColor: chartColors.jitter
                        }]
                    },
                    options: chartOptions
                });

                // Probability Chart
                new Chart(document.getElementById('probabilityChart'), {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Degradation Risk (%)',
                            data: probabilityData,
                            borderColor: chartColors.probability,
                            backgroundColor: 'rgba(191, 0, 255, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.4,
                            pointRadius: 2,
                            pointBackgroundColor: chartColors.probability
                        }]
                    },
                    options: {
                        ...chartOptions,
                        scales: {
                            ...chartOptions.scales,
                            y: {
                                ...chartOptions.scales.y,
                                max: 100,
                                ticks: { ...chartOptions.scales.y.ticks, callback: v => v + '%' }
                            }
                        }
                    }
                });
            } catch (error) {
                console.error('Failed to load history:', error);
                
                ['sk-prob', 'sk-lat', 'sk-jit'].forEach(id => {
                    const el = document.getElementById(id);
                    if (el) {
                        el.style.animation = 'none';
                        el.style.backgroundColor = 'rgba(191, 0, 255, 0.08)';
                        el.innerHTML = '<div style="color: #a1a1aa; padding: 20px; text-align: center;">Error loading chart data.</div>';
                    }
                });
            }
        }

        // Initialize charts when page loads
        document.addEventListener('DOMContentLoaded', initCharts);
    </script>
</body>
</html>
"""

@app.route('/api/history')
def api_history():
    """Return historical data for the past 12 hours (used by chart.js graphs)."""
    history_path = "/var/lib/netprophet/cache/history.json"

    try:
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            return jsonify(history)
        else:
            return jsonify([])
    except Exception as e:
        print(f"[app] Failed to read history: {e}")
        return jsonify([])

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