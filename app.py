from flask import Flask, render_template_string
import inference

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Network degredation detection</title>
    <meta http-equiv="refresh" content="60"> <meta name="viewport" content="width=device-width, initial-scale=1">
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
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(232, 65, 24, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(232, 65, 24, 0); }
            100% { box-shadow: 0 0 0 0 rgba(232, 65, 24, 0); }
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>AI Network degredation detection</h1>
        <p class="subtitle">Real-time Anomaly Detection</p>
        
        {% if error %}
            <p style="color:#e84118">System Error: {{ error }}</p>
        {% else %}
            <div class="status-box {{ 'danger' if prediction.probability > 0.4 else 'safe' }}">
                {{ 'DEGRADATION RISK' if prediction.probability > 0.4 else '✅ SYSTEM HEALTHY' }}
            </div>
            
            <p style="font-size: 14px; color: #a2a8d3;">
                AI Calculated Degredation Probability: <strong>{{ (prediction.probability * 100)|round(1) }}%</strong>
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
            
            <p style="margin-top: 30px; font-size: 12px; color: #535c68;">Last Updated: {{ prediction.timestamp }} UTC</p>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    try:
        data = inference.get_live_prediction()
        if "error" in data:
            return render_template_string(HTML_TEMPLATE, error=data['error'])
        return render_template_string(HTML_TEMPLATE, prediction=data)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)