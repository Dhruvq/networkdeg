#!/usr/bin/env python3
"""
Standalone script to update prediction cache.
Run this via cron every 5-10 minutes to keep cache fresh.

Example cron entry (every 10 minutes):
*/10 * * * * /usr/bin/python3 /path/to/update_cache.py >> /var/log/netprophet_cache.log 2>&1
"""

import time
import json
import os
import inference

HISTORY_PATH = "/var/lib/netprophet/cache/history.json"
HISTORY_TTL_SECONDS = 12 * 3600  # 12 hours

def append_to_history(result):
    """Append prediction to history file, pruning entries older than 12 hours."""
    try:
        # Load existing history
        history = []
        if os.path.exists(HISTORY_PATH):
            try:
                with open(HISTORY_PATH, "r") as fh:
                    history = json.load(fh)
            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warning: failed to read history: {e}")
                history = []

        # Create history entry
        timestamp = int(time.time())
        entry = {
            "timestamp": timestamp,
            "latency": result.get("latency"),
            "jitter": result.get("jitter"),
            "probability": result.get("probability")
        }
        history.append(entry)

        # Prune old entries (older than 12 hours)
        cutoff_time = timestamp - HISTORY_TTL_SECONDS
        history = [h for h in history if h["timestamp"] >= cutoff_time]

        # Ensure directory exists
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

        # Write back
        with open(HISTORY_PATH, "w") as fh:
            json.dump(history, fh)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Appended to history ({len(history)} entries)")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Failed to update history: {e}")

if __name__ == "__main__":
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting cache update...")
    try:
        result = inference.get_live_prediction()
        if "error" in result:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Cache update failed: {result['error']}")
            exit(1)
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Cache updated successfully!")
            print(f"  Probability: {result['probability']:.2%}")
            print(f"  Latency: {result['latency']:.1f} ms")
            # Append to history for graphs
            append_to_history(result)
            exit(0)
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Cache update exception: {e}")
        exit(1)
