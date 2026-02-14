#!/usr/bin/env python3
"""
Standalone script to update prediction cache.
Run this via cron every 5-10 minutes to keep cache fresh.

Example cron entry (every 10 minutes):
*/10 * * * * /usr/bin/python3 /path/to/update_cache.py >> /var/log/netprophet_cache.log 2>&1
"""

import time
import inference

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
            exit(0)
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Cache update exception: {e}")
        exit(1)
