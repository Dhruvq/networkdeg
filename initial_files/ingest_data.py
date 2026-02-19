import requests
import pandas as pd
import time
import os
import concurrent.futures

# Measurement ID 1001 is a "Root DNS" anchor measurement (High stability)
MEASUREMENT_ID = 1001 
RIPE_API_URL = f"https://atlas.ripe.net/api/v2/measurements/{MEASUREMENT_ID}/results/"

def fetch_chunk(start_ts, stop_ts, limit):
    """Helper function to fetch a specific time slice."""
    params = {
        "format": "json",
        "limit": limit,
        "start": int(start_ts),
        "stop": int(stop_ts)
    }
    try:
        # 30s timeout per chunk is usually enough for smaller windows
        response = requests.get(RIPE_API_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to fetch chunk {start_ts}-{stop_ts}: {e}")
        return []

def fetch_ripe_data(limit=100000):
    """
    Fetches ping results in parallel to speed up large time windows.
    """
    print(f"Connecting to RIPE Atlas (Measurement {MEASUREMENT_ID})...")
    
    end_time = time.time()
    start_time = end_time - 86400 # 24 hours
    
    # Split 24 hours into 6 chunks (4 hours each) for speed
    chunks = 6
    time_step = (end_time - start_time) / chunks
    intervals = [(start_time + i*time_step, start_time + (i+1)*time_step) for i in range(chunks)]
    
    all_data = []
    
    # Run requests in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=chunks) as executor:
        futures = [executor.submit(fetch_chunk, s, e, limit) for s, e in intervals]
        for future in concurrent.futures.as_completed(futures):
            all_data.extend(future.result())
            
    print(f"Received {len(all_data)} total data points across {chunks} parallel requests.")
    return all_data

def parse_telemetry(raw_data):
    """
    Extracts only the useful signals: Timestamp, RTT, and Probe ID.
    """
    clean_records = []
    
    for record in raw_data:
        # RIPE structure is nested. We need the 'min', 'avg', or 'max' RTT.
        # usually record['result'][0]['rtt']
        
        probe_id = record.get('prb_id')
        timestamp = record.get('timestamp')
        
        # Check if the result actually exists (sometimes probes fail)
        if 'result' in record and len(record['result']) > 0:
            # Iterate through the ping attempts (usually 3 per result)
            for attempt in record['result']:
                if 'rtt' in attempt:
                    clean_records.append({
                        'timestamp': timestamp,
                        'probe_id': probe_id,
                        'rtt': attempt['rtt']
                    })
    
    df = pd.DataFrame(clean_records)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

if __name__ == "__main__":
    # 1. Get Data
    raw_json = fetch_ripe_data()
    
    # 2. Clean Data
    df = parse_telemetry(raw_json)
    
    if not df.empty:
        # 3. Save to disk (Simulating a database for now)
        output_file = "network_telemetry.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} cleaned records to {output_file}")
        print(df.head())
    else:
        print("No valid data found.")