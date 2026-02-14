# Cron Job Setup for NetProphet Cache

This document explains how to set up the background cron job to keep the prediction cache fresh.

## How It Works

1. **Cron job** runs `update_cache.py` every 10 minutes
2. Script fetches RIPE data and saves prediction to `/tmp/last_prediction.json`
3. **Flask app** reads from cache file instantly (no API call)
4. Users get instant page loads with data ≤10 minutes old

## Setup on EC2

### 1. SSH into your EC2 instance

```bash
ssh -i your-key.pem ubuntu@54.215.23.12
```

### 2. Find the path to update_cache.py in the container

```bash
# First, get a shell inside the running container
sudo docker exec -it netprophet-main /bin/bash

# Inside container, find the path
pwd
# Should be: /
ls -la update_cache.py
# Should exist

# Make it executable
chmod +x update_cache.py

# Test it manually
python3 update_cache.py

# Exit container
exit
```

### 3. Create cron job on the HOST (not in container)

```bash
# Edit crontab
crontab -e

# Add this line (runs every 10 minutes):
*/10 * * * * sudo docker exec netprophet-main python3 /update_cache.py >> /var/log/netprophet_cache.log 2>&1

# Save and exit (Ctrl+X, then Y, then Enter)
```

### 4. Verify cron job is running

```bash
# Check cron is running
sudo systemctl status cron

# Wait 10 minutes, then check the log
tail -f /var/log/netprophet_cache.log
```

You should see output like:
```
[2026-02-13 10:00:01] Starting cache update...
[2026-02-13 10:00:45] Cache updated successfully!
  Probability: 12.34%
  Latency: 23.4 ms
```

### 5. Test the website

Visit http://54.215.23.12/

- **First load after deploy:** Might be slow (fetching fresh data)
- **After cron runs:** Instant! Just reads from file
- **Every subsequent visit:** Instant! Data refreshed every 10 minutes

## Troubleshooting

**Cron not running?**
```bash
# Check cron service
sudo systemctl status cron
sudo systemctl start cron
```

**Script failing?**
```bash
# Run manually to see errors
sudo docker exec netprophet-main python3 /update_cache.py
```

**Cache file not being created?**
```bash
# Check inside container
sudo docker exec netprophet-main ls -la /tmp/last_prediction.json
sudo docker exec netprophet-main cat /tmp/last_prediction.json
```

## Alternative: Run Every 5 Minutes

For fresher data, use:
```
*/5 * * * * sudo docker exec netprophet-main python3 /update_cache.py >> /var/log/netprophet_cache.log 2>&1
```

## Memory Benefits

- **Before:** Every page load = RIPE API call + processing (memory spike)
- **After:** Only cron job uses memory, users just read a small file
- **Result:** Stable memory usage, no crashes on t3.micro
