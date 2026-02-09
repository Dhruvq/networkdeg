# Network Degradation Prediction Using Real Internet Telemetry
## Overview
This project builds an end-to-end machine learning system to predict short-term network performance degradation using real-world internet measurement data from RIPE Atlas.

The goal is to identify early warning signals before users experience severe latency or packet loss by learning patterns in recent network behavior. Rather than relying on hand-tuned thresholds, the system uses supervised machine learning to output a probabilistic risk score indicating the likelihood of near-future degradation.

```mermaid
graph LR
    A[RIPE Atlas API] -->|Raw JSON Telemetry| B(Data Ingestion Service)
    B -->|Cleaning & Forward Fill| C{Feature Engineering}
    C -->|Jitter, Hops, RTT| D[XGBoost Inference Engine]
    D -->|Degradation Probability| E[Dashboard / Alert System]
    D -->|Feature Importance| F[Root Cause Analysis]
```

## Key Features

## Tech Stack

## Methodology

## Results