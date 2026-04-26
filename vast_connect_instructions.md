# Vast Jupyter CLI Fix - 2026-04-25

Purpose: prevent agents from wasting time on stale Cloudflare/Jupyter tunnel
URLs while the pod is burning money.

## Problem

`scripts/vast_jupyter_exec.py` used to take the first `Default Tunnel started
for Jupyter` URL from Vast logs and trust it. Vast logs can contain multiple
Jupyter tunnel URLs, and Cloudflare quick tunnels can go stale or fail DNS.

Observed failure:

- helper selected a stale/bad `trycloudflare.com` Jupyter URL
- DNS failed before the Jupyter API call
- agents could waste time manually chasing tunnel strings

## Fix

`scripts/vast_jupyter_exec.py` now:

1. collects all Jupyter tunnel URLs from Vast logs;
2. collects bearer-token candidates;
3. probes candidates newest-to-oldest with `GET /api/kernels`;
4. skips stale DNS/API failures automatically;
5. only runs a command against a tunnel that actually answers the Jupyter API.

It prints `JUPYTER_DISCOVERY_SKIPPED=...` when bad candidates are skipped. That
is not a failure.

## Required Command Pattern

Use this pattern for pod commands:

```bash
/home/frosty40/miniconda3/bin/python3 scripts/vast_jupyter_exec.py 34405931 "cd /workspace/SOTA_FINAL && date -Is && pwd" --tail 1200 --timeout 120 --probe-timeout 5
```

Use this pattern for run monitoring:

```bash
/home/frosty40/miniconda3/bin/python3 scripts/vast_jupyter_exec.py 34405931 "cd /workspace/SOTA_FINAL && date -Is && pgrep -af 'torchrun|pr1493_decompressed' || true && nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits && tail -120 <log_path>" --tail 1200 --timeout 120 --probe-timeout 5
```

## Stop Rule

Do not manually chase Cloudflare URLs. If the helper fails twice with
`--tail 1200 --probe-timeout 5`, record the failure and ask Jordan before using
direct SSH.

