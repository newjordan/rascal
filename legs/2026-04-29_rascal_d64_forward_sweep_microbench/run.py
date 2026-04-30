#!/usr/bin/env python3
"""D64 real-shape Rascal forward-config sweep against FA3.

This is a condition-locked microbench only: no training, no tokenizer/data
dependency, and no active run state touched.
"""

from __future__ import annotations

import json
import os
import re
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path


TEST = {
    "test_id": "2026-04-29_rascal_d64_forward_sweep_microbench",
    "date": "2026-04-29",
    "hypothesis": "The D64 whale-fwd/FA3-bwd path was 1.5% slower than FA3 at the exact Rascal per-rank shape; tune only the forward-side configs on B=48,T=2048,H=8,KV=4,D=64 before any training integration.",
    "parent_baseline": "notes/2026-04-29_kernel_decision_queue.md plus /home/frosty40/SOTA_FINAL/legs/2026-04-18_whale_cross_gpu_validation_prep_t2048/winner_anchor.json.",
    "parent_benchmark_id": "RUNPY_RASCAL_D64_REALSHAPE_KERNEL_MICROBENCH",
    "standard_label": "mechanics_1x_forward_kernel_scout",
    "gpu_count": 1,
    "build_seconds": 300,
    "eval_seconds": 0,
    "size_cap_bytes": 16_000_000,
}

MODEL = {
    "donor_lineage": "rascal_sp4096_raphe1p1_d64",
    "source_body": "quarantine/2026-04-28_tokenizer_provenance/candidate_submissions/Raphe_1.1/train_gpt_8xgpu.py",
    "source_sha256": "b93227b9f394b81fa77c412e0ecc394f17fb9effb250b96fe77903b9e1687d39",
    "seed": 444,
    "num_layers": 10,
    "model_dim": 512,
    "num_heads": 8,
    "num_kv_heads": 4,
    "head_dim": 64,
    "xsa_last_n": 11,
    "train_seq_len": 2048,
    "train_microbatch_per_rank": 48,
    "dtype": "bfloat16",
    "tokenizer_path": "/workspace/data/tokenizers/fineweb_4096_bpe.model",
    "dataset_dir": "/workspace/data/datasets/fineweb10B_sp4096",
}

KERNEL = {
    "attn_mode": "hybrid_forward_sweep",
    "hybrid_backend": "fa3_bwd",
    "benchmark_id": "RUNPY_RASCAL_D64_FORWARD_SWEEP_MICROBENCH",
    "shape": "B=48,T=2048,H=8,KV=4,D=64",
    "causal": 1,
    "warmup_iters": 3,
    "timed_iters": 20,
    "kernel_source_path": "vault/whale_kernel_triton.py",
    "kernel_source_sha256": "4d8340fc2ed9a656527d25bc8fbd7dfddc691e9ece287ff35017c9b0ee4d26f8",
    "source_evidence": "/workspace/SOTA_FINAL/legs/2026-04-18_whale_cross_gpu_validation_prep_t2048/evidence/cross_primary_validate_dq_128_64_8_5_dkdv_128_64_8_3_mr_256_qmr_1984_20260421_000841.json",
    "profiles": [
        {
            "profile_id": "fwd_auto_current",
            "whale_fwd_variant": "",
            "whale_fwd_config": "",
            "whale_fwd_maxnreg": "",
            "whale_fwd_tma_config": "",
            "whale_fwd_tma_maxnreg": "",
            "reason": "current autotuned default for the exact Rascal D64 shape",
        },
        {
            "profile_id": "fwd_forced_64_64_w4_s3_mr160",
            "whale_fwd_variant": "",
            "whale_fwd_config": "64,64,4,3",
            "whale_fwd_maxnreg": "160",
            "whale_fwd_tma_config": "",
            "whale_fwd_tma_maxnreg": "",
            "reason": "D64 small-tile non-TMA winner family from fwd evidence",
        },
        {
            "profile_id": "fwd_forced_64_64_w4_s3_mr128",
            "whale_fwd_variant": "",
            "whale_fwd_config": "64,64,4,3",
            "whale_fwd_maxnreg": "128",
            "whale_fwd_tma_config": "",
            "whale_fwd_tma_maxnreg": "",
            "reason": "same small tile with higher occupancy pressure",
        },
        {
            "profile_id": "fwd_forced_64_64_w4_s2_mr160",
            "whale_fwd_variant": "",
            "whale_fwd_config": "64,64,4,2",
            "whale_fwd_maxnreg": "160",
            "whale_fwd_tma_config": "",
            "whale_fwd_tma_maxnreg": "",
            "reason": "shallower pipeline on the small-tile family",
        },
        {
            "profile_id": "fwd_forced_64_64_w8_s3_mr160",
            "whale_fwd_variant": "",
            "whale_fwd_config": "64,64,8,3",
            "whale_fwd_maxnreg": "160",
            "whale_fwd_tma_config": "",
            "whale_fwd_tma_maxnreg": "",
            "reason": "more warps on the small-tile family",
        },
        {
            "profile_id": "fwd_forced_128_64_w4_s3_mr160",
            "whale_fwd_variant": "",
            "whale_fwd_config": "128,64,4,3",
            "whale_fwd_maxnreg": "160",
            "whale_fwd_tma_config": "",
            "whale_fwd_tma_maxnreg": "",
            "reason": "larger M tile from primary autotune grid",
        },
        {
            "profile_id": "fwd_forced_128_64_w4_s2_mr160",
            "whale_fwd_variant": "",
            "whale_fwd_config": "128,64,4,2",
            "whale_fwd_maxnreg": "160",
            "whale_fwd_tma_config": "",
            "whale_fwd_tma_maxnreg": "",
            "reason": "larger M tile with shallower pipeline",
        },
        {
            "profile_id": "fwd_forced_128_128_w4_s3_mr224",
            "whale_fwd_variant": "",
            "whale_fwd_config": "128,128,4,3",
            "whale_fwd_maxnreg": "224",
            "whale_fwd_tma_config": "",
            "whale_fwd_tma_maxnreg": "",
            "reason": "big tile with larger register cap from the expanded fwd grid",
        },
        {
            "profile_id": "fwd_tma_auto",
            "whale_fwd_variant": "tma",
            "whale_fwd_config": "",
            "whale_fwd_maxnreg": "",
            "whale_fwd_tma_config": "",
            "whale_fwd_tma_maxnreg": "",
            "reason": "TMA forward variant autotune on the exact D64 shape",
        },
        {
            "profile_id": "fwd_tma_64_64_w4_s3_mr160",
            "whale_fwd_variant": "tma",
            "whale_fwd_config": "",
            "whale_fwd_maxnreg": "",
            "whale_fwd_tma_config": "64,64,4,3",
            "whale_fwd_tma_maxnreg": "160",
            "reason": "forced TMA small-tile candidate",
        },
        {
            "profile_id": "fwd_tma_128_64_w4_s3_mr160",
            "whale_fwd_variant": "tma",
            "whale_fwd_config": "",
            "whale_fwd_maxnreg": "",
            "whale_fwd_tma_config": "128,64,4,3",
            "whale_fwd_tma_maxnreg": "160",
            "reason": "forced TMA larger-M candidate",
        },
    ],
}

RESULT_HEADER = (
    "test_id\tdate\tgpu_count\tbuild_seconds\teval_seconds\tsize_cap_bytes\t"
    "raw_step\traw_bpb\tpost_ema_bpb\tquant_bpb\tsliding_bpb\t"
    "ttt_bpb\ttotal_bytes\tdecision\tlog_path\trun_py\n"
)

MICRO_HEADER = (
    "profile_id\tgpu\tbench\tmode\tmedian_ms\tmin_ms\tp90_ms\tmean_ms\t"
    "ratio_vs_fa3_median\terror\n"
)

CHILD_CODE = r'''
import json
import os
import time
import traceback

cfg = json.loads(os.environ["BENCH_CONFIG_JSON"])
root = cfg["root"]
os.chdir(root)

import torch
from flash_attn_interface import flash_attn_func
from vault.whale_kernel_triton import whale_fwd_fa3_bwd

torch.manual_seed(int(cfg["seed"]) + int(cfg["gpu"]))
torch.cuda.set_device(0)
device = torch.device("cuda")
dtype = torch.bfloat16

B = int(cfg["B"])
T = int(cfg["T"])
H = int(cfg["H"])
KV = int(cfg["KV"])
D = int(cfg["D"])
warmup = int(cfg["warmup_iters"])
timed = int(cfg["timed_iters"])

q0 = torch.randn(B, T, H, D, device=device, dtype=dtype)
k0 = torch.randn(B, T, KV, D, device=device, dtype=dtype)
v0 = torch.randn(B, T, KV, D, device=device, dtype=dtype)


def stats(times):
    times = sorted(times)
    return {
        "median_ms": times[len(times) // 2],
        "min_ms": times[0],
        "p90_ms": times[min(len(times) - 1, int(len(times) * 0.9))],
        "mean_ms": sum(times) / len(times),
    }


def bench_full(fn):
    for _ in range(warmup):
        q = q0.clone().requires_grad_(True)
        k = k0.clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)
        out = fn(q, k, v)
        out.sum().backward()
    torch.cuda.synchronize()
    times = []
    for _ in range(timed):
        q = q0.clone().requires_grad_(True)
        k = k0.clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(q, k, v)
        out.sum().backward()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return stats(times)


def bench_fwd(fn):
    with torch.no_grad():
        for _ in range(warmup):
            out = fn(q0, k0, v0)
            _ = out.mean()
        torch.cuda.synchronize()
        times = []
        for _ in range(timed):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(q0, k0, v0)
            _ = out.mean()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return stats(times)


def whale_fn(q, k, v):
    return whale_fwd_fa3_bwd(q, k, v, causal=True)


def fa3_fn(q, k, v):
    return flash_attn_func(q, k, v, causal=True)


result = {
    "profile_id": cfg["profile_id"],
    "gpu": cfg["gpu"],
    "shape": {"B": B, "T": T, "H": H, "KV": KV, "D": D},
    "env": {
        "WHALE_FWD_VARIANT": os.environ.get("WHALE_FWD_VARIANT", ""),
        "WHALE_FWD_CONFIG": os.environ.get("WHALE_FWD_CONFIG", ""),
        "WHALE_FWD_MAXNREG": os.environ.get("WHALE_FWD_MAXNREG", ""),
        "WHALE_FWD_TMA_CONFIG": os.environ.get("WHALE_FWD_TMA_CONFIG", ""),
        "WHALE_FWD_TMA_MAXNREG": os.environ.get("WHALE_FWD_TMA_MAXNREG", ""),
        "WHALE_BWD_VARIANT": os.environ.get("WHALE_BWD_VARIANT", ""),
    },
}

try:
    result["whale_fa3_bwd"] = bench_full(whale_fn)
except Exception as exc:
    result["whale_error"] = repr(exc)
    result["whale_traceback"] = traceback.format_exc()

try:
    result["fa3"] = bench_full(fa3_fn)
except Exception as exc:
    result["fa3_error"] = repr(exc)
    result["fa3_traceback"] = traceback.format_exc()

if "whale_fa3_bwd" in result and "fa3" in result:
    result["ratio_vs_fa3_full_median"] = (
        result["whale_fa3_bwd"]["median_ms"] / result["fa3"]["median_ms"]
    )

try:
    result["whale_fwd"] = bench_fwd(whale_fn)
except Exception as exc:
    result["whale_fwd_error"] = repr(exc)
    result["whale_fwd_traceback"] = traceback.format_exc()

try:
    result["fa3_fwd"] = bench_fwd(fa3_fn)
except Exception as exc:
    result["fa3_fwd_error"] = repr(exc)
    result["fa3_fwd_traceback"] = traceback.format_exc()

if "whale_fwd" in result and "fa3_fwd" in result:
    result["ratio_vs_fa3_fwd_median"] = (
        result["whale_fwd"]["median_ms"] / result["fa3_fwd"]["median_ms"]
    )

print("JSON_RESULT " + json.dumps(result, sort_keys=True), flush=True)
'''


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def leg_dir() -> Path:
    return Path(__file__).resolve().parent


def resolved_config(root: Path, log_path: Path, result_path: Path, micro_path: Path) -> dict[str, object]:
    return {
        "test": TEST,
        "model": MODEL,
        "kernel": KERNEL,
        "repo_root": str(root),
        "run_py": str(Path(__file__).resolve()),
        "log_path": str(log_path),
        "result_path": str(result_path),
        "microbench_path": str(micro_path),
    }


def preflight(root: Path) -> None:
    required = [
        root / "vault/whale_kernel_triton.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("FATAL: missing required paths:\n" + "\n".join(missing))
    if not Path("/venv/main/bin/python3").is_file():
        raise SystemExit("FATAL: /venv/main/bin/python3 not found")


def child_env(root: Path, profile: dict[str, str], gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = "/venv/main/bin:" + env.get("PATH", "")
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["WHALE_BWD_VARIANT"] = "fused_delta_tma"
    env["WHALE_FUSED_DELTA_T_MAX"] = "3072"
    if profile.get("whale_fwd_variant"):
        env["WHALE_FWD_VARIANT"] = profile["whale_fwd_variant"]
    else:
        env.pop("WHALE_FWD_VARIANT", None)
    if profile.get("whale_fwd_config"):
        env["WHALE_FWD_CONFIG"] = profile["whale_fwd_config"]
    else:
        env.pop("WHALE_FWD_CONFIG", None)
    if profile.get("whale_fwd_maxnreg"):
        env["WHALE_FWD_MAXNREG"] = profile["whale_fwd_maxnreg"]
    else:
        env.pop("WHALE_FWD_MAXNREG", None)
    if profile.get("whale_fwd_tma_config"):
        env["WHALE_FWD_TMA_CONFIG"] = profile["whale_fwd_tma_config"]
    else:
        env.pop("WHALE_FWD_TMA_CONFIG", None)
    if profile.get("whale_fwd_tma_maxnreg"):
        env["WHALE_FWD_TMA_MAXNREG"] = profile["whale_fwd_tma_maxnreg"]
    else:
        env.pop("WHALE_FWD_TMA_MAXNREG", None)
    env.pop("WHALE_BWD_Q_CONFIG", None)
    env.pop("WHALE_BWD_Q_MAXNREG", None)
    env.pop("WHALE_BWD_KV_TMA_CONFIG", None)
    env.pop("WHALE_BWD_KV_TMA_MAXNREG", None)
    env["BENCH_CONFIG_JSON"] = json.dumps(
        {
            "root": str(root),
            "seed": MODEL["seed"],
            "gpu": gpu,
            "profile_id": profile["profile_id"],
            "B": MODEL["train_microbatch_per_rank"],
            "T": MODEL["train_seq_len"],
            "H": MODEL["num_heads"],
            "KV": MODEL["num_kv_heads"],
            "D": MODEL["head_dim"],
            "warmup_iters": KERNEL["warmup_iters"],
            "timed_iters": KERNEL["timed_iters"],
        },
        sort_keys=True,
    )
    return env


def launch_profile(root: Path, profile: dict[str, str], logs: Path) -> list[dict[str, object]]:
    procs: list[tuple[int, subprocess.Popen[str], Path]] = []
    for gpu in range(TEST["gpu_count"]):
        worker_log = logs / f"{profile['profile_id']}_gpu{gpu}.log"
        env = child_env(root, profile, gpu)
        proc = subprocess.Popen(
            ["/venv/main/bin/python3", "-c", CHILD_CODE],
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            start_new_session=True,
        )
        procs.append((gpu, proc, worker_log))

    deadline = time.time() + 900
    results: list[dict[str, object]] = []
    for gpu, proc, worker_log in procs:
        assert proc.stdout is not None
        with worker_log.open("w", encoding="utf-8") as log:
            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ)
            try:
                while True:
                    if time.time() > deadline:
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        result = {
                            "profile_id": profile["profile_id"],
                            "gpu": gpu,
                            "error": "timeout",
                        }
                        results.append(result)
                        break
                    events = selector.select(timeout=1.0)
                    for key, _ in events:
                        line = key.fileobj.readline()
                        if line:
                            sys.stdout.write(line)
                            log.write(line)
                            log.flush()
                            if line.startswith("JSON_RESULT "):
                                results.append(json.loads(line[len("JSON_RESULT "):]))
                    if proc.poll() is not None:
                        tail = proc.stdout.read()
                        if tail:
                            sys.stdout.write(tail)
                            log.write(tail)
                            for line in tail.splitlines():
                                if line.startswith("JSON_RESULT "):
                                    results.append(json.loads(line[len("JSON_RESULT "):]))
                        if proc.returncode != 0 and not any(
                            r.get("profile_id") == profile["profile_id"] and r.get("gpu") == gpu
                            for r in results
                        ):
                            results.append(
                                {
                                    "profile_id": profile["profile_id"],
                                    "gpu": gpu,
                                    "error": f"rc_{proc.returncode}",
                                }
                            )
                        break
            finally:
                selector.unregister(proc.stdout)
                selector.close()
                proc.stdout.close()
    return results


def append_result(result_path: Path, log_path: Path, decision: str) -> None:
    if not result_path.exists():
        result_path.write_text(RESULT_HEADER, encoding="utf-8")
    row = [
        TEST["test_id"],
        TEST["date"],
        str(TEST["gpu_count"]),
        str(TEST["build_seconds"]),
        str(TEST["eval_seconds"]),
        str(TEST["size_cap_bytes"]),
        "NA",
        "NA",
        "NA",
        "NA",
        "NA",
        "NA",
        "NA",
        decision,
        str(log_path),
        str(Path(__file__).resolve()),
    ]
    with result_path.open("a", encoding="utf-8") as f:
        f.write("\t".join(row) + "\n")


def write_microbench_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(MICRO_HEADER, encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            profile_id = str(row.get("profile_id", "NA"))
            gpu = str(row.get("gpu", "NA"))
            ratio_full = row.get("ratio_vs_fa3_full_median", "NA")
            ratio_fwd = row.get("ratio_vs_fa3_fwd_median", "NA")
            for bench, modes, ratio in (
                ("full", ("whale_fa3_bwd", "fa3"), ratio_full),
                ("fwd", ("whale_fwd", "fa3_fwd"), ratio_fwd),
            ):
                for mode in modes:
                    stats = row.get(mode)
                    if mode == "whale_fa3_bwd":
                        error_key = "whale_error"
                    elif mode == "fa3":
                        error_key = "fa3_error"
                    elif mode == "whale_fwd":
                        error_key = "whale_fwd_error"
                    else:
                        error_key = "fa3_fwd_error"
                    error = row.get("error") or row.get(error_key) or ""
                    if isinstance(stats, dict):
                        values = [
                            profile_id,
                            gpu,
                            bench,
                            mode,
                            f"{float(stats['median_ms']):.6f}",
                            f"{float(stats['min_ms']):.6f}",
                            f"{float(stats['p90_ms']):.6f}",
                            f"{float(stats['mean_ms']):.6f}",
                            f"{float(ratio):.6f}" if isinstance(ratio, (int, float)) else "NA",
                            str(error),
                        ]
                    else:
                        values = [profile_id, gpu, bench, mode, "NA", "NA", "NA", "NA", "NA", str(error)]
                    f.write("\t".join(values) + "\n")


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    preflight_only = "--preflight-only" in sys.argv
    root = repo_root()
    leg = leg_dir()
    logs = leg / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    run_id = TEST["test_id"] + "_s" + str(MODEL["seed"])
    log_path = logs / (run_id + ".log")
    result_path = leg / "results.tsv"
    micro_path = leg / "microbench.tsv"
    json_path = leg / "microbench.json"
    config_path = leg / "resolved_config.json"

    config = resolved_config(root, log_path, result_path, micro_path)
    print(json.dumps(config, indent=2, sort_keys=True))
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if dry_run:
        print("DRY_RUN: wrote resolved config; not launching")
        return 0
    preflight(root)
    if preflight_only:
        print("PREFLIGHT_OK: not launching")
        return 0

    all_rows: list[dict[str, object]] = []
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(config, indent=2, sort_keys=True) + "\n")
        log.flush()
        for profile in KERNEL["profiles"]:
            msg = f"PROFILE_START {profile['profile_id']} {profile['reason']}\n"
            print(msg, end="")
            log.write(msg)
            log.flush()
            rows = launch_profile(root, profile, logs)
            all_rows.extend(rows)
            for row in rows:
                line = "PROFILE_RESULT " + json.dumps(row, sort_keys=True) + "\n"
                print(line, end="")
                log.write(line)
                log.flush()

    write_microbench_rows(micro_path, all_rows)
    json_path.write_text(json.dumps(all_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    decision = "kernel_microbench_complete"
    if any("error" in row or "whale_error" in row or "fa3_error" in row for row in all_rows):
        decision = "kernel_microbench_partial"
    append_result(result_path, log_path, decision)
    print(f"MICROBENCH_TSV: {micro_path}")
    print(f"MICROBENCH_JSON: {json_path}")
    print(f"RESULTS: {result_path}")
    print(f"LOG: {log_path}")
    return 0 if decision == "kernel_microbench_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
