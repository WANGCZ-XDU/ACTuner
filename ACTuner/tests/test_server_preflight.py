"""Validate data, executables, and experiment settings before tuning."""

from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import struct
import sys
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from lrrl import configure_openmp, project_path

from hnsw_backend import assert_no_external_hnsw_process


EFSEARCH_GRID = (10, 20, 30, 40, 50, 80, 100, 200, 300, 600, 1200, 2400)
SUPPORTED_TARGETS = (0.90, 0.92, 0.94, 0.96, 0.98, 0.99)
EXPECTED_BASE_COUNT = 1_000_000
EXPECTED_QUERY_COUNT = 1_000
EXPECTED_DIM = 960
EXPECTED_GT_DIM = 100


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be 0/1 or true/false, got {value!r}")


def _check_vecs(path: str, expected_dim: int, expected_count: int, label: str) -> int:
    file_path = project_path(path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"{label} file not found: {file_path}")

    file_size = file_path.stat().st_size
    if file_size < 4:
        raise ValueError(f"{label} file is empty or truncated: {file_path}")

    with file_path.open("rb") as handle:
        header = handle.read(4)
        dim = struct.unpack("<i", header)[0]
        if dim != expected_dim:
            raise ValueError(
                f"{label} dimension mismatch: {file_path}: {dim} != {expected_dim}"
            )

        record_size = 4 * (dim + 1)
        if file_size % record_size != 0:
            raise ValueError(
                f"{label} file size is not divisible by its record size "
                f"({record_size} bytes): {file_path}"
            )
        count = file_size // record_size
        if count != expected_count:
            raise ValueError(
                f"{label} record-count mismatch: {file_path}: "
                f"{count} != {expected_count}"
            )


        sample_rows = sorted({0, count // 2, count - 1})
        for row_index in sample_rows:
            handle.seek(row_index * record_size)
            row_header = handle.read(4)
            if len(row_header) != 4 or struct.unpack("<i", row_header)[0] != dim:
                raise ValueError(
                    f"{label} row {row_index} has an inconsistent dimension header: "
                    f"{file_path}"
                )

    print(
        f"[OK] {label}: {file_path} "
        f"(count={count}, dim={dim}, bytes={file_size})"
    )
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _check_executable(runtime: dict, key: str) -> tuple[Path, str]:
    configured = runtime.get(key)
    if not configured:
        raise ValueError(f"runtime.{key} is not configured")
    executable = Path(str(configured)).expanduser()
    if not executable.is_absolute():
        executable = PROJECT_DIR / executable
    executable = executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"runtime.{key} not found: {executable}")
    size = executable.stat().st_size
    if size <= 0:
        raise ValueError(f"runtime.{key} is empty: {executable}")
    if not os.access(executable, os.X_OK):
        raise PermissionError(f"runtime.{key} is not executable: {executable}")

    digest = _sha256(executable)
    expected = str(runtime.get(f"{key}_sha256", "")).strip().lower()
    if expected and digest.lower() != expected:
        raise ValueError(
            f"runtime.{key} SHA256 mismatch: {digest} != configured {expected}"
        )
    print(
        f"[OK] {key}: {executable}\n"
        f"     size={size} bytes\n"
        f"     sha256={digest}"
    )
    return executable, digest


def _check_target(cfg: dict) -> float:
    raw = os.environ.get("TARGET_RECALL", cfg["lrrl"].get("R_target"))
    try:
        target = round(float(raw), 2)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid target recall: {raw!r}") from exc
    if target not in SUPPORTED_TARGETS:
        raise ValueError(
            f"Target recall must be one of {SUPPORTED_TARGETS}, got {raw!r}"
        )
    print(f"[OK] target recall: {target:.2f}")
    return target


def main() -> None:
    config_path = PROJECT_DIR / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    for module in ("numpy", "pandas", "torch", "requests", "yaml", "openpyxl"):
        importlib.import_module(module)
        print(f"[OK] import: {module}")

    data = cfg["data"]
    if int(data.get("base_limit", -1)) != EXPECTED_BASE_COUNT:
        raise ValueError(
            f"data.base_limit must be {EXPECTED_BASE_COUNT}, "
            f"got {data.get('base_limit')}"
        )
    if int(data.get("query_limit", -1)) != EXPECTED_QUERY_COUNT:
        raise ValueError(
            f"data.query_limit must be {EXPECTED_QUERY_COUNT}, "
            f"got {data.get('query_limit')}"
        )
    if int(data.get("expected_dim", -1)) != EXPECTED_DIM:
        raise ValueError(
            f"data.expected_dim must be {EXPECTED_DIM}, got {data.get('expected_dim')}"
        )
    if int(data.get("expected_gt_dim", -1)) != EXPECTED_GT_DIM:
        raise ValueError(
            f"data.expected_gt_dim must be {EXPECTED_GT_DIM}, "
            f"got {data.get('expected_gt_dim')}"
        )

    _check_vecs(
        os.environ.get("GIST_BASE_PATH", data["base_path"]),
        EXPECTED_DIM,
        EXPECTED_BASE_COUNT,
        "GIST base",
    )
    query_count = _check_vecs(
        os.environ.get("GIST_QUERY_PATH", data["query_path"]),
        EXPECTED_DIM,
        EXPECTED_QUERY_COUNT,
        "GIST query",
    )
    gt_count = _check_vecs(
        os.environ.get("GIST_GROUNDTRUTH_PATH", data["groundtruth_path"]),
        EXPECTED_GT_DIM,
        EXPECTED_QUERY_COUNT,
        "GIST groundtruth",
    )
    if query_count != gt_count:
        raise ValueError(
            f"Query/groundtruth count mismatch: query={query_count}, gt={gt_count}"
        )

    hnsw = cfg["hnsw"]
    if int(hnsw.get("recall_k", -1)) != 10:
        raise ValueError(f"hnsw.recall_k must be 10, got {hnsw.get('recall_k')}")
    for key in ("efSearch_train", "efSearch_full"):
        observed = tuple(int(value) for value in hnsw.get(key, ()))
        if observed != EFSEARCH_GRID:
            raise ValueError(
                f"hnsw.{key} must match the test_search output grid: "
                f"{observed} != {EFSEARCH_GRID}"
            )
        print(f"[OK] {key}: {list(observed)}")

    lrrl = cfg["lrrl"]
    budget_protocol = (
        int(lrrl.get("n_total_budget", -1)),
        int(lrrl.get("n_warmup", -1)),
        int(lrrl.get("n_real_per_step", -1)),
    )
    if budget_protocol != (30, 10, 2):
        raise ValueError(
            "lrrl budget must be exactly "
            "(n_total_budget, n_warmup, n_real_per_step)=(30,10,2), "
            f"got {budget_protocol}"
        )
    print("[OK] real-evaluation budget: 30 = 10 warmup + 20 online")

    runtime = cfg["runtime"]
    if int(runtime.get("train_query_sample", -1)) != EXPECTED_QUERY_COUNT:
        raise ValueError(
            f"runtime.train_query_sample must be {EXPECTED_QUERY_COUNT}, "
            f"got {runtime.get('train_query_sample')}"
        )
    if int(runtime.get("search_repeat", -1)) != 5:
        raise ValueError(
            f"runtime.search_repeat must be 5, got {runtime.get('search_repeat')}"
        )
    if runtime.get("warmup_cache"):
        raise ValueError(
            "runtime.warmup_cache must be empty: every warmup point must be "
            "measured by the HNSW evaluator"
        )
    work_dir = Path(str(runtime.get(
        "hnsw_work_dir", "runs/hnsw_eval_tmp"
    )))
    if not work_dir.is_absolute():
        work_dir = PROJECT_DIR / work_dir
    disk_probe = work_dir
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    free_gb = shutil.disk_usage(disk_probe).free / (1024 ** 3)
    required_free_gb = float(runtime.get("min_free_disk_gb", 20))
    if free_gb < required_free_gb:
        raise RuntimeError(
            f"Insufficient free disk near {work_dir}: {free_gb:.1f} GiB "
            f"< required {required_free_gb:.1f} GiB"
        )
    print(
        f"[OK] free disk near evaluation work dir: {free_gb:.1f} GiB "
        f"(minimum {required_free_gb:.1f} GiB)"
    )
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        available_kib = None
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_kib = int(line.split()[1])
                break
        if available_kib is not None:
            available_gb = available_kib / (1024 ** 2)
            required_ram_gb = float(runtime.get(
                "min_available_ram_gb", 24
            ))
            if available_gb < required_ram_gb:
                raise RuntimeError(
                    f"Insufficient available RAM: {available_gb:.1f} GiB "
                    f"< required {required_ram_gb:.1f} GiB"
                )
            print(
                f"[OK] available RAM: {available_gb:.1f} GiB "
                f"(minimum {required_ram_gb:.1f} GiB)"
            )

    omp_threads = configure_openmp(runtime)
    print(f"[OK] OpenMP threads: {omp_threads}")

    build_exe, build_sha = _check_executable(runtime, "hnsw_build_exe")
    search_exe, search_sha = _check_executable(runtime, "hnsw_search_exe")
    if build_exe == search_exe or build_sha == search_sha:
        raise ValueError(
            "test_index and test_search unexpectedly resolve to the same executable"
        )
    assert_no_external_hnsw_process()
    print("[OK] no external test_index/test_search process is running")

    _check_target(cfg)

    skip_llm = _env_flag("SKIP_LLM", default=False)
    llm = cfg.get("llm", {})
    for key in ("connect_timeout_s", "request_timeout_s"):
        if float(llm.get(key, 0)) <= 0:
            raise ValueError(f"llm.{key} must be positive")
    if int(llm.get("retries", 0)) <= 0:
        raise ValueError("llm.retries must be positive")
    if float(llm.get("retry_backoff_s", -1)) < 0:
        raise ValueError("llm.retry_backoff_s must be non-negative")
    api_env = cfg.get("llm", {}).get("api_key_env", "DOUBAO_API_KEY")
    if skip_llm:
        print("[OK] SKIP_LLM=1: Doubao will be bypassed; no API key is required")
    elif not os.environ.get(api_env):
        raise RuntimeError(
            f"{api_env} is not set. Export it, or run with SKIP_LLM=1."
        )
    else:
        print(f"[OK] {api_env} is set (value not displayed)")

    print("Preflight passed. No API request was sent and nothing was compiled.")


if __name__ == "__main__":
    main()
