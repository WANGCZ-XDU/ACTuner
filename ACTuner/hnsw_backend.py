"""Execute HNSW benchmarks and interpolate QPS at a target recall."""

from __future__ import annotations

import csv
import atexit
import hashlib
import math
import os
import struct
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np


EFSEARCH_GRID = (10, 20, 30, 40, 50, 80, 100, 200, 300, 600, 1200, 2400)
RECALL_K = 10
SEARCH_REPEAT = 5
DEFAULT_GIST_QUERY_COUNT = 1000
SUPPORTED_TARGETS = (0.90, 0.92, 0.94, 0.96, 0.98, 0.99)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_no_external_hnsw_process() -> None:
    proc_root = Path("/proc")
    if os.name != "posix" or not proc_root.is_dir():
        return
    current_pid = os.getpid()
    offenders = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == current_pid:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
            argv0 = raw.split(b"\0", 1)[0].decode("utf-8", errors="ignore")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if Path(argv0).name in {"test_index", "test_search"}:
            offenders.append((entry.name, argv0))
    if offenders:
        detail = ", ".join(f"pid={pid}:{cmd}" for pid, cmd in offenders)
        raise RuntimeError(
            "Another test_index/test_search process is already running and "
            f"would invalidate QPS: {detail}"
        )


def count_vecs_records(path: str | Path) -> tuple[int, int]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(4)
    if len(header) != 4:
        raise ValueError(f"Empty or truncated vecs file: {path}")
    dim = struct.unpack("<i", header)[0]
    if dim <= 0:
        raise ValueError(f"Invalid vecs dimension in {path}: {dim}")
    record_size = 4 * (dim + 1)
    if size % record_size != 0:
        raise ValueError(f"Invalid vecs file size for dimension {dim}: {path}")
    return size // record_size, dim


def normalize_efsearch_subset(values: Iterable[int] | None) -> tuple[int, ...]:
    if values is None:
        return EFSEARCH_GRID
    normalized = tuple(int(value) for value in values)
    if not normalized:
        raise ValueError("efSearch interpolation grid must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate efSearch values are not allowed: {normalized}")
    unknown = [value for value in normalized if value not in EFSEARCH_GRID]
    if unknown:
        raise ValueError(
            f"efSearch values {unknown} are not emitted by test_search; "
            f"available={list(EFSEARCH_GRID)}"
        )
    expected_order = tuple(value for value in EFSEARCH_GRID if value in normalized)
    if normalized != expected_order:
        raise ValueError(
            "efSearch interpolation grid must preserve the evaluator output order: "
            f"{normalized} != {expected_order}"
        )
    return normalized


def _curve_for_interpolation(
    raw_results: list[tuple[int, float, float]],
    selected_efsearch: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, float, float]]]:
    selected_set = set(selected_efsearch)
    selected = [row for row in raw_results if row[0] in selected_set]
    observed = tuple(row[0] for row in selected)
    if observed != selected_efsearch:
        raise ValueError(
            f"Selected efSearch rows mismatch: {observed} != {selected_efsearch}"
        )


    indexed = [
        (float(ef_search), float(recall), float(qps), index)
        for index, (ef_search, recall, qps) in enumerate(selected)
    ]
    indexed.sort(key=lambda item: item[1])
    unique = []
    seen_recalls = set()
    for row in indexed:
        if row[1] in seen_recalls:
            continue
        seen_recalls.add(row[1])
        unique.append(row)
    if not unique:
        raise ValueError("No valid rows remain for interpolation")
    return (
        np.asarray([row[0] for row in unique], dtype=float),
        np.asarray([row[1] for row in unique], dtype=float),
        np.asarray([row[2] for row in unique], dtype=float),
        selected,
    )


def read_performance_curve(
    path: str | Path,
    selected_efsearch: Iterable[int] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[int, float, float]],
    list[tuple[int, float, float]],
]:
    path = Path(path)
    full_results: list[tuple[int, float, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 3:
                raise ValueError(
                    f"{path}: line {line_number} must contain efSearch,recall,qps"
                )
            try:
                ef_raw, recall, qps = map(float, row)
            except ValueError as exc:
                raise ValueError(
                    f"{path}: non-numeric row {line_number}: {row!r}"
                ) from exc
            if not all(math.isfinite(value) for value in (ef_raw, recall, qps)):
                raise ValueError(f"{path}: non-finite value on line {line_number}")
            ef_search = int(ef_raw)
            if float(ef_search) != ef_raw:
                raise ValueError(f"{path}: efSearch must be integral on line {line_number}")
            if not 0.0 <= recall <= 1.0:
                raise ValueError(f"{path}: invalid recall on line {line_number}: {recall}")
            if qps <= 0.0:
                raise ValueError(f"{path}: QPS must be positive on line {line_number}: {qps}")
            full_results.append((ef_search, float(recall), float(qps)))

    observed = tuple(row[0] for row in full_results)
    if observed != EFSEARCH_GRID:
        raise ValueError(
            f"{path}: efSearch mismatch: {observed} != {EFSEARCH_GRID}"
        )
    selected = normalize_efsearch_subset(selected_efsearch)
    ef_search, recalls, qps, selected_rows = _curve_for_interpolation(
        full_results, selected
    )
    return ef_search, recalls, qps, selected_rows, full_results


def interpolate_qps(
    target: float,
    ef_search: np.ndarray,
    recalls: np.ndarray,
    qps: np.ndarray,
) -> dict[str, float | bool]:
    """Interpolate QPS within the measured recall range."""
    target = float(target)
    if not math.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError(f"Invalid target recall: {target!r}")
    min_recall = float(recalls[0])
    max_recall = float(recalls[-1])
    target_reached = target <= max_recall + 1e-12
    index = int(np.searchsorted(recalls, target, side="left"))
    if len(recalls) == 1:
        left_recall = right_recall = min_recall
    elif index <= 0:
        left_recall = right_recall = min_recall
    elif index >= len(recalls):
        left_recall = float(recalls[-2])
        right_recall = max_recall
    else:
        left_recall = float(recalls[index - 1])
        right_recall = float(recalls[index])
    return {
        "qps_at_target": (
            float(np.interp(target, recalls, qps)) if target_reached else 0.0
        ),
        "efSearch_interp": (
            float(np.interp(target, recalls, ef_search)) if target_reached else 0.0
        ),
        "target_reached": target_reached,
        "min_recall": min_recall,
        "max_recall": max_recall,
        "left_recall": left_recall,
        "right_recall": right_recall,
    }


def _read_internal_build_time(path: Path) -> float | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            return None
        value = float(rows[0]["Time(s)"])
        return value if math.isfinite(value) and value >= 0.0 else None
    except (KeyError, TypeError, ValueError):
        return None


class HNSWEvaluator:

    """Build an index and measure its recall-QPS curve."""
    def __init__(
        self,
        *,
        base_path: str | Path,
        query_path: str | Path,
        groundtruth_path: str | Path,
        build_exe: str | Path,
        search_exe: str | Path,
        work_dir: str | Path,
        expected_base_count: int | None = 1_000_000,
        expected_query_count: int | None = DEFAULT_GIST_QUERY_COUNT,
        expected_dim: int | None = 960,
        recall_k: int = RECALL_K,
        search_repeat: int = SEARCH_REPEAT,
        optimization_efsearch: Iterable[int] | None = None,
        keep_indexes: bool = False,
        build_timeout_s: int | None = 7200,
        search_timeout_s: int | None = 1800,
        expected_build_sha256: str | None = None,
        expected_search_sha256: str | None = None,
    ):
        self.base_path = Path(base_path).resolve()
        self.query_path = Path(query_path).resolve()
        self.groundtruth_path = Path(groundtruth_path).resolve()
        self.build_exe = Path(build_exe).resolve()
        self.search_exe = Path(search_exe).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._run_lock_handle = None
        self.optimization_efsearch = normalize_efsearch_subset(
            optimization_efsearch
        )
        self.keep_indexes = bool(keep_indexes)
        self.build_timeout_s = build_timeout_s
        self.search_timeout_s = search_timeout_s
        self._counter = 0
        self._lock = threading.Lock()

        if int(recall_k) != RECALL_K:
            raise ValueError(f"HNSW evaluator requires Recall@{RECALL_K}")
        if int(search_repeat) != SEARCH_REPEAT:
            raise ValueError(
                f"HNSW evaluator requires {SEARCH_REPEAT} search repeats"
            )
        for path, label in (
            (self.base_path, "base"),
            (self.query_path, "query"),
            (self.groundtruth_path, "groundtruth"),
            (self.build_exe, "test_index"),
            (self.search_exe, "test_search"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")
        for path, label in (
            (self.build_exe, "test_index"),
            (self.search_exe, "test_search"),
        ):
            if not os.access(path, os.X_OK):
                raise PermissionError(f"{label} is not executable: {path}")
        for path, label, expected_digest in (
            (self.build_exe, "test_index", expected_build_sha256),
            (self.search_exe, "test_search", expected_search_sha256),
        ):
            expected_digest = str(expected_digest or "").strip().lower()
            if expected_digest:
                observed_digest = file_sha256(path)
                if observed_digest.lower() != expected_digest:
                    raise ValueError(
                        f"{label} SHA256 mismatch: {observed_digest} "
                        f"!= configured {expected_digest}"
                    )

        base_count, base_dim = count_vecs_records(self.base_path)
        query_count, query_dim = count_vecs_records(self.query_path)
        gt_count, gt_dim = count_vecs_records(self.groundtruth_path)
        if expected_base_count is not None and base_count != int(expected_base_count):
            raise ValueError(
                f"Physical base count mismatch: {base_count} != {expected_base_count}"
            )
        if expected_query_count is not None and query_count != int(expected_query_count):
            raise ValueError(
                f"Physical query count mismatch: {query_count} != {expected_query_count}"
            )
        if query_count != gt_count:
            raise ValueError(
                f"Physical query/groundtruth count mismatch: {query_count} != {gt_count}"
            )
        if base_dim != query_dim:
            raise ValueError(
                f"Base/query dimension mismatch: {base_dim} != {query_dim}"
            )
        if expected_dim is not None and base_dim != int(expected_dim):
            raise ValueError(
                f"Dataset dimension mismatch: {base_dim} != {expected_dim}"
            )
        if gt_dim < RECALL_K:
            raise ValueError(
                f"Groundtruth width {gt_dim} is smaller than Recall@{RECALL_K}"
            )
        self.base_count = base_count
        self.query_count = query_count
        self.dim = base_dim
        self.gt_dim = gt_dim
        self._acquire_run_lock()

        print(
            f"[HNSW] evaluation | base={self.base_count} | "
            f"queries={self.query_count} | dim={self.dim} | "
            f"Recall@{RECALL_K} | repeats={SEARCH_REPEAT}"
        )
        print(f"[HNSW] test_index:  {self.build_exe}")
        print(f"[HNSW] test_search: {self.search_exe}")
        print(f"[HNSW] binary efSearch: {list(EFSEARCH_GRID)}")
        print(
            "[HNSW] optimization efSearch: "
            f"{list(self.optimization_efsearch)}"
        )

    @staticmethod
    def _run(command: list[str], timeout_s: int | None) -> float:
        assert_no_external_hnsw_process()
        print("  " + " ".join(command), flush=True)
        started = time.perf_counter()
        subprocess.run(
            command,
            check=True,
            env=os.environ.copy(),
            timeout=timeout_s,
        )
        return time.perf_counter() - started

    def _acquire_run_lock(self):
        if os.name != "posix":
            return
        import fcntl

        lock_path = self.work_dir / ".gist_hnsw_eval.lock"
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(
                "Another HNSW tuning run already owns the CPU/QPS "
                "lock. Run target recalls serially."
            ) from exc
        self._run_lock_handle = handle
        atexit.register(self.close)

    def close(self):
        handle = self._run_lock_handle
        if handle is None:
            return
        self._run_lock_handle = None
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()

    def evaluate(
        self,
        M: int,
        ef_construction: int,
        target_recall: float,
        interpolation_efsearch: Iterable[int] | None = None,
    ) -> dict:
        M = int(M)
        ef_construction = int(ef_construction)
        target_recall = float(target_recall)
        selected_grid = normalize_efsearch_subset(
            self.optimization_efsearch
            if interpolation_efsearch is None
            else interpolation_efsearch
        )
        if (
            not 4 <= M <= 100
            or not 20 <= ef_construction <= 800
            or ef_construction < M
        ):
            raise ValueError(
                "Invalid HNSW parameters; require M in [4,100], "
                "efConstruction in [20,800], efConstruction>=M: "
                f"M={M}, efConstruction={ef_construction}"
            )
        if not 0.0 < target_recall < 1.0:
            raise ValueError(f"Invalid target recall: {target_recall}")


        with self._lock:
            self._counter += 1
            token = uuid.uuid4().hex[:8]
            stem = f"eval_{self._counter:05d}_M{M}_efC{ef_construction}_{token}"
            prefix = self.work_dir / stem
            index_path = self.work_dir / f"{stem}.hnsw"
            index_time_path = Path(str(prefix) + "_indexTime.csv")
            performance_path = Path(str(prefix) + "_performance.csv")
            for stale in (index_path, index_time_path, performance_path):
                if stale.exists():
                    stale.unlink()

            build_time_s = 0.0
            search_time_s = 0.0
            try:
                print(
                    f"[HNSW] M={M}, efConstruction={ef_construction}, "
                    f"target={target_recall:.6f}"
                )
                build_time_s = self._run(
                    [
                        str(self.build_exe),
                        str(self.base_path),
                        str(M),
                        str(ef_construction),
                        str(index_path),
                        str(prefix),
                    ],
                    self.build_timeout_s,
                )
                if not index_path.is_file() or index_path.stat().st_size == 0:
                    raise RuntimeError(
                        "test_index returned success but did not create a non-empty "
                        f"index: {index_path}"
                    )
                index_size_bytes = index_path.stat().st_size

                search_time_s = self._run(
                    [
                        str(self.search_exe),
                        str(self.base_path),
                        str(index_path),
                        str(self.query_path),
                        str(self.groundtruth_path),
                        str(prefix),
                    ],
                    self.search_timeout_s,
                )
                if not performance_path.is_file() or performance_path.stat().st_size == 0:
                    raise RuntimeError(
                        f"test_search did not create a non-empty curve: {performance_path}"
                    )

                ef_search, recalls, qps, selected_rows, full_rows = (
                    read_performance_curve(performance_path, selected_grid)
                )
                target = interpolate_qps(target_recall, ef_search, recalls, qps)
                interpolation_targets = sorted(
                    {*SUPPORTED_TARGETS, target_recall}
                )
                interp = {
                    value: (
                        float(np.interp(value, recalls, qps))
                        if value <= target["max_recall"] + 1e-12 else 0.0
                    )
                    for value in interpolation_targets
                }

                full_ef, full_recalls, full_qps, _, _ = read_performance_curve(
                    performance_path, EFSEARCH_GRID
                )
                full_target = interpolate_qps(
                    target_recall, full_ef, full_recalls, full_qps
                )
                ignored = [row for row in full_rows if row[0] not in selected_grid]
                print(
                    f"[HNSW] optimization QPS@R={target_recall:.6f}="
                    f"{target['qps_at_target']:.6g}, "
                    f"selected_max_recall={target['max_recall']:.6f}, "
                    f"full_max_recall={full_target['max_recall']:.6f}, "
                    f"queries={self.query_count}"
                )
                return {
                    **target,
                    "interp": interp,
                    "raw_results": selected_rows,
                    "raw_results_full": full_rows,
                    "ignored_results": ignored,
                    "full_qps_at_target": float(full_target["qps_at_target"]),
                    "full_target_reached": bool(full_target["target_reached"]),
                    "full_max_recall": float(full_target["max_recall"]),
                    "interpolation_efsearch": list(selected_grid),
                    "binary_efsearch": list(EFSEARCH_GRID),
                    "build_time_s": build_time_s,
                    "internal_build_time_s": _read_internal_build_time(
                        index_time_path
                    ),
                    "search_time_s": search_time_s,
                    "total_eval_time_s": build_time_s + search_time_s,
                    "index_size_bytes": index_size_bytes,
                    "index_time_curve": str(index_time_path),
                    "performance_curve": str(performance_path),
                }
            finally:
                if index_path.exists() and not self.keep_indexes:
                    index_path.unlink()
