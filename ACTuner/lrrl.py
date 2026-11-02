"""PPO, LLM access, and GIST1M evaluation primitives."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import time
import os
import json
import requests
import re
import random
import yaml
import struct
from dataclasses import dataclass
from pathlib import Path

from hnsw_backend import (
    DEFAULT_GIST_QUERY_COUNT,
    EFSEARCH_GRID,
    RECALL_K,
    SEARCH_REPEAT,
    HNSWEvaluator,
    normalize_efsearch_subset,
)

REQUIRED_QUERY_COUNT = DEFAULT_GIST_QUERY_COUNT
REQUIRED_RECALL_K = RECALL_K
REQUIRED_SEARCH_REPEAT = SEARCH_REPEAT
QPS_STATE_SCALE = 2000.0
RECALL_COMPARISON_EPS = 1e-12


def validate_target_recall(value):
    if isinstance(value, bool):
        raise ValueError("R_target must be a decimal number in (0, 1], not a boolean")
    try:
        target = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"R_target must be a decimal number in (0, 1], got {value!r}"
        ) from exc
    if not np.isfinite(target) or not 0.0 < target <= 1.0:
        raise ValueError(
            f"R_target must be a decimal number in (0, 1], got {value!r}. "
            "Use 0.96 for the target and reserve 096 only for run names."
        )
    return target


def action_boundary_penalty(action, boundary_beta):
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.size != 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"action must contain two finite values, got {action!r}")
    beta = float(boundary_beta)
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError(f"boundary_beta must be finite and non-negative, got {boundary_beta!r}")
    return float(beta * np.maximum(np.abs(values) - 1.0, 0.0).sum())


def recall_target_reached(recall, target_recall):
    recall = float(recall)
    target = validate_target_recall(target_recall)
    return bool(np.isfinite(recall) and recall + RECALL_COMPARISON_EPS >= target)


def should_update_phase2_best(
        global_eval_index, n_warmup, recall, target_recall, qps, best_qps):
    return (
        recall_target_reached(recall, target_recall)
        and float(qps) > float(best_qps)
    )


def configure_qps_state_scale(value):
    global QPS_STATE_SCALE
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"qps_state_scale must be positive and finite, got {value!r}")
    QPS_STATE_SCALE = scale
    print(f"[Runtime] QPS state scale: {QPS_STATE_SCALE:g}")
    return QPS_STATE_SCALE


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print(f"Device set to: {torch.cuda.get_device_name(device)}")
else:
    print("Device set to: cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
    np.random.seed(seed)
    random.seed(seed)


PROJECT_DIR = Path(__file__).resolve().parent


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def available_cpu_count():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def configure_openmp(runtime_cfg):
    value = os.environ.get("OMP_NUM_THREADS") or runtime_cfg.get("hnswlib_threads", "max")
    available = available_cpu_count()
    count = available if str(value).lower() == "max" else int(value)
    if not 1 <= count <= available:
        raise ValueError(f"OpenMP thread count must be in [1, {available}], got {count}")
    os.environ["OMP_NUM_THREADS"] = str(count)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    return count


CURRENT_DATA_PATHS = {}
CURRENT_CONFIG = {}


def configure_hnsw_threads(runtime_cfg):
    count = configure_openmp(runtime_cfg)
    print(f"[Runtime] hnswlib/OpenMP threads: {count}")
    return str(count)


def load_project_config(config_path=None):
    config_path = config_path or os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    global CURRENT_CONFIG
    CURRENT_CONFIG = cfg
    return cfg


@dataclass(frozen=True)
class VecsMetadata:

    path: str
    shape: tuple[int, int]

    def __len__(self):
        return self.shape[0]


def _read_vecs_metadata(path, expected_dim=None, expected_count=None):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(4)
    if len(header) != 4:
        raise ValueError(f"Dataset file is empty or truncated: {path}")
    dim = struct.unpack("<i", header)[0]
    if dim <= 0:
        raise ValueError(f"Invalid vecs dimension {dim}: {path}")
    record_size = 4 * (dim + 1)
    if size % record_size:
        raise ValueError(f"Invalid vecs file size for dimension {dim}: {path}")
    count = size // record_size
    if expected_dim is not None and dim != int(expected_dim):
        raise ValueError(f"Unexpected dimension in {path}: {dim} != {expected_dim}")
    if expected_count is not None and count != int(expected_count):
        raise ValueError(f"Unexpected row count in {path}: {count} != {expected_count}")
    return VecsMetadata(str(path), (int(count), int(dim)))


def load_gist1m(config_path=None):
    """Validate GIST1M file metadata without loading vector payloads."""
    cfg = load_project_config(config_path)
    data_cfg = cfg.get("data", {})
    config_file = Path(
        config_path or Path(__file__).resolve().parent / "config.yaml"
    ).resolve()

    def resolve_data_path(value):
        path = Path(str(value)).expanduser()
        return str(path if path.is_absolute() else config_file.parent / path)

    base_path = resolve_data_path(os.environ.get(
        "GIST_BASE_PATH", data_cfg.get("base_path", "")
    ))
    query_path = resolve_data_path(os.environ.get(
        "GIST_QUERY_PATH", data_cfg.get("query_path", "")
    ))
    gt_path = resolve_data_path(os.environ.get(
        "GIST_GROUNDTRUTH_PATH", data_cfg.get("groundtruth_path", "")
    ))
    base_count = int(data_cfg.get("base_limit", 1_000_000))
    query_count = int(data_cfg.get("query_limit", DEFAULT_GIST_QUERY_COUNT))
    expected_dim = int(data_cfg.get("expected_dim", 960))
    expected_gt_dim = int(data_cfg.get("expected_gt_dim", 100))

    global CURRENT_DATA_PATHS
    CURRENT_DATA_PATHS = {
        "base_path": base_path,
        "query_path": query_path,
        "groundtruth_path": gt_path,
        "base_limit": base_count,
        "query_limit": query_count,
        "expected_dim": expected_dim,
        "expected_gt_dim": expected_gt_dim,
    }
    base = _read_vecs_metadata(base_path, expected_dim, base_count)
    query = _read_vecs_metadata(query_path, expected_dim, query_count)
    groundtruth = _read_vecs_metadata(gt_path, expected_gt_dim, query_count)
    if len(query) != len(groundtruth):
        raise ValueError(
            f"Query/groundtruth count mismatch: {len(query)} != {len(groundtruth)}"
        )
    print(
        f"[Data metadata-only] base={base.shape} query={query.shape} "
        f"gt={groundtruth.shape}; vector payloads are loaded by the C++ evaluators"
    )
    return base, query, groundtruth


class LLMKnowledgeProcessor:
    def __init__(
        self,
        api_key=None,
        llm_type="doubao",
        base_url=None,
        model=None,
        connect_timeout_s=10,
        request_timeout_s=120,
        retries=3,
        retry_backoff_s=10,
        skip_llm=False,
    ):
        self.api_key = api_key or os.environ.get("DOUBAO_API_KEY", "")
        self.llm_type = llm_type
        self.base_url = base_url or os.environ.get(
            "DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
        self.model = model or os.environ.get("DOUBAO_MODEL", "doubao-seed-2-0-pro-260215")
        self.connect_timeout_s = max(1.0, float(connect_timeout_s))
        self.request_timeout_s = max(1.0, float(request_timeout_s))
        self.retries = max(1, int(retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.skip_llm = bool(skip_llm) or os.environ.get(
            "SKIP_LLM", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.circuit_open = False
        self.circuit_reason = ""
        self.api_request_attempts = 0
        self.api_successful_calls = 0
        self.session = requests.Session()
        self.usage_records = []

    @staticmethod
    def _usage_int(usage, *keys):
        for key in keys:
            val = usage.get(key)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    return 0
        return 0

    def _record_usage(self, usage, prompt_chars, max_tokens):
        usage = usage or {}
        record = {
            "call": len(self.usage_records) + 1,
            "prompt_chars": int(prompt_chars),
            "max_tokens": int(max_tokens),
            "prompt_tokens": self._usage_int(usage, "prompt_tokens", "input_tokens"),
            "completion_tokens": self._usage_int(usage, "completion_tokens", "output_tokens"),
            "total_tokens": self._usage_int(usage, "total_tokens"),
            "has_usage": bool(usage),
            "raw_usage": usage,
        }
        if record["total_tokens"] == 0:
            record["total_tokens"] = record["prompt_tokens"] + record["completion_tokens"]
        self.usage_records.append(record)
        if record["has_usage"]:
            print(
                "[LLM Usage] "
                f"prompt_tokens={record['prompt_tokens']}, "
                f"completion_tokens={record['completion_tokens']}, "
                f"total_tokens={record['total_tokens']}"
            )
        else:
            print("[LLM Usage] API response did not include usage; exact token count unavailable")

    def get_usage_summary(self):
        records = list(self.usage_records)
        return {
            "records": records,
            "api_request_attempts": int(self.api_request_attempts),
            "api_successful_calls": int(self.api_successful_calls),
            "circuit_open": bool(self.circuit_open),
            "circuit_reason": self.circuit_reason,
            "calls_with_usage": sum(1 for r in records if r.get("has_usage")),
            "prompt_chars": sum(int(r.get("prompt_chars", 0)) for r in records),
            "max_tokens_requested": sum(int(r.get("max_tokens", 0)) for r in records),
            "prompt_tokens": sum(int(r.get("prompt_tokens", 0)) for r in records),
            "completion_tokens": sum(int(r.get("completion_tokens", 0)) for r in records),
            "total_tokens": sum(int(r.get("total_tokens", 0)) for r in records),
        }

    def _call_api(self, prompt, timeout=None, max_tokens=2048, retries=None):
        if self.skip_llm:
            print("[LLM] SKIP_LLM enabled; using deterministic fallback knowledge")
            return None
        if self.circuit_open:
            print(
                "[LLM] Circuit open after an earlier failure; using fallback "
                f"({self.circuit_reason})"
            )
            return None
        timeout = self.request_timeout_s if timeout is None else min(
            float(timeout), self.request_timeout_s
        )
        retries = self.retries if retries is None else min(
            max(1, int(retries)), self.retries
        )
        if self.llm_type == "doubao":
            if not self.api_key:
                self.circuit_open = True
                self.circuit_reason = "DOUBAO_API_KEY is not set"
                print("[LLM] DOUBAO_API_KEY is not set; using fallback knowledge")
                return None
            url = self.base_url
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Connection": "close",
            }
            data = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")
        retryable_status = {429, 500, 502, 503, 504}
        for attempt in range(1, retries + 1):
            try:
                self.api_request_attempts += 1
                print(f"[LLM] Querying {self.llm_type} API... "
                      f"attempt={attempt}/{retries}, prompt_chars={len(prompt)}, "
                      f"max_tokens={max_tokens}, timeout={timeout}")
                response = self.session.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=(self.connect_timeout_s, timeout),
                )
                if response.status_code != 200:
                    print(f"[LLM] API error: {response.status_code} - {response.text[:200]}")
                    if response.status_code in retryable_status and attempt < retries:
                        wait_s = min(60.0, self.retry_backoff_s * attempt)
                        print(f"[LLM] Retrying after {wait_s}s...")
                        time.sleep(wait_s)
                        self.session = requests.Session()
                        continue
                    self.circuit_open = True
                    self.circuit_reason = f"HTTP {response.status_code}"
                    return None
                payload = response.json()
                self._record_usage(payload.get("usage"), len(prompt), max_tokens)
                self.api_successful_calls += 1
                return payload["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"[LLM] Exception on attempt {attempt}/{retries}: {e}")
                self.session = requests.Session()
                if attempt < retries:
                    wait_s = min(60.0, self.retry_backoff_s * attempt)
                    print(f"[LLM] Retrying after {wait_s}s...")
                    time.sleep(wait_s)
                    continue
                self.circuit_open = True
                self.circuit_reason = f"{type(e).__name__}: {e}"
                print("[LLM] Retry budget exhausted; circuit opened for this run")
                return None

    @staticmethod
    def _normalize_regions(regions):
        if not isinstance(regions, list) or len(regions) == 0:
            raise ValueError("Missing regions array")
        normalized = []
        required = ("mu_M", "delta_M", "mu_ef", "delta_ef")
        for index, region in enumerate(regions[:3], start=1):
            if not isinstance(region, dict):
                raise ValueError(f"Region {index} must be a JSON object")
            missing = [key for key in required if key not in region]
            if missing:
                raise ValueError(f"Region {index} is missing fields: {missing}")
            try:
                mu_M = int(round(float(region["mu_M"])))
                delta_M = int(round(float(region["delta_M"])))
                mu_ef = int(round(float(region["mu_ef"])))
                delta_ef = int(round(float(region["delta_ef"])))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Region {index} has non-numeric bounds") from exc

            mu_M = int(np.clip(mu_M, 4, 100))
            mu_ef = int(np.clip(max(mu_ef, mu_M), 20, 800))
            delta_M = int(np.clip(delta_M, 1, 96))
            delta_ef = int(np.clip(delta_ef, 1, 780))
            item = {
                "mu_M": mu_M,
                "delta_M": delta_M,
                "mu_ef": mu_ef,
                "delta_ef": delta_ef,
                "reasoning": str(region.get("reasoning", "")),
            }
            if "weight" in region:
                try:
                    weight = float(region["weight"])
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"Region {index} has a non-numeric weight") from exc
                if not np.isfinite(weight) or weight < 0.0:
                    raise ValueError(f"Region {index} weight must be finite and non-negative")
                item["weight"] = weight
            normalized.append(item)
        if any("weight" in region for region in normalized):
            weights = [region.get("weight", 1.0) for region in normalized]
            total = sum(weights)
            if total <= 0.0:
                raise ValueError("At least one LLM region weight must be positive")
            for region, weight in zip(normalized, weights):
                region["weight"] = float(weight / total)
        return normalized

    @classmethod
    def _parse_regions(cls, raw):
        clean = re.sub(r'```json|```', '', raw).strip()
        data = json.loads(clean)
        if not isinstance(data, dict):
            raise ValueError("LLM response must be a JSON object")
        return cls._normalize_regions(data.get('regions', []))

    def get_initial_prior(self, workload="gist1m", target_recall=0.99):
        prompt = f"""You are an expert in vector database optimization (hnswlib HNSW).
Task: Recommend 2-3 high-potential search regions for HNSW parameter tuning on GIST1M.

[Workload]
{workload}  (1M vectors, 960-dim, L2 distance)

[Parameter bounds]
- M (graph degree): [4, 100]
- efConstruction (build quality): [20, 800]

[Goal]
Maximize QPS while achieving recall@10 >= {target_recall}

[Output — strict JSON only]
{{"regions": [
  {{"weight": 0.5, "mu_M": 32, "delta_M": 12, "mu_ef": 300, "delta_ef": 100}},
  {{"weight": 0.3, "mu_M": 48, "delta_M": 16, "mu_ef": 500, "delta_ef": 150}},
  {{"weight": 0.2, "mu_M": 64, "delta_M": 14, "mu_ef": 200, "delta_ef": 80}}
]}}
Weights must sum to 1.0."""

        default_regions = [
            {"weight": 0.4, "mu_M": 32, "delta_M": 12, "mu_ef": 300, "delta_ef": 100},
            {"weight": 0.35, "mu_M": 48, "delta_M": 14, "mu_ef": 500, "delta_ef": 120},
            {"weight": 0.25, "mu_M": 64, "delta_M": 16, "mu_ef": 200, "delta_ef": 80},
        ]
        try:
            resp = self._call_api(prompt)
            if not resp:
                raise ValueError("Empty response")
            regions = self._parse_regions(resp)
            print(f"[LLM] Initial prior: {len(regions)} regions")
            return regions
        except Exception as e:
            print(f"[LLM] Initial prior failed ({e}), using defaults")
            return default_regions

    def get_reflection_prior(self, history_records, best_result, workload="gist1m", target_recall=0.99):

        history_lines = []
        recent = history_records[-20:]
        for r in recent:
            history_lines.append(
                f"  M={r['M']:3d} efC={r['efC']:3d} | "
                f"recall={r['recall']:.4f} QPS@R={target_recall}={r.get('qps_at_target', 0):.0f} | "
                f"reward={r['reward']:.3f} | src={r.get('source', '?')}"
            )
        history_str = "\n".join(history_lines)


        if best_result:
            best_str = (f"Best so far: M={best_result['M']}, efConstruction={best_result['efC']}, "
                        f"recall@10={best_result['recall']:.4f}, "
                        f"QPS@R={target_recall}={best_result['qps_at_target']:.0f}")
        else:
            best_str = "No successful result yet."

        prompt = f"""You are an expert in hnswlib HNSW parameter optimization.
We are tuning (M, efConstruction) for the GIST1M dataset (1M x 960-dim, L2).
Goal: recall@10 >= {target_recall} with MAXIMUM QPS.

[Current Best]
{best_str}

[Recent Search History]
{history_str}

[Parameter Bounds]
M ∈ [4, 100], efConstruction ∈ [20, 800]

[Your Task]
Based on the history above, recommend 2-3 NEW search regions likely to find
BETTER QPS at recall>={target_recall}. Consider:
- High M improves recall but reduces QPS (more distance computations)
- High efConstruction improves graph quality but increases build time
- The sweet spot balances graph connectivity against search speed
- If current best has M~50, try refining around that region
- If recall is already 1.0, try REDUCING M to gain QPS while keeping recall>={target_recall}

[Output — strict JSON only]
{{"regions": [
  {{"weight": <float>, "mu_M": <int>, "delta_M": <int>, "mu_ef": <int>, "delta_ef": <int>}},
  ...
]}}
Weights must sum to 1.0. Give 2-3 regions."""

        default_regions = [
            {"weight": 0.45, "mu_M": 40, "delta_M": 12, "mu_ef": 400, "delta_ef": 120},
            {"weight": 0.35, "mu_M": 56, "delta_M": 14, "mu_ef": 550, "delta_ef": 130},
            {"weight": 0.20, "mu_M": 24, "delta_M": 8, "mu_ef": 250, "delta_ef": 80},
        ]
        try:
            resp = self._call_api(prompt)
            if not resp:
                raise ValueError("Empty response")
            regions = self._parse_regions(resp)
            print(f"[LLM] Reflection prior: {len(regions)} regions (based on {len(recent)} history records)")
            return regions
        except Exception as e:
            print(f"[LLM] Reflection failed ({e}), using defaults")
            return default_regions


class HNSWEnv:

    """Evaluate HNSW configurations at a fixed recall target."""
    def __init__(
        self,
        base_vectors,
        query_vectors,
        groundtruth,
        R_target=0.99,
        efSearch_train=None,
        efSearch_full=None,
        n_query_sample=DEFAULT_GIST_QUERY_COUNT,
        search_repeat=SEARCH_REPEAT,
        recall_k=RECALL_K,
    ):
        self.base = base_vectors
        self.query = query_vectors
        self.gt = groundtruth
        self.R_target = validate_target_recall(R_target)
        self.recall_k = int(recall_k)
        self.dim = int(base_vectors.shape[1])
        self.n_query_sample = int(n_query_sample)
        self.search_repeat = int(search_repeat)
        self.project_dir = Path(__file__).resolve().parent
        self.runtime_cfg = (CURRENT_CONFIG or load_project_config()).get(
            "runtime", {}
        )
        self.data_paths = dict(CURRENT_DATA_PATHS)
        self.efSearch_train = list(normalize_efsearch_subset(efSearch_train))
        self.efSearch_full = list(normalize_efsearch_subset(efSearch_full))

        if len(query_vectors) != REQUIRED_QUERY_COUNT:
            raise ValueError(
                f"GIST1M evaluation requires exactly {REQUIRED_QUERY_COUNT} "
                f"physical queries; got {len(query_vectors)}"
            )
        if len(groundtruth) != REQUIRED_QUERY_COUNT:
            raise ValueError(
                f"GIST1M evaluation requires exactly {REQUIRED_QUERY_COUNT} "
                f"groundtruth rows; got {len(groundtruth)}"
            )
        if self.n_query_sample != REQUIRED_QUERY_COUNT:
            raise ValueError(
                f"train_query_sample must equal {REQUIRED_QUERY_COUNT}; "
                f"got {self.n_query_sample}"
            )
        if self.recall_k != RECALL_K:
            raise ValueError(f"test_search uses Recall@{RECALL_K}")
        if self.search_repeat != SEARCH_REPEAT:
            raise ValueError(
                f"test_search uses {SEARCH_REPEAT} repeats"
            )
        if tuple(self.efSearch_train) != EFSEARCH_GRID:
            raise ValueError(
                "GIST optimization must use the complete evaluation curve: "
                f"{list(EFSEARCH_GRID)}"
            )
        if tuple(self.efSearch_full) != EFSEARCH_GRID:
            raise ValueError(
                "GIST final evaluation must use the complete evaluation curve: "
                f"{list(EFSEARCH_GRID)}"
            )
        missing = [
            key for key in ("base_path", "query_path", "groundtruth_path")
            if not self.data_paths.get(key)
        ]
        if missing:
            raise RuntimeError(f"Missing evaluator data paths: {missing}")

        def project_path(value, default):
            path = Path(value or default).expanduser()
            return path if path.is_absolute() else self.project_dir / path

        self.backend = HNSWEvaluator(
            base_path=self.data_paths["base_path"],
            query_path=self.data_paths["query_path"],
            groundtruth_path=self.data_paths["groundtruth_path"],
            build_exe=project_path(
                self.runtime_cfg.get("hnsw_build_exe"), "tests/test_index"
            ),
            search_exe=project_path(
                self.runtime_cfg.get("hnsw_search_exe"), "tests/test_search"
            ),
            work_dir=project_path(
                self.runtime_cfg.get("hnsw_work_dir"),
                "runs/backend",
            ),
            expected_base_count=int(
                self.data_paths.get("base_limit", 1_000_000)
            ),
            expected_query_count=REQUIRED_QUERY_COUNT,
            expected_dim=self.dim,
            recall_k=self.recall_k,
            search_repeat=self.search_repeat,
            optimization_efsearch=self.efSearch_train,
            keep_indexes=bool(
                self.runtime_cfg.get("keep_hnsw_indexes", False)
            ),
            build_timeout_s=int(
                self.runtime_cfg.get("hnsw_build_timeout_s", 7200)
            ),
            search_timeout_s=int(
                self.runtime_cfg.get("hnsw_search_timeout_s", 1800)
            ),
            expected_build_sha256=self.runtime_cfg.get(
                "hnsw_build_exe_sha256"
            ),
            expected_search_sha256=self.runtime_cfg.get(
                "hnsw_search_exe_sha256"
            ),
        )
        self.last_evaluation = None
        self.evaluation_cache = {}

    @staticmethod
    def _stable_curve(raw_results):
        ordered = sorted(
            enumerate(raw_results), key=lambda item: (float(item[1][1]), item[0])
        )
        unique = []
        seen = set()
        for _, row in ordered:
            recall = float(row[1])
            if recall in seen:
                continue
            seen.add(recall)
            unique.append((int(row[0]), recall, float(row[2])))
        if not unique:
            raise ValueError("Cannot interpolate an empty search curve")
        return unique

    def interpolate_results(self, raw_results, efsearch=None):
        grid = (
            getattr(self, "efSearch_train", EFSEARCH_GRID)
            if efsearch is None else efsearch
        )
        selected = set(normalize_efsearch_subset(grid))
        rows = [row for row in raw_results if int(row[0]) in selected]
        curve = self._stable_curve(rows)
        recalls = np.asarray([row[1] for row in curve], dtype=float)
        qps = np.asarray([row[2] for row in curve], dtype=float)
        max_recall = float(recalls[-1])
        targets = sorted(
            {0.90, 0.92, 0.94, 0.96, 0.98, 0.99, self.R_target}
        )
        interp = {
            target: (
                float(np.interp(target, recalls, qps))
                if target <= max_recall + RECALL_COMPARISON_EPS else 0.0
            )
            for target in targets
        }
        return interp, float(interp[self.R_target]), max_recall

    def build_and_evaluate(self, M, efConstruction):
        result = self.backend.evaluate(
            int(M),
            int(efConstruction),
            self.R_target,
            interpolation_efsearch=self.efSearch_train,
        )
        key = (int(M), int(efConstruction))
        self.last_evaluation = result
        self.evaluation_cache[key] = result
        return (
            bool(result["target_reached"]),
            float(result["max_recall"]),
            float(result["qps_at_target"]),
            None,
            dict(result["interp"]),
            list(result["raw_results"]),
        )

    def get_evaluation(self, M, efConstruction):
        return self.evaluation_cache.get((int(M), int(efConstruction)))

    def full_evaluate(self, M, efConstruction, force_remeasure=False):
        key = (int(M), int(efConstruction))
        result = None if force_remeasure else self.evaluation_cache.get(key)
        if result is None:
            result = self.backend.evaluate(
                key[0],
                key[1],
                self.R_target,
                interpolation_efsearch=self.efSearch_full,
            )
            self.last_evaluation = result
            self.evaluation_cache[key] = result
        rows = list(result["raw_results_full"])
        for efS, rec, qps in rows:
            print(f"    efSearch={efS:4d}  recall={rec:.6f}  QPS={qps:.1f}")
        return rows


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.is_terminals.clear()


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super().__init__()
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def act(self, state):
        with torch.no_grad():
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var)
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return dist_entropy, action_logprobs, state_values


class PPO:
    """Optimize residual actions with proximal policy optimization."""
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic,
                 gamma, K_epochs, eps_clip, action_std_init=0.6):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.action_std = action_std_init

        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.mseloss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(self.action_std)
        self.policy_old.set_action_std(self.action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
        self.set_action_std(self.action_std)

    def select_action(self, state):
        state_t = torch.FloatTensor(state).to(device)
        action, action_logprob, state_val = self.policy_old.act(state_t)
        self.buffer.states.append(state_t)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        return action.cpu().numpy()

    def update(self):
        if not self.buffer.rewards:
            return
        n_items = len(self.buffer.rewards)
        lengths = {
            "states": len(self.buffer.states),
            "actions": len(self.buffer.actions),
            "logprobs": len(self.buffer.logprobs),
            "state_values": len(self.buffer.state_values),
            "terminals": len(self.buffer.is_terminals),
        }
        if any(length != n_items for length in lengths.values()):
            raise RuntimeError(
                f"PPO rollout buffer is inconsistent: rewards={n_items}, {lengths}"
            )
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards),
                                        reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        old_states = torch.stack(self.buffer.states, dim=0).detach().to(device)
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).reshape(-1).detach().to(device)

        for _ in range(self.K_epochs):
            dist_entropy, logprobs, state_values = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.squeeze(-1)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-7
            )

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.mseloss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda s, l: s))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda s, l: s))


def compute_reward(recall, qps, R_target, lambda_pen, kappa, boundary_beta, action):
    """Combine log-QPS, recall shortfall, and action-boundary penalties."""
    target = validate_target_recall(R_target)
    recall = float(recall)
    qps = float(qps)
    lambda_pen = float(lambda_pen)
    kappa = float(kappa)
    if not np.isfinite(recall) or not 0.0 <= recall <= 1.0:
        raise ValueError(f"recall must be finite and in [0, 1], got {recall!r}")
    if not np.isfinite(qps) or qps < 0.0:
        raise ValueError(f"qps must be finite and non-negative, got {qps!r}")
    if not np.isfinite(lambda_pen) or lambda_pen < 0.0:
        raise ValueError(f"lambda_penalty must be finite and non-negative, got {lambda_pen!r}")
    if not np.isfinite(kappa) or kappa < 0.0:
        raise ValueError(f"kappa must be finite and non-negative, got {kappa!r}")
    penalty_bound = action_boundary_penalty(action, boundary_beta)
    barrier = lambda_pen * np.logaddexp(0.0, kappa * (target - recall))
    reward = np.log(max(qps, 1e-5)) - barrier
    return float(reward - penalty_bound)


def build_state(prior_mu_M, prior_delta_M, prior_mu_ef, prior_delta_ef,
                c_t, best_recall, best_qps, epoch_progress):
    return np.array([
        prior_mu_M / 100.0,
        prior_delta_M / 100.0,
        prior_mu_ef / 800.0,
        prior_delta_ef / 800.0,
        c_t,
        best_recall,
        min(max(float(best_qps), 0.0) / QPS_STATE_SCALE, 1.0),
        epoch_progress,
    ], dtype=np.float32)


def train():
    raise RuntimeError(
        "lrrl.py is a core module, not the experiment entry point. "
        "Run `bash run_server.sh --target-recall 0.90` or "
        "`python tuner.py --target-recall 0.90` so the 30-evaluation "
        "budget, deduplication, timing, and result logging are enforced."
    )


if __name__ == '__main__':
    result = train()
