"""ACTuner search loop for HNSW construction parameters."""

import sys
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import json
import csv
import yaml
import re
import math
import hashlib
from string import Template

from lrrl import (
    LLMKnowledgeProcessor, HNSWEnv, load_gist1m,
    PPO, compute_reward, build_state, set_seed, device,
    configure_hnsw_threads, validate_target_recall, action_boundary_penalty,
    recall_target_reached, should_update_phase2_best,
    configure_qps_state_scale, project_path
)
from result_logger import save_param_result, save_aggregated_results
import result_logger

SUPPORTED_TARGET_RECALLS = (0.90, 0.92, 0.94, 0.96, 0.98, 0.99)


WARMUP_GRID_CANDIDATES = (
    (10, 150), (10, 500), (10, 700),
    (50, 150), (50, 500), (50, 700),
    (80, 150), (80, 500), (80, 700),
    (100,150),
)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_warmup_candidates(target_recall):
    target = round(float(target_recall), 2)
    if target not in SUPPORTED_TARGET_RECALLS:
        raise ValueError(
            f"Unsupported target recall {target_recall!r}; "
            f"supported={SUPPORTED_TARGET_RECALLS}"
        )
    return list(WARMUP_GRID_CANDIDATES)


class CaseBase:

    """Store and retrieve prior tuning results."""
    def __init__(self, cache_path=None):
        self.cases = []
        self.cache_path = cache_path or os.path.join(
            os.path.dirname(__file__), "state", "case_base.json")
        self._load()

    def _load(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                self.cases = json.load(f)
            print(f"[CaseBase] Loaded {len(self.cases)} prior cases from {self.cache_path}")
        else:
            print(f"[CaseBase] No prior cases found (cold start)")

    def _save(self):
        os.makedirs(os.path.dirname(self.cache_path) or '.', exist_ok=True)
        temp_path = f"{self.cache_path}.tmp.{os.getpid()}"
        try:
            with open(temp_path, 'w') as f:
                json.dump(self.cases, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.cache_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        print(f"[CaseBase] Saved {len(self.cases)} cases to {self.cache_path}")

    def add_case(self, dataset_name, N, dim, target_recall, best_M, best_efC, best_qps):
        case = {
            "dataset": dataset_name,
            "feature": {"N": int(N), "dim": int(dim), "target_recall": float(target_recall)},
            "best_region": {"M": int(best_M), "efC": int(best_efC)},
            "best_qps": float(best_qps)
        }
        for c in self.cases:
            if (c["dataset"] == dataset_name
                    and math.isclose(
                        float(c["feature"]["target_recall"]), float(target_recall),
                        rel_tol=0.0, abs_tol=1e-12)):
                c.update(case)
                self._save()
                return
        self.cases.append(case)
        self._save()

    @property
    def size(self):
        return len(self.cases)

    def encode(self, N, dim, target_recall):
        return np.array([
            np.log10(max(N, 1)) / 10.0,
            float(dim) / 1000.0,
            float(target_recall),
        ])

    def retrieve(self, N, dim, target_recall, top_k=3):
        if self.size == 0:
            return []
        x_query = self.encode(N, dim, target_recall)
        scored = []
        for case in self.cases:
            x_case = self.encode(
                case["feature"]["N"], case["feature"]["dim"],
                case["feature"]["target_recall"])


            delta = x_query - x_case
            delta[2] *= 20.0
            sim = float(np.exp(-np.linalg.norm(delta)))
            scored.append((sim, case))
        scored.sort(key=lambda s: s[0], reverse=True)
        return scored[:top_k]

    def empirical_prior(self, N, dim, target_recall, top_k=3):
        retrieved = self.retrieve(N, dim, target_recall, top_k)
        if not retrieved:
            return None, None
        M_vals = np.asarray(
            [c["best_region"]["M"] for _, c in retrieved], dtype=float
        )
        efC_vals = np.asarray(
            [c["best_region"]["efC"] for _, c in retrieved], dtype=float
        )
        sims = np.asarray([max(float(s), 1e-8) for s, _ in retrieved])
        weights = sims / sims.sum()
        mu_M = float(np.sum(weights * M_vals))
        mu_ef = float(np.sum(weights * efC_vals))
        return {
            "mu_M": mu_M,
            "sigma_M": float(
                np.sqrt(np.sum(weights * (M_vals - mu_M) ** 2))
            ) + 1e-6,
            "mu_ef": mu_ef,
            "sigma_ef": float(
                np.sqrt(np.sum(weights * (efC_vals - mu_ef) ** 2))
            ) + 1e-6,
            "M_vals": [int(m) for m in M_vals],
            "efC_vals": [int(e) for e in efC_vals],
            "top_similarity": float(sims[0]) if len(sims) else 0.0,
        }, retrieved

    def weight_evidence(self, tau=3.0):
        return 1.0 - np.exp(-self.size / tau)


def infer_dataset_name(data_cfg, dim):
    text = " ".join(str(data_cfg.get(k, "")) for k in (
        "name", "base_path", "query_path", "groundtruth_path"
    )).lower()
    if "sift" in text:
        return "sift1m"
    if "gist" in text:
        return "gist1m"
    if "glove" in text:
        return "glove"
    if "deep" in text:
        return "deep"
    return f"dim{int(dim)}"


class SurrogateMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class SurrogateEnsemble:
    """Estimate recall, QPS, and prediction uncertainty."""
    def __init__(self, n_members=5, hidden_dim=64, device='cpu'):
        self.n_members = n_members
        self.device = device
        self.members = [SurrogateMLP(2, hidden_dim, 2).to(device) for _ in range(n_members)]
        self.optimizers = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in self.members]

        self.M_min, self.M_max = 4, 100
        self.efC_min, self.efC_max = 20, 800
        self.qps_min, self.qps_max = 1.0, 1000000.0
        self.log_qps_min = float(np.log(self.qps_min))
        self.log_qps_max = float(np.log(self.qps_max))

        self.X_train = []
        self.Y_train = []

    def _normalize_input(self, M, efC):
        if np.isscalar(M):
            return np.array([
                (M - self.M_min) / (self.M_max - self.M_min),
                (efC - self.efC_min) / (self.efC_max - self.efC_min)
            ], dtype=np.float32)
        M_norm = (np.array(M, dtype=np.float64) - self.M_min) / (self.M_max - self.M_min)
        efC_norm = (np.array(efC, dtype=np.float64) - self.efC_min) / (self.efC_max - self.efC_min)
        return np.stack([M_norm, efC_norm], axis=-1).astype(np.float32)

    def _normalize_target(self, recall, qps):
        recall = np.clip(float(recall), 0.0, 1.0)
        qps_clipped = np.clip(float(qps), self.qps_min, self.qps_max)
        log_qps = np.log(qps_clipped)
        log_qps_norm = (log_qps - self.log_qps_min) / (self.log_qps_max - self.log_qps_min)
        return np.array([recall, log_qps_norm], dtype=np.float32)

    def _denormalize_output(self, y):
        recall = np.clip(y[..., 0], 0.0, 1.0)
        log_qps_norm = np.clip(y[..., 1], 0.0, 1.0)
        log_qps = log_qps_norm * (self.log_qps_max - self.log_qps_min) + self.log_qps_min
        qps = np.exp(log_qps)
        return recall, qps

    def add_data(self, M, efC, recall, qps):
        self.X_train.append(self._normalize_input(M, efC))
        self.Y_train.append(self._normalize_target(recall, qps))

    def train(self, epochs=200, batch_size=64, verbose=False):
        if len(self.X_train) < 4:
            return
        X = torch.FloatTensor(np.stack(self.X_train)).to(self.device)
        Y = torch.FloatTensor(np.stack(self.Y_train)).to(self.device)
        n = len(X)
        for model, opt in zip(self.members, self.optimizers):
            idx = np.random.choice(n, size=n, replace=True)
            X_boot, Y_boot = X[idx], Y[idx]
            model.train()
            for _ in range(epochs):
                perm = torch.randperm(n)
                for j in range(0, n, batch_size):
                    batch_idx = perm[j:j + batch_size]
                    pred = model(X_boot[batch_idx])
                    loss = F.mse_loss(pred, Y_boot[batch_idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        if verbose:
            with torch.no_grad():
                errs = []
                for model in self.members:
                    model.eval()
                    pred = model(X).cpu().numpy()
                    errs.append(np.mean((pred - Y.cpu().numpy())**2))
                model.eval()
            print(f"[Surrogate] Trained on {n} samples, MSE: {np.mean(errs):.4f}")

    def predict(self, M, efC):
        scalar = np.isscalar(M)
        if scalar:
            M_arr = np.array([M])
            efC_arr = np.array([efC])
        else:
            M_arr = np.asarray(M)
            efC_arr = np.asarray(efC)
        x = self._normalize_input(M_arr, efC_arr)
        x_t = torch.FloatTensor(x).to(self.device)
        all_preds = []
        for model in self.members:
            model.eval()
            with torch.no_grad():
                all_preds.append(model(x_t).cpu().numpy())
        all_preds = np.stack(all_preds, axis=0)
        mu = all_preds.mean(axis=0)
        sigma = all_preds.std(axis=0)
        mu_recall, mu_qps = self._denormalize_output(mu)
        hi_qps = self._denormalize_output(mu + sigma)[1]
        lo_qps = self._denormalize_output(mu - sigma)[1]
        sigma_qps = np.maximum(hi_qps - lo_qps, 0.0) / 2.0
        sigma_recall = sigma[..., 0]
        if scalar:
            return (float(mu_recall[0]), float(mu_qps[0]),
                    float(sigma_recall[0]), float(sigma_qps[0]))
        return mu_recall, mu_qps, sigma_recall, sigma_qps

    def predict_reward(self, M, efC, R_target=0.99, lambda_pen=10.0, kappa=50.0):
        scalar = np.isscalar(M)
        if scalar:
            M_arr = np.array([M])
            efC_arr = np.array([efC])
        else:
            M_arr = np.asarray(M)
            efC_arr = np.asarray(efC)

        mu_recall, mu_qps, _, _ = self.predict(M_arr, efC_arr)
        if scalar:
            mu_recall_list = [float(mu_recall[0])]
            mu_qps_list = [float(mu_qps[0])]
        else:
            mu_recall_list = [float(r) for r in mu_recall]
            mu_qps_list = [float(q) for q in mu_qps]

        mu_reward = np.array([
            np.log(max(q, 1e-5))
            - lambda_pen * np.logaddexp(0.0, kappa * (R_target - r))
            for r, q in zip(mu_recall_list, mu_qps_list)
        ])

        x = self._normalize_input(M_arr, efC_arr)
        x_t = torch.FloatTensor(x).to(self.device)
        member_rewards = []
        for model in self.members:
            model.eval()
            with torch.no_grad():
                pred = model(x_t).cpu().numpy()
            rec, q = self._denormalize_output(pred)
            if rec.ndim == 0:
                rec = np.array([float(rec)])
                q = np.array([float(q)])
            rw = np.array([
                np.log(max(float(qq), 1e-5))
                - lambda_pen * np.logaddexp(
                    0.0, kappa * (R_target - float(rr)))
                for rr, qq in zip(rec, q)
            ])
            member_rewards.append(rw)
        sigma_reward = np.std(np.array(member_rewards), axis=0)

        if scalar:
            return float(mu_reward[0]), float(sigma_reward[0])
        return mu_reward, sigma_reward

    def scan_hot_zones(self, R_target=0.99, grid_size=25, top_k=8,
                       lambda_pen=10.0, kappa=50.0):
        M_vals = np.linspace(self.M_min, self.M_max, grid_size)
        efC_vals = np.linspace(self.efC_min, self.efC_max, grid_size)
        MM, EE = np.meshgrid(M_vals, efC_vals)
        M_flat = MM.flatten()
        efC_flat = EE.flatten()
        valid = efC_flat >= M_flat
        M_flat = M_flat[valid]
        efC_flat = efC_flat[valid]

        mu_rw, sigma_rw = self.predict_reward(
            M_flat, efC_flat, R_target=R_target,
            lambda_pen=lambda_pen, kappa=kappa)
        ucb = mu_rw + 2.0 * sigma_rw

        top_idx = np.argsort(ucb)[-top_k:][::-1]
        top_points = []
        for i in top_idx:
            mu_rec, mu_qps, _, _ = self.predict(float(M_flat[i]), float(efC_flat[i]))
            top_points.append({
                'M': int(round(float(M_flat[i]))),
                'efC': int(round(float(efC_flat[i]))),
                'pred_recall': float(mu_rec),
                'pred_qps': float(mu_qps),
                'pred_reward': float(mu_rw[i]),
                'uncertainty': float(sigma_rw[i]),
                'ucb_score': float(ucb[i]),
            })


        all_sorted_idx = np.argsort(ucb)[::-1]
        all_sorted = []
        for i in all_sorted_idx:
            all_sorted.append({
                'M': int(round(float(M_flat[i]))),
                'efC': int(round(float(efC_flat[i]))),
                'ucb_score': float(ucb[i]),
                'pred_reward': float(mu_rw[i]),
            })


        threshold = np.percentile(ucb, 75)
        high_mask = ucb >= threshold
        high_M = M_flat[high_mask]
        high_efC = efC_flat[high_mask]

        if len(high_M) >= 6:
            try:
                from sklearn.cluster import KMeans
                coords = np.stack([
                    (high_M - self.M_min) / (self.M_max - self.M_min),
                    (high_efC - self.efC_min) / (self.efC_max - self.efC_min)
                ], axis=-1)
                k = min(3, len(high_M) // 2)
                km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(coords)
                regions = []
                for c in range(k):
                    mask_c = km.labels_ == c
                    if mask_c.sum() < 2:
                        continue
                    cM = high_M[mask_c]
                    cE = high_efC[mask_c]
                    delta_m = max(4, int(round((cM.max() - cM.min()) / 2)))
                    delta_e = max(10, int(round((cE.max() - cE.min()) / 2)))
                    regions.append({
                        'mu_M': int(round(cM.mean())),
                        'delta_M': delta_m,
                        'mu_ef': int(round(cE.mean())),
                        'delta_ef': delta_e,
                        'n_points': int(mask_c.sum()),
                    })
            except Exception:
                regions = []
        else:
            regions = []

        if not regions and top_points:
            regions = [{
                'mu_M': p['M'], 'delta_M': max(6, int(p['M'] * 0.2)),
                'mu_ef': p['efC'], 'delta_ef': max(30, int(p['efC'] * 0.2)),
            } for p in top_points[:3]]

        return {'top_points': top_points, 'regions': regions, 'all_sorted': all_sorted}

    @property
    def n_samples(self):
        return len(self.X_train)


class CoGuidedLLMProcessor(LLMKnowledgeProcessor):

    """Generate search regions from measurements and model predictions."""

    INITIAL_PROMPT = Template("""You are a senior vector database and ANN indexing researcher.
Your role is to act as the search-space architect for an automated hnswlib HNSW
parameter tuning system. A PPO agent will sample inside the regions you recommend.

Task:
Recommend 2-3 high-potential HNSW build-parameter search regions for GIST1M.

Dataset and evaluation:
- Dataset: GIST1M, 1,000,000 base vectors, 960 dimensions, L2 distance.
- Evaluation uses Recall@10, not Recall@100.
- Target: maximize QPS while satisfying Recall@10 >= ${target_recall}.
- For each candidate graph, the evaluator sweeps efSearch and estimates QPS at
  target recall by interpolation.
- The server uses the common multi-threaded C++ hnswlib/OpenMP timing, so
  focus on graph structures that can achieve high recall without requiring very
  large efSearch.

HNSW parameter meaning:
- M controls graph out-degree/connectivity.
  * Lower M usually improves memory locality and raw QPS.
  * Too low M may create weak connectivity and require high efSearch, hurting QPS
    at Recall@10 >= ${target_recall}.
- efConstruction controls graph construction quality.
  * Higher efConstruction can improve graph quality and reduce query-time efSearch.
  * Too high efConstruction may not improve final QPS enough and increases build cost.

Hard constraints:
- M must be in [4, 100].
- efConstruction must be in [20, 800].
- efConstruction >= M.

Prior knowledge for GIST1M:
- Very high M is usually unnecessary and often lowers QPS.
- Good regions often lie in low-to-mid M with enough efConstruction to preserve recall.
- Recommend at least one aggressive low-M/high-QPS region, one balanced safe region,
  and optionally one narrow expert-prior region if you believe it is promising.

Output requirements:
- Return ONLY strict JSON. No markdown, no prose outside JSON.
- Recommend 2-3 regions.
- Each region is a center plus half-width:
  [mu_M - delta_M, mu_M + delta_M] and [mu_ef - delta_ef, mu_ef + delta_ef].
- Use smaller deltas when confident, larger deltas when uncertainty is high.
- Reasoning should be concise but technical: explain expected QPS/recall tradeoff
  and the risk of the region.

Schema:
{"regions":[
  {"mu_M":24,"delta_M":8,"mu_ef":360,"delta_ef":120,"reasoning":"technical reason"}
]}
${warmup_context}

Now recommend regions using the real warm-up measurements above. If a warm-up
point or warm-up band is clearly strong, center at least one region near it.""")

    REFLECTION_PROMPT = Template("""You are a senior hnswlib HNSW architect cooperating with a PPO-based
parameter tuning agent. Your job is not to output one parameter pair; your job is
to design 1-3 promising local search regions for the next optimization epoch.

Workload and target:
- Dataset: GIST1M, 1,000,000 vectors, 960 dimensions, L2 distance.
- Metric: Recall@10.
- Objective: maximize QPS while satisfying Recall@10 >= ${target_recall}.
- Real evaluation progress: ${total_real_evals} total
  (${warmup_count} warm-up + ${online_count} online).
- The evaluator builds and searches each (M, efConstruction) in one C++ process,
  then interpolates QPS at target recall.
- Search timing uses all 1,000 GIST1M queries with five timed repeats.
  The evaluator reports efSearch
  [10,20,30,40,50,80,100,200,300,600,1200,2400].

HNSW design intuition:
- M controls graph degree/connectivity.
  * Low M tends to improve cache locality, memory footprint, and QPS.
  * Too-low M can damage navigability; then efSearch must grow and QPS collapses.
  * Very-high M can over-connect GIST1M and waste query work.
- efConstruction controls build-time graph quality.
  * Increasing efConstruction can improve neighbor quality and reduce the efSearch
    needed for Recall@10 >= ${target_recall}.
  * Excessive efConstruction may not improve final QPS and costs much more build time.
- Build time is logged for tuning-cost analysis, but it is not part of the reward
  or final parameter ranking. The optimization objective is target-recall QPS.

Hard constraints:
- M in [4, 100].
- efConstruction in [20, 800].
- efConstruction >= M.

Decision policy:
1. Trust real measurements more than surrogate predictions.
2. Use surrogate hot zones only as hypotheses. If a hot zone is low-M but has no
   real high-recall evidence, treat it as risky rather than automatically optimal.
3. If several real points already reach Recall@10 >= ${target_recall}, prefer the one with the
   highest QPS and search around it with moderate deltas.
4. If a region has high recall but mediocre QPS, try lowering M or efConstruction
   slightly, but avoid falling below the recall boundary.
5. If recall is marginal, first increase efConstruction; increase M only when graph
   connectivity appears insufficient.
6. Recommend a mix:
   - one exploitation region centered near the best verified result;
   - one nearby improvement region that might raise QPS;
   - optionally one controlled exploration region if evidence supports it.

Current best verified result:
${best_result}

Recent real evaluations:
${history}

Real warm-up evidence:
${warmup_context}

Surrogate-predicted hot zones:
${hot_zones}

Now recommend the next regions.
Output ONLY strict JSON. Do not output markdown or explanatory text outside JSON.
For each region, give center and half-width. Keep reasoning technical and specific:
mention whether the region is exploitation, QPS-seeking, recall-safety, or exploration,
and explain the expected M/efConstruction tradeoff.

Schema:
{"regions":[
  {"mu_M":24,"delta_M":8,"mu_ef":400,"delta_ef":120,"reasoning":"technical reason"},
  {"mu_M":16,"delta_M":6,"mu_ef":300,"delta_ef":80,"reasoning":"technical reason"}
]}""")

    def __init__(
        self,
        api_key=None,
        llm_type="doubao",
        llm_cache_path=None,
        base_url=None,
        model=None,
        connect_timeout_s=10,
        request_timeout_s=120,
        retries=3,
        retry_backoff_s=10,
        skip_llm=False,
    ):
        super().__init__(
            api_key=api_key,
            llm_type=llm_type,
            base_url=base_url,
            model=model,
            connect_timeout_s=connect_timeout_s,
            request_timeout_s=request_timeout_s,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
            skip_llm=skip_llm,
        )
        self.cache_path = llm_cache_path
        self.cache = {}
        self.call_index = 0
        self.suggestions_made = []
        if llm_cache_path and os.path.exists(llm_cache_path):
            with open(llm_cache_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                self._cached_regions = data
                self.cache = {}
            elif isinstance(data, dict):
                self.cache = {int(k): v for k, v in data.items()}
                self._cached_regions = None
            else:
                self.cache = {}
                self._cached_regions = None
            print(f"[LLM Cache] Loaded from {llm_cache_path}")

    def _save_initial_regions(self, regions):
        path = self.cache_path or os.path.join(
            os.path.dirname(__file__), "logs", "llm_initial_regions.json")
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        regions = self._normalize_regions(regions)
        temp_path = f"{path}.tmp.{os.getpid()}"
        try:
            with open(temp_path, 'w') as f:
                json.dump(regions, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        print(f"[LLM] Initial regions saved to: {path}")

    def _load_initial_regions(self):
        path = self.cache_path or os.path.join(
            os.path.dirname(__file__), "logs", "llm_initial_regions.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                regions = json.load(f)
            regions = self._normalize_regions(regions)
            print(f"[LLM] Loaded initial regions from: {path}")
            return regions
        return None


    def _format_warmup_context(self, warmup_data, target_recall=0.99):
        if not warmup_data:
            return ""

        lines = [
            "",
            "Real warm-up measurements from this run:",
            "| M | efConstruction | Recall@10 | QPS@target | Reward |",
            "|---:|---:|---:|---:|---:|",
        ]
        valid = []
        for w in warmup_data:
            rec = float(w.get('recall', 0.0))
            qps = float(w.get('qps_at_target', 0.0))
            rw = float(w.get('reward', 0.0))
            M = int(w['M'])
            efC = int(w['efC'])
            lines.append(f"| {M} | {efC} | {rec:.4f} | {qps:.1f} | {rw:.3f} |")
            valid.append((qps, rec, M, efC, rw))

        best_qps = max(valid, key=lambda x: (x[0], x[1]))
        best_recall = max(valid, key=lambda x: (x[1], x[0]))
        lines.extend([
            "",
            "Warm-up interpretation to use as evidence:",
            f"- Best QPS@target warm-up point: M={best_qps[2]}, efConstruction={best_qps[3]}, "
            f"Recall@10={best_qps[1]:.4f}, QPS@target={best_qps[0]:.1f}.",
            f"- Highest-recall warm-up point: M={best_recall[2]}, efConstruction={best_recall[3]}, "
            f"Recall@10={best_recall[1]:.4f}, QPS@target={best_recall[0]:.1f}.",
            "- GIST1M prior: strong regions often appear in low-to-mid "
            f"M with enough efConstruction to preserve Recall@10 >= {target_recall}; very high M "
            "is useful as validation evidence but is not automatically best for QPS.",
            "- Use these real measurements more strongly than generic prior knowledge.",
            "",
        ])
        return "\n".join(lines)

    def get_architect_initial(self, warmup_data=None, target_recall=0.99):

        saved = self._load_initial_regions()
        if saved:
            self.suggestions_made = saved
            print(f"[LLM Architect] Initial: {len(saved)} regions (from saved file)")
            for i, r in enumerate(saved):
                print(f"  Region {i+1}: M={r['mu_M']}±{r['delta_M']}, "
                      f"efC={r['mu_ef']}±{r['delta_ef']} | {r.get('reasoning', '?')[:60]}")
            return saved

        warmup_context = self._format_warmup_context(warmup_data, target_recall)


        prompt = self.INITIAL_PROMPT.substitute(target_recall=target_recall,
                               warmup_context=warmup_context)

        default_regions = [
            {"mu_M": 24, "delta_M": 12, "mu_ef": 300, "delta_ef": 120,
             "reasoning": "Default: broad low-to-mid-M exploration for GIST1M"},
            {"mu_M": 44, "delta_M": 10, "mu_ef": 450, "delta_ef": 100,
             "reasoning": "Default: mid-M conservative region"},
        ]
        try:
            resp = self._call_api(prompt, timeout=240, max_tokens=2048)
            if not resp:
                raise ValueError("Empty response")
            regions = self._parse_regions(resp)
            self._save_initial_regions(regions)
            self.suggestions_made = regions
            print(f"[LLM Architect] Initial: {len(regions)} regions from LLM")
            for i, r in enumerate(regions):
                print(f"  Region {i+1}: M={r['mu_M']}±{r['delta_M']}, "
                      f"efC={r['mu_ef']}±{r['delta_ef']} | {r.get('reasoning', '?')[:60]}")
            return regions
        except Exception as e:
            print(f"[LLM Architect] Initial failed ({e}), using default regions")
            self._save_initial_regions(default_regions)
            self.suggestions_made = default_regions
            return default_regions

    def get_architect_reflection(self, history_records, best_result, hot_zone_data, warmup_data=None,
                                 target_recall=0.99):
        all_online_real = [
            r for r in history_records if r.get('recall') is not None
        ]
        recent = all_online_real[-20:]
        warmup_count = len(warmup_data or [])
        warmup_context = self._format_warmup_context(warmup_data, target_recall)
        history_lines = []
        for r in recent[-15:]:
            history_lines.append(
                f"M={r['M']:3d} efC={r['efC']:3d} | "
                f"recall={float(r['recall']):.4f} QPS@R={target_recall}={float(r['qps_at_target']):.0f} | "
                f"reward={float(r['reward']):.3f} | src={r.get('source', '?')}"
            )

        if best_result:
            best_str = (f"M={best_result['M']}, efConstruction={best_result['efC']}, "
                        f"recall@10={float(best_result['recall']):.4f}, "
                        f"QPS@R={target_recall}={float(best_result['qps_at_target']):.0f}")
        else:
            best_str = f"No result satisfying recall>={target_recall} yet; keep exploring."

        hot_lines = []
        if hot_zone_data and hot_zone_data.get('top_points'):
            for i, p in enumerate(hot_zone_data['top_points'][:5]):
                hot_lines.append(
                    f"  #{i+1} M={p['M']:3d} efC={p['efC']:3d} | "
                    f"predicted recall={p['pred_recall']:.4f} predicted QPS={p['pred_qps']:.0f} | "
                    f"predicted reward={p['pred_reward']:.2f} (uncertainty={p['uncertainty']:.3f})"
                )
        hot_str = "\n".join(hot_lines) if hot_lines else "(insufficient surrogate data)"

        total_real_evals = warmup_count + len(all_online_real)


        prompt = self.REFLECTION_PROMPT.substitute(
            target_recall=target_recall,
            total_real_evals=total_real_evals, warmup_count=warmup_count,
            online_count=len(all_online_real), best_result=best_str,
            history="\n".join(history_lines) if history_lines else "(none)",
            warmup_context=warmup_context or "(none)", hot_zones=hot_str)

        default_regions = [
            {"mu_M": 32, "delta_M": 12, "mu_ef": 400, "delta_ef": 120,
             "reasoning": "Fallback: conservative default region"}
        ]
        try:
            resp = self._call_api(prompt, timeout=300, max_tokens=2048)
            if not resp:
                raise ValueError("Empty response")
            regions = self._parse_regions(resp)
            print(f"[LLM Architect] Reflection: {len(regions)} regions")
            for i, r in enumerate(regions[:2]):
                print(f"  R{i+1}: M={r['mu_M']}±{r['delta_M']}, "
                      f"efC={r['mu_ef']}±{r['delta_ef']} | {r.get('reasoning', '?')[:60]}")
            return regions
        except Exception as e:
            print(f"[LLM Architect] Reflection failed ({e}), using safe fallback")
            if best_result:
                best_M = int(best_result['M'])
                best_efC = int(best_result['efC'])
                default_regions = [{
                    "mu_M": best_M,
                    "delta_M": max(6, min(12, best_M // 2)),
                    "mu_ef": best_efC,
                    "delta_ef": max(60, min(150, best_efC // 4)),
                    "reasoning": f"Fallback: centered on current best after API error ({str(e)[:50]})"
                }]
            elif hot_zone_data and hot_zone_data.get('top_points'):
                p = hot_zone_data['top_points'][0]
                safe_M = max(16, int(p['M']))
                safe_efC = max(safe_M, int(p['efC']))
                default_regions = [{
                    "mu_M": safe_M, "delta_M": 8,
                    "mu_ef": safe_efC, "delta_ef": 80,
                    "reasoning": f"Fallback: clamped surrogate top pick after API error ({str(e)[:50]})"
                }]
            return default_regions


def train_coguided(llm_cache_path=None, seed_override=None, log_dir=None,
                   target_recall_override=None):
    """Run warmup, guided optimization, and final evaluation."""
    experiment_start = time.time()
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            'common': {'random_seed': 42, 'gamma': 0.99, 'K_epochs': 40,
                       'eps_clip': 0.2, 'update_timestep': 12,
                       'lr_actor': 0.0003, 'lr_critic': 0.002},
            'lrrl': {'R_target': 0.99, 'lambda_penalty': 10.0, 'kappa': 50.0,
                     'trust_alpha0': 3.0, 'trust_beta0': 1.0, 'trust_gamma': 0.95,
                     'boundary_beta': 5.0, 'epoch_steps': 6,
                     'epsilon_explore': 0.10, 'trust_cap': 0.85},
            'hnsw': {'M_min': 4, 'M_max': 100, 'efConstruction_min': 20,
                     'efConstruction_max': 800, 'state_dim': 8, 'action_dim': 2}
        }

    common = cfg['common']
    lrrl_cfg = cfg['lrrl']
    hnsw_cfg = cfg['hnsw']
    llm_cfg = cfg.get('llm', {})
    runtime_cfg = cfg.get('runtime', {})
    if target_recall_override is None:
        target_recall_override = os.environ.get("TARGET_RECALL")
    if target_recall_override is not None:
        lrrl_cfg['R_target'] = target_recall_override
    R_target = validate_target_recall(lrrl_cfg['R_target'])
    if not any(
        math.isclose(R_target, target, rel_tol=0.0, abs_tol=1e-12)
        for target in SUPPORTED_TARGET_RECALLS
    ):
        raise ValueError(
            f"Unsupported target recall {R_target}; "
            f"supported={SUPPORTED_TARGET_RECALLS}"
        )
    lrrl_cfg['R_target'] = R_target
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if log_dir is None:
        target_tag = f"{int(round(R_target * 100)):03d}"
        run_id = time.strftime(f"actuner_R{target_tag}_%Y%m%d_%H%M%S")
        log_dir = os.path.join(project_dir, "runs", run_id)
    log_dir = str(project_path(log_dir).resolve())
    allowed_existing = {"preflight.log", "console.log"}
    if os.path.isdir(log_dir):
        unexpected = sorted(
            name for name in os.listdir(log_dir)
            if name not in allowed_existing
        )
        if unexpected:
            raise FileExistsError(
                "Refusing to mix with or overwrite an existing experiment "
                f"directory: {log_dir}; existing entries={unexpected[:10]}"
            )
    elif os.path.exists(log_dir):
        raise FileExistsError(f"Result path exists and is not a directory: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)
    result_logger.LOG_DIR = log_dir
    with open(os.path.join(log_dir, "effective_config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
    timeline_path = os.path.join(log_dir, "real_eval_timeline.csv")
    timeline_fields = [
        "evaluation_event_index", "real_eval_index", "counts_toward_budget",
        "phase", "ppo_step", "cum_time_s", "cum_time_min", "wall_end_time",
        "target_recall", "M", "efConstruction", "max_recall", "qps_at_target",
        "target_reached", "full_max_recall", "full_qps_at_target",
        "evaluation_time_s", "build_time_s", "search_time_s",
        "internal_build_time_s", "index_size_bytes", "source",
        "binary_efsearch", "performance_curve",
    ]
    with open(timeline_path, "w", newline="", encoding="utf-8") as timeline_file:
        csv.DictWriter(timeline_file, fieldnames=timeline_fields).writeheader()

    timeline_event_count = 0

    def record_real_evaluation(
        index,
        phase,
        step,
        M,
        efC,
        recall,
        qps,
        source,
        metrics=None,
        counts_toward_budget=True,
    ):
        nonlocal timeline_event_count
        timeline_event_count += 1
        metrics = metrics or {}
        cum_time_s = time.time() - experiment_start
        row = {
            "evaluation_event_index": timeline_event_count,
            "real_eval_index": int(index) if counts_toward_budget else "",
            "counts_toward_budget": bool(counts_toward_budget),
            "phase": phase,
            "ppo_step": step,
            "cum_time_s": cum_time_s,
            "cum_time_min": cum_time_s / 60.0,
            "wall_end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "target_recall": float(R_target),
            "M": int(M),
            "efConstruction": int(efC),
            "max_recall": float(recall),
            "qps_at_target": float(qps),
            "target_reached": bool(metrics.get("target_reached", qps > 0.0)),
            "full_max_recall": metrics.get("full_max_recall", recall),
            "full_qps_at_target": metrics.get("full_qps_at_target", qps),
            "evaluation_time_s": metrics.get("total_eval_time_s", ""),
            "build_time_s": metrics.get("build_time_s", ""),
            "search_time_s": metrics.get("search_time_s", ""),
            "internal_build_time_s": metrics.get("internal_build_time_s", ""),
            "index_size_bytes": metrics.get("index_size_bytes", ""),
            "source": source,
            "binary_efsearch": json.dumps(
                metrics.get("binary_efsearch", []), separators=(",", ":")
            ),
            "performance_curve": metrics.get("performance_curve", ""),
        }
        with open(timeline_path, "a", newline="", encoding="utf-8") as timeline_file:
            writer = csv.DictWriter(timeline_file, fieldnames=timeline_fields)
            writer.writerow(row)
            timeline_file.flush()
            os.fsync(timeline_file.fileno())
        return row["cum_time_min"]

    seed = common['random_seed']
    if seed_override is not None:
        seed = seed_override
        print(f"[Seed] Override: {seed}")
    gamma = common['gamma']
    K_epochs = common['K_epochs']
    eps_clip = common['eps_clip']
    lr_actor = common['lr_actor']
    lr_critic = common['lr_critic']
    update_timestep = common['update_timestep']

    lambda_pen = float(lrrl_cfg['lambda_penalty'])
    kappa = float(lrrl_cfg['kappa'])
    boundary_beta = float(lrrl_cfg['boundary_beta'])
    alpha_trust = float(lrrl_cfg['trust_alpha0'])
    beta_trust = float(lrrl_cfg['trust_beta0'])
    gamma_trust = float(lrrl_cfg['trust_gamma'])
    if (not all(np.isfinite(value) for value in (
            lambda_pen, kappa, boundary_beta, alpha_trust, beta_trust, gamma_trust))):
        raise ValueError("Reward and trust hyperparameters must be finite")
    if lambda_pen < 0.0 or kappa < 0.0 or boundary_beta < 0.0:
        raise ValueError("lambda_penalty, kappa, and boundary_beta must be non-negative")
    if alpha_trust <= 0.0 or beta_trust <= 0.0 or not 0.0 < gamma_trust <= 1.0:
        raise ValueError(
            "trust_alpha0 and trust_beta0 must be positive; trust_gamma must be in (0, 1]"
        )

    epoch_steps = int(lrrl_cfg.get('epoch_steps', 6))
    epsilon = float(lrrl_cfg.get('epsilon_explore', 0.10))
    trust_cap = float(lrrl_cfg.get('trust_cap', 0.85))

    M_min = int(hnsw_cfg['M_min'])
    M_max = int(hnsw_cfg['M_max'])
    efC_min = int(hnsw_cfg['efConstruction_min'])
    efC_max = int(hnsw_cfg['efConstruction_max'])
    state_dim = int(hnsw_cfg['state_dim'])
    action_dim = int(hnsw_cfg['action_dim'])


    n_total_budget = int(lrrl_cfg.get('n_total_budget', 30))
    n_warmup = int(lrrl_cfg.get('n_warmup', 10))
    n_real_per_step = int(lrrl_cfg.get('n_real_per_step', 2))
    n_ppo_batch = int(lrrl_cfg.get('n_ppo_batch', 6))
    surrogate_retrain_freq = int(lrrl_cfg.get('surrogate_retrain_freq', 4))
    if n_total_budget <= 0:
        raise ValueError(f"n_total_budget must be positive, got {n_total_budget}")
    if not 0 < n_warmup <= n_total_budget:
        raise ValueError(
            f"n_warmup must be in [1, n_total_budget], got {n_warmup}"
        )
    if not 0 < n_real_per_step <= n_ppo_batch:
        raise ValueError(
            "n_real_per_step must be positive and no larger than n_ppo_batch; "
            f"got {n_real_per_step} and {n_ppo_batch}"
        )
    if (n_total_budget, n_warmup, n_real_per_step) != (30, 10, 2):
        raise ValueError(
            "GIST1M evaluation requires "
            "n_total_budget=30, n_warmup=10, n_real_per_step=2; got "
            f"{(n_total_budget, n_warmup, n_real_per_step)}"
        )
    if epoch_steps <= 0 or surrogate_retrain_freq <= 0 or update_timestep <= 0:
        raise ValueError(
            "epoch_steps, surrogate_retrain_freq, and update_timestep must be positive"
        )
    if (not np.isfinite(epsilon) or not np.isfinite(trust_cap)
            or not 0.0 <= epsilon <= 1.0 or not 0.0 <= trust_cap <= 1.0):
        raise ValueError("epsilon_explore and trust_cap must be in [0, 1]")
    if (M_min, M_max, efC_min, efC_max) != (4, 100, 20, 800):
        raise ValueError(
            "GIST1M evaluation requires bounds "
            "M=[4,100] and efConstruction=[20,800]"
        )
    if state_dim != 8 or action_dim != 2:
        raise ValueError("This PPO implementation requires state_dim=8 and action_dim=2")
    kappa_acq_init = 3.0
    kappa_acq_final = 0.5
    lcb_beta = 1.0

    action_std_init = float(lrrl_cfg.get("action_std_init", 0.65))
    action_std_decay_rate = float(lrrl_cfg.get("action_std_decay_rate", 0.01))
    min_action_std = float(lrrl_cfg.get("min_action_std", 0.25))
    action_std_decay_freq = int(lrrl_cfg.get("action_std_decay_freq", 5))
    diversity_candidate_every = int(
        lrrl_cfg.get("diversity_candidate_every", 1)
    )
    if (
        not np.isfinite(action_std_init)
        or not np.isfinite(action_std_decay_rate)
        or not np.isfinite(min_action_std)
        or action_std_init <= 0.0
        or action_std_decay_rate < 0.0
        or min_action_std <= 0.0
        or min_action_std > action_std_init
        or action_std_decay_freq <= 0
        or diversity_candidate_every <= 0
    ):
        raise ValueError("Invalid PPO exploration/decay configuration")

    remaining_budget = n_total_budget - n_warmup
    max_steps = math.ceil(remaining_budget / n_real_per_step) if remaining_budget else 0

    default_efsearch = [
        10, 20, 30, 40, 50, 80, 100, 200, 300, 600, 1200, 2400
    ]
    efSearch_train = hnsw_cfg.get('efSearch_train', default_efsearch)
    efSearch_full = hnsw_cfg.get('efSearch_full', [10, 20, 30, 40, 50, 80, 100, 200, 300, 600, 1200, 2400])
    recall_k = int(hnsw_cfg.get('recall_k', 10))

    set_seed(seed)
    configure_qps_state_scale(lrrl_cfg.get("qps_state_scale", 2000.0))
    configure_hnsw_threads(runtime_cfg)
    search_repeat = int(runtime_cfg.get('search_repeat', 5))


    base, query, gt = load_gist1m(config_path)
    data_cfg = cfg.get('data', {})
    dataset_N = int(base.shape[0])
    dataset_dim = int(base.shape[1])
    dataset_name = str(data_cfg.get('name') or infer_dataset_name(data_cfg, dataset_dim))
    train_query_sample = int(runtime_cfg.get(
        'train_query_sample', data_cfg.get('query_limit', len(query))))
    if train_query_sample != len(query):
        raise ValueError(
            "test_search reads the complete query file; "
            f"train_query_sample must equal {len(query)}, got {train_query_sample}"
        )

    binary_manifest = {}
    for label, setting in (
        ("test_index", "hnsw_build_exe"),
        ("test_search", "hnsw_search_exe"),
    ):
        binary_path = project_path(runtime_cfg.get(setting, f"tests/{label}"))
        if not os.path.isfile(binary_path):
            raise FileNotFoundError(f"Missing {label}: {binary_path}")
        binary_manifest[label] = {
            "path": os.path.abspath(binary_path),
            "size_bytes": os.path.getsize(binary_path),
            "sha256": file_sha256(binary_path),
        }
    run_manifest = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target_recall": R_target,
        "dataset": {
            "name": dataset_name,
            "base_path": getattr(base, "path", data_cfg.get("base_path")),
            "query_path": getattr(query, "path", data_cfg.get("query_path")),
            "groundtruth_path": getattr(
                gt, "path", data_cfg.get("groundtruth_path")
            ),
            "base_shape": list(base.shape),
            "query_shape": list(query.shape),
            "groundtruth_shape": list(gt.shape),
        },
        "protocol": {
            "recall_k": recall_k,
            "search_repeat_compiled": search_repeat,
            "efSearch_train": list(efSearch_train),
            "efSearch_full": list(efSearch_full),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_dynamic": os.environ.get("OMP_DYNAMIC"),
            "note": (
                "Search uses schedule(static,100), providing ten query "
                "chunks for the 1000-query GIST1M file."
            ),
        },
        "binaries": binary_manifest,
    }
    with open(
        os.path.join(log_dir, "run_manifest.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(run_manifest, handle, ensure_ascii=False, indent=2)


    casebase_path = runtime_cfg.get('case_base_path', 'state/case_base.json')
    if not os.path.isabs(casebase_path):
        casebase_path = os.path.join(project_dir, casebase_path)
    casebase = CaseBase(casebase_path)
    w_evidence = casebase.weight_evidence(tau=3.0)
    eta = w_evidence

    emp_prior, retrieved_cases = casebase.empirical_prior(
        N=dataset_N, dim=dataset_dim, target_recall=R_target, top_k=3)

    print("=" * 65)
    print("ACTuner")
    print(f"Dataset: {dataset_name} | N={dataset_N} | dim={dataset_dim}")
    print(f"Training real-eval queries: {train_query_sample}/{len(query)}")
    print(f"CaseBase |D|={casebase.size} | w_evidence={w_evidence:.3f}")
    if retrieved_cases:
        top_sim, top_case = retrieved_cases[0]
        print(f"CaseBase top match: {top_case['dataset']} | similarity={top_sim:.3f} "
              f"| prior M={top_case['best_region']['M']}, efC={top_case['best_region']['efC']}")
    print(f"Target recall: {R_target}  |  M ∈ [{M_min}, {M_max}]  |  efC ∈ [{efC_min}, {efC_max}]")
    print(f"Budget: {n_total_budget} real evals ({n_warmup} warmup + {max_steps}×{n_real_per_step})")
    print(f"PPO batch: {n_ppo_batch}  |  Action std decay: {action_std_decay_rate} / {action_std_decay_freq} steps")
    print(f"UCB κ: {kappa_acq_init} → {kappa_acq_final}  |  LCB β: {lcb_beta}")
    print("=" * 65)


    hnsw_env = HNSWEnv(base, query, gt, R_target=R_target,
                              efSearch_train=efSearch_train,
                              efSearch_full=efSearch_full,
                              n_query_sample=train_query_sample,
                              search_repeat=search_repeat,
                              recall_k=recall_k)
    api_key = os.environ.get(llm_cfg.get('api_key_env', 'DOUBAO_API_KEY'), '')
    base_url = os.environ.get(
        llm_cfg.get('base_url_env', 'DOUBAO_BASE_URL'), llm_cfg.get('base_url'))
    model = os.environ.get(
        llm_cfg.get('model_env', 'DOUBAO_MODEL'), llm_cfg.get('model'))
    llm = CoGuidedLLMProcessor(
        api_key=api_key, llm_type="doubao", llm_cache_path=llm_cache_path,
        base_url=base_url, model=model,
        connect_timeout_s=llm_cfg.get("connect_timeout_s", 10),
        request_timeout_s=llm_cfg.get("request_timeout_s", 120),
        retries=llm_cfg.get("retries", 3),
        retry_backoff_s=llm_cfg.get("retry_backoff_s", 10),
        skip_llm=os.environ.get("SKIP_LLM", "").strip().lower()
        in {"1", "true", "yes", "on"})
    if llm_cache_path and llm.cache:
        print(f"[LLM Cache] Replaying {len(llm.cache)} cached suggestions (no API calls)")
    elif llm_cache_path:
        print(f"[LLM Cache] New cache file: {llm_cache_path} (will save suggestions after run)")
    ppo = PPO(state_dim=state_dim, action_dim=action_dim,
              lr_actor=lr_actor, lr_critic=lr_critic,
              gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip,
              action_std_init=action_std_init)
    surrogate = SurrogateEnsemble(n_members=5, hidden_dim=64, device=str(device))


    print("\n" + "-" * 50)
    print(f"Phase 1: Warm-up ({n_warmup} evals)")
    print("-" * 50)

    warmup_cache = runtime_cfg.get('warmup_cache')
    if warmup_cache and not os.path.isabs(warmup_cache):
        warmup_cache = os.path.join(project_dir, warmup_cache)
    if warmup_cache:
        raise ValueError(
            "warmup_cache must be empty for GIST1M evaluation. "
            "Run a fresh warm-up so binary hash, target, full curve and "
            "cumulative time cannot be mixed with another evaluator."
        )
    print("  Warm-up evaluations")
    warmup_candidates = select_warmup_candidates(R_target)
    if len(warmup_candidates) < n_warmup:
        raise ValueError(
            f"Warmup pool for target {R_target} has {len(warmup_candidates)} "
            f"unique points but n_warmup={n_warmup}"
        )
    print(
        f"  Target-independent stratified warmup for R_target={R_target}: "
        f"{warmup_candidates[:n_warmup]}"
    )
    warmup_results = []
    for i, (M, efC) in enumerate(warmup_candidates[:n_warmup]):
        print(
            f"  Warm-up {i+1}/{n_warmup}: M={M:3d}, "
            f"efC={efC:3d} ... ",
            end="",
            flush=True,
        )
        t0 = time.time()
        survived, recall, qps, _, interp, raw_results = (
            hnsw_env.build_and_evaluate(M, efC)
        )
        dt = time.time() - t0
        eval_meta = dict(hnsw_env.last_evaluation or {})
        reward = compute_reward(
            recall,
            qps,
            R_target,
            lambda_pen,
            kappa,
            boundary_beta,
            np.array([0.0, 0.0], dtype=np.float32),
        )
        cum_time_min = record_real_evaluation(
            i + 1,
            "warmup",
            i + 1,
            M,
            efC,
            recall,
            qps,
            "warmup",
            metrics=eval_meta,
        )
        warmup_results.append({
            'M': M, 'efC': efC, 'recall': float(recall),
            'qps_at_target': float(qps), 'reward': float(reward),
            'interp': interp, 'time': dt, 'cum_time_min': cum_time_min,
            'raw_results': list(raw_results),
            'raw_results_full': list(eval_meta.get('raw_results_full', [])),
            'evaluation_metrics': eval_meta,
            'phase': 'warmup', 'global_eval_index': i + 1,
        })
        save_param_result(
            M,
            efC,
            recall,
            qps,
            reward,
            interp,
            raw_results,
            eval_meta.get('raw_results_full'),
            dt,
            source='warmup',
            step=i + 1,
            survived=survived,
            target_recall=R_target,
            cum_time_min=cum_time_min,
            evaluation_metrics=eval_meta,
            global_eval_index=i + 1,
            phase='warmup',
        )
        surrogate.add_data(M, efC, recall, qps)
        print(
            f"recall={float(recall):.4f} "
            f"QPS@R={R_target}={float(qps):.0f} "
            f"reward={float(reward):.3f} ({dt:.0f}s) "
            f"cum_time={cum_time_min:.4f}min"
        )

    if len(warmup_results) != n_warmup:
        raise RuntimeError(
            f"Warmup produced {len(warmup_results)} results; expected {n_warmup}"
        )

    print(f"\n  Training initial surrogate on {surrogate.n_samples} samples...")
    initial_surrogate_started = time.time()
    surrogate.train(epochs=300, verbose=True)
    initial_surrogate_train_s = time.time() - initial_surrogate_started


    print(f"\n  Running initial hot zone scan (625-point grid)...")
    initial_hot_zone_started = time.time()
    hot_zone_cache = surrogate.scan_hot_zones(
        R_target=R_target, grid_size=25, top_k=8,
        lambda_pen=lambda_pen, kappa=kappa)
    initial_hot_zone_scan_s = time.time() - initial_hot_zone_started
    print(f"  Top UCB zones:")
    for p in hot_zone_cache['top_points'][:4]:
        print(f"    M={p['M']:3d} efC={p['efC']:3d} → "
              f"pred_recall={p['pred_recall']:.4f} pred_QPS={p['pred_qps']:.0f} "
              f"ucb={p['ucb_score']:.2f}")


    print("\n" + "-" * 50)
    print(f"Phase 2: Co-Guided PPO ({max_steps} steps, {n_real_per_step} real/step)")
    print("-" * 50)

    timestep = 0
    n_real_evals = len(warmup_results)
    total_surrogate_evals = 0
    llm_call_count = 0

    best_result = None
    best_global_qps = 0.0
    best_global_recall = max(float(row['recall']) for row in warmup_results)
    ema_all_reward = 0.0
    ema_llm_reward = 0.0
    history_records = []
    evaluated_params = {(int(row['M']), int(row['efC'])) for row in warmup_results}

    feasible_warmups = [
        (index, row)
        for index, row in enumerate(warmup_results, start=1)
        if (recall_target_reached(row['recall'], R_target)
            and float(row['qps_at_target']) > 0.0)
    ]
    if feasible_warmups:
        warmup_index, warmup_best = max(
            feasible_warmups, key=lambda item: float(item[1]['qps_at_target'])
        )
        best_global_qps = float(warmup_best['qps_at_target'])
        best_result = {
            'M': int(warmup_best['M']),
            'efC': int(warmup_best['efC']),
            'recall': float(warmup_best['recall']),
            'qps_at_target': best_global_qps,
            'interp': warmup_best.get('interp'),
            'step': warmup_index,
            'source': 'warmup',
            'cum_time_min': warmup_best.get('cum_time_min'),
            'raw_results': warmup_best.get('raw_results'),
            'raw_results_full': warmup_best.get('raw_results_full'),
            'evaluation_metrics': warmup_best.get('evaluation_metrics', {}),
            'global_eval_index': warmup_best.get(
                'global_eval_index', warmup_index
            ),
        }
        print(
            f"  Warm-up best retained: M={best_result['M']}, "
            f"efC={best_result['efC']}, QPS@R={R_target}={best_global_qps:.0f}"
        )

    global_mu_M = (M_min + M_max) / 2
    global_delta_M = (M_max - M_min) / 2
    global_mu_ef = (efC_min + efC_max) / 2
    global_delta_ef = (efC_max - efC_min) / 2

    llm_regions = None

    stats = {
        'real_eval': sum(
            float(row.get('time', 0.0)) for row in warmup_results
        ),
        'surrogate_eval': 0.0,
        'ppo_update': 0.0,
        'surrogate_train': initial_surrogate_train_s,
        'hot_zone_scan': initial_hot_zone_scan_s,
        'llm_call': 0.0,
    }
    source_stats = {'LLM-guided': 0, 'Global Explore': 0, 'Explore (ε)': 0}


    surrogate_train_history = []
    llm_suggestions = []
    step_log = []


    t_llm0 = time.time()
    try:
        llm_regions = llm.get_architect_initial(
            warmup_results, target_recall=R_target)
        llm_call_count = int(getattr(llm, "api_request_attempts", 0))
        for i, r in enumerate(llm_regions):
            llm_suggestions.append({
                'epoch': 0, 'type': f'initial_region_{i+1}',
                'M': r['mu_M'], 'efConstruction': r['mu_ef'],
                'delta_M': r['delta_M'], 'delta_ef': r['delta_ef'],
                'reasoning': r.get('reasoning', ''),
                'fallback': False,
            })
    except Exception as e:
        print(f"[WARN] Initial LLM failed: {e}")
        llm_regions = [{"mu_M": 32, "delta_M": 12, "mu_ef": 340, "delta_ef": 120,
                         "reasoning": "Fallback default"}]
        llm_suggestions.append({
            'epoch': 0, 'type': 'initial',
            'M': 32, 'efConstruction': 340,
            'reasoning': f"Fallback: {e}",
            'fallback': True,
        })
    stats['llm_call'] += time.time() - t_llm0

    active_region_idx = 0
    stagnation_counter = 0
    stagnation_threshold = 2
    best_qps_per_region = {i: 0.0 for i in range(len(llm_regions))}
    region_step_count = {i: 0 for i in range(len(llm_regions))}

    active_region = llm_regions[active_region_idx]
    print(f"[Init] {len(llm_regions)} regions | Active: R{active_region_idx+1} "
          f"M={active_region['mu_M']}±{active_region['delta_M']}, "
          f"efC={active_region['mu_ef']}±{active_region['delta_ef']}")

    t_start = experiment_start

    while timestep < max_steps and n_real_evals < n_total_budget:


        epoch = timestep // epoch_steps
        is_epoch_start = (timestep % epoch_steps == 0)

        if is_epoch_start and timestep > 0:

            if surrogate.n_samples >= 10:
                t_scan = time.time()
                hot_zone_cache = surrogate.scan_hot_zones(
                    R_target=R_target, grid_size=25, top_k=8,
                    lambda_pen=lambda_pen, kappa=kappa)
                stats['hot_zone_scan'] += time.time() - t_scan
                print(f"\n[Surrogate Scan] Top predicted hot zones:")
                for p in hot_zone_cache['top_points'][:4]:
                    print(f"  M={p['M']:3d} efC={p['efC']:3d} → "
                          f"pred_recall={p['pred_recall']:.4f} pred_QPS={p['pred_qps']:.0f} "
                          f"ucb={p['ucb_score']:.2f}")


            t_llm = time.time()
            try:
                llm_regions = llm.get_architect_reflection(
                    history_records, best_result, hot_zone_cache, warmup_results,
                    target_recall=R_target)
                llm_call_count = int(getattr(llm, "api_request_attempts", 0))
                stats['llm_call'] += time.time() - t_llm
            except Exception as e:
                print(f"[WARN] LLM reflection failed: {e}")
                if hot_zone_cache and hot_zone_cache.get('top_points'):
                    p = hot_zone_cache['top_points'][0]
                    llm_regions = [{"mu_M": p['M'], "delta_M": 8,
                                    "mu_ef": p['efC'], "delta_ef": 60,
                                    "reasoning": "Surrogate fallback"}]
                else:
                    llm_regions = [{"mu_M": 32, "delta_M": 12,
                                    "mu_ef": 400, "delta_ef": 120,
                                    "reasoning": "Default fallback"}]


            active_region_idx = 0
            stagnation_counter = 0
            best_qps_per_region = {i: 0.0 for i in range(len(llm_regions))}
            region_step_count = {i: 0 for i in range(len(llm_regions))}

            active_region = llm_regions[active_region_idx]
            print(f"\n[Epoch {epoch}] LLM Architect ({len(llm_regions)} regions): "
                  f"R1 active: M={active_region['mu_M']}±{active_region['delta_M']}, "
                  f"efC={active_region['mu_ef']}±{active_region['delta_ef']} "
                  f"| {active_region.get('reasoning', '?')[:60]}")

            for i, r in enumerate(llm_regions[:2]):
                llm_suggestions.append({
                    'epoch': epoch, 'type': f'reflection_region_{i+1}',
                    'M': r['mu_M'], 'efConstruction': r['mu_ef'],
                    'delta_M': r['delta_M'], 'delta_ef': r['delta_ef'],
                    'reasoning': r.get('reasoning', ''),
                    'fallback': 'fallback' in r.get('reasoning', '').lower(),
                })


        if llm_regions:
            active_region = llm_regions[active_region_idx]
            anchor_M = active_region['mu_M']
            anchor_efC = active_region['mu_ef']

            frac = max(0.15, ppo.action_std / action_std_init)
            prior_delta_M = max(4, int(active_region['delta_M'] * frac))
            prior_delta_ef = max(10, int(active_region['delta_ef'] * frac))
        else:
            anchor_M = int(global_mu_M)
            anchor_efC = int(global_mu_ef)
            prior_delta_M = int(global_delta_M)
            prior_delta_ef = int(global_delta_ef)


        c_online = alpha_trust / (alpha_trust + beta_trust)
        c_retrieval = emp_prior["top_similarity"] if emp_prior else 0.0
        c_t = min(eta * c_retrieval + (1.0 - eta) * c_online, trust_cap)


        surr_guide = None
        if hot_zone_cache and 'all_sorted' in hot_zone_cache:
            for p in hot_zone_cache['all_sorted']:
                if (p['M'], p['efC']) not in evaluated_params:
                    surr_guide = p
                    break


        if llm_regions and len(llm_regions) > 1:
            n_regions = len(llm_regions)
            region_weights = [0.50] + [(1.0 - 0.50) / max(n_regions - 1, 1)] * (n_regions - 1)
        else:
            region_weights = [1.0]

        if random.random() < epsilon:

            if surr_guide:
                mu_M, delta_M = float(surr_guide['M']), 8.0
                mu_ef, delta_ef = float(surr_guide['efC']), 60.0
                source = "Explore(ε-Surr)"
            elif llm_regions:
                picked_region_idx = random.choices(range(len(llm_regions)), weights=region_weights, k=1)[0]
                picked_region = llm_regions[picked_region_idx]
                frac_eps = max(0.15, ppo.action_std / action_std_init)
                mu_M = float(picked_region['mu_M'])
                mu_ef = float(picked_region['mu_ef'])
                delta_M = max(4.0, float(picked_region['delta_M']) * frac_eps)
                delta_ef = max(10.0, float(picked_region['delta_ef']) * frac_eps)
                source = f"Explore(ε-R{picked_region_idx+1})"
            else:
                mu_M, delta_M = global_mu_M, global_delta_M
                mu_ef, delta_ef = global_mu_ef, global_delta_ef
                source = "Explore(ε-Global)"
        elif random.random() < c_t and llm_regions is not None:
            mu_M, delta_M = float(anchor_M), float(prior_delta_M)
            mu_ef, delta_ef = float(anchor_efC), float(prior_delta_ef)
            source = f"LLM-guided(R{active_region_idx+1})"
        else:

            if surr_guide:
                mu_M, delta_M = float(surr_guide['M']), 8.0
                mu_ef, delta_ef = float(surr_guide['efC']), 60.0
                source = "Surrogate-guided"
            else:
                mu_M, delta_M = global_mu_M, global_delta_M
                mu_ef, delta_ef = global_mu_ef, global_delta_ef
                source = "Global Explore"


        epoch_progress = (timestep % epoch_steps) / max(epoch_steps, 1)
        batch_candidates = []
        batch_keys = set()
        include_diversity = (
            n_ppo_batch >= 2
            and timestep % diversity_candidate_every == 0
        )
        for k in range(n_ppo_batch):
            candidate_source = source
            cand_mu_M, cand_delta_M = mu_M, delta_M
            cand_mu_ef, cand_delta_ef = mu_ef, delta_ef
            if include_diversity and k == n_ppo_batch - 1:
                cand_mu_M, cand_delta_M = global_mu_M, global_delta_M
                cand_mu_ef, cand_delta_ef = global_mu_ef, global_delta_ef
                candidate_source = "Diversity(Global)"

            for attempt in range(500):


                if attempt == 50:
                    cand_mu_M, cand_delta_M = global_mu_M, global_delta_M
                    cand_mu_ef, cand_delta_ef = global_mu_ef, global_delta_ef
                    candidate_source = f"{candidate_source}->GlobalRetry"
                state = build_state(
                    cand_mu_M, cand_delta_M, cand_mu_ef, cand_delta_ef, c_t,
                                    best_global_recall, best_global_qps, epoch_progress)
                action = ppo.select_action(state)
                M_raw = cand_mu_M + action[0] * cand_delta_M
                efC_raw = cand_mu_ef + action[1] * cand_delta_ef
                M = int(np.clip(M_raw, M_min, M_max))
                efC = int(np.clip(efC_raw, efC_min, efC_max))
                efC = max(efC, M)
                key = (M, efC)
                if key not in batch_keys and key not in evaluated_params:
                    break
                ppo.buffer.states.pop()
                ppo.buffer.actions.pop()
                ppo.buffer.logprobs.pop()
                ppo.buffer.state_values.pop()
            else:
                raise RuntimeError(
                    "Could not sample an unseen legal HNSW parameter after "
                    "500 attempts; refusing to spend a graph build on a duplicate"
                )
            batch_candidates.append((M, efC, action, candidate_source))
            batch_keys.add(key)
            source_stats[candidate_source] = (
                source_stats.get(candidate_source, 0) + 1
            )


        t_surr = time.time()
        surrogate_rewards = []
        surrogate_uncertainties = []
        surrogate_means = []
        boundary_penalties = []

        for M, efC, action, src in batch_candidates:
            mu_rw, sigma_rw = surrogate.predict_reward(
                M, efC, R_target, lambda_pen, kappa)
            penalty = action_boundary_penalty(action, boundary_beta)
            conservative_rw = mu_rw - lcb_beta * sigma_rw - penalty
            surrogate_means.append(float(mu_rw))
            surrogate_rewards.append(float(conservative_rw))
            surrogate_uncertainties.append(float(sigma_rw))
            boundary_penalties.append(float(penalty))

        total_surrogate_evals += n_ppo_batch
        stats['surrogate_eval'] += time.time() - t_surr

        for rw in surrogate_rewards:
            ppo.buffer.rewards.append(rw)
            ppo.buffer.is_terminals.append(True)


        progress = timestep / max(max_steps, 1)
        kappa_acq = kappa_acq_init + progress * (kappa_acq_final - kappa_acq_init)

        acq_scores = []
        for mu_rw, sigma_rw, penalty in zip(
                surrogate_means, surrogate_uncertainties, boundary_penalties):
            acq_scores.append(float(mu_rw + kappa_acq * sigma_rw - penalty))

        ranked = sorted(range(len(acq_scores)), key=lambda i: acq_scores[i], reverse=True)
        real_slots = min(n_real_per_step, n_total_budget - n_real_evals)
        selected_indices = ranked[:real_slots]
        real_eval_indices = set(selected_indices)

        print(f"\n--- Step {timestep}/{max_steps} | c_t={c_t:.2f} κ={kappa_acq:.2f} "
              f"action_std={ppo.action_std:.3f} | {source} ---")
        print(f"    Anchor: M={int(mu_M)}±{int(delta_M)}, efC={int(mu_ef)}±{int(delta_ef)}")

        llm_batch_rewards = []
        improved_this_step = False

        for idx, (M, efC, action, src) in enumerate(batch_candidates):
            in_real = idx in real_eval_indices
            print(f"    [{'REAL' if in_real else 'SURR':4s}] #{idx} M={M:3d} efC={efC:3d} | "
                  f"surr_μ={surrogate_rewards[idx]:.3f} σ={surrogate_uncertainties[idx]:.3f} "
                  f"acq={acq_scores[idx]:.3f}", end="")

            if in_real:
                t0 = time.time()
                survived, recall, qps, _, interp, raw_results = hnsw_env.build_and_evaluate(M, efC)
                dt_real = time.time() - t0
                eval_meta = dict(hnsw_env.last_evaluation or {})
                stats['real_eval'] += dt_real
                n_real_evals += 1
                cum_time_min = record_real_evaluation(
                    n_real_evals, "phase2_real", timestep, M, efC,
                    recall, qps, src, metrics=eval_meta)

                real_reward = compute_reward(float(recall), float(qps), R_target,
                                             lambda_pen, kappa, boundary_beta, action)
                surrogate.add_data(M, efC, recall, qps)
                evaluated_params.add((M, efC))
                best_global_recall = max(best_global_recall, float(recall))

                history_records.append({
                    'M': M, 'efC': efC, 'recall': float(recall),
                    'qps_at_target': float(qps), 'reward': float(real_reward),
                    'source': src,
                    'surrogate_mu': float(surrogate_rewards[idx]),
                    'surrogate_sigma': float(surrogate_uncertainties[idx]),
                    'time': dt_real, 'cum_time_min': cum_time_min,
                    'raw_results': list(raw_results),
                    'raw_results_full': list(
                        eval_meta.get('raw_results_full', [])
                    ),
                    'evaluation_metrics': eval_meta,
                    'phase': 'phase2_real',
                    'global_eval_index': n_real_evals,
                    'ppo_step': timestep,
                })


                save_param_result(M, efC, recall, qps, real_reward, interp, raw_results,
                                  None, dt_real, source=src, step=timestep, survived=survived,
                                  surrogate_mu=float(surrogate_rewards[idx]),
                                  surrogate_sigma=float(surrogate_uncertainties[idx]),
                                  acq_score=float(acq_scores[idx]),
                                  llm_anchor=(int(anchor_M), int(anchor_efC)) if llm_regions else None,
                                  action_std=ppo.action_std, trust=c_t,
                                  target_recall=R_target,
                                  cum_time_min=cum_time_min,
                                  full_results=eval_meta.get('raw_results_full'),
                                  evaluation_metrics=eval_meta,
                                  global_eval_index=n_real_evals,
                                  phase='phase2_real')


                step_log.append({
                    'step': timestep, 'epoch': epoch, 'source': src,
                    'M': M, 'efC': efC, 'real_recall': float(recall),
                    'real_qps': float(qps), 'real_reward': float(real_reward),
                    'surrogate_mu': float(surrogate_rewards[idx]),
                    'surrogate_sigma': float(surrogate_uncertainties[idx]),
                    'acq_score': float(acq_scores[idx]),
                    'anchor_M': int(mu_M), 'anchor_efC': int(mu_ef),
                    'delta_M': int(delta_M), 'delta_efC': int(delta_ef),
                    'action_std': ppo.action_std, 'trust': c_t,
                    'kappa_acq': kappa_acq,
                    'evaluation_time': dt_real,
                    'build_time': eval_meta.get('build_time_s'),
                    'search_time': eval_meta.get('search_time_s'),
                    'cum_time_min': cum_time_min,
                    'best_qps_so_far': best_global_qps,
                    'n_real_evals': n_real_evals,
                    'llm_M': active_region['mu_M'] if llm_regions else '',
                    'llm_efC': active_region['mu_ef'] if llm_regions else '',
                })

                if src.startswith("LLM-guided"):
                    llm_batch_rewards.append(float(real_reward))

                print(f" | real: recall={float(recall):.4f} QPS@R={R_target}={float(qps):.0f} "
                      f"reward={float(real_reward):.3f} cum_time={cum_time_min:.4f}min")

                if should_update_phase2_best(
                        n_real_evals,
                        n_warmup,
                        recall,
                        R_target,
                        qps,
                        best_global_qps):
                    best_global_qps = float(qps)
                    best_result = {
                        'M': M, 'efC': efC, 'recall': float(recall),
                        'qps_at_target': float(qps), 'interp': interp,
                        'step': timestep, 'source': src,
                        'cum_time_min': cum_time_min,
                        'raw_results': list(raw_results),
                        'raw_results_full': list(
                            eval_meta.get('raw_results_full', [])
                        ),
                        'evaluation_metrics': eval_meta,
                        'global_eval_index': n_real_evals,
                    }
                    improved_this_step = True
                    best_qps_per_region[active_region_idx] = max(
                        best_qps_per_region[active_region_idx], float(qps))
                    print(
                        f"    >>> NEW BEST: M={M}, efC={efC}, "
                        f"QPS@R={R_target}={float(qps):.0f}"
                    )

                step_log[-1]['best_qps_so_far'] = best_global_qps
                region_step_count[active_region_idx] += 1
            else:
                print()
                history_records.append({
                    'M': M, 'efC': efC, 'recall': None, 'qps_at_target': None,
                    'reward': float(surrogate_rewards[idx]),
                    'source': src + '(surr)',
                    'surrogate_mu': float(surrogate_rewards[idx]),
                    'surrogate_sigma': float(surrogate_uncertainties[idx]),
                    'phase': 'phase2_surrogate',
                    'ppo_step': timestep,
                })

                step_log.append({
                    'step': timestep, 'epoch': epoch, 'source': src + '(surr)',
                    'M': M, 'efC': efC, 'real_recall': None,
                    'real_qps': None, 'real_reward': None,
                    'surrogate_mu': float(surrogate_rewards[idx]),
                    'surrogate_sigma': float(surrogate_uncertainties[idx]),
                    'acq_score': float(acq_scores[idx]),
                    'anchor_M': int(mu_M), 'anchor_efC': int(mu_ef),
                    'delta_M': int(delta_M), 'delta_efC': int(delta_ef),
                    'action_std': ppo.action_std, 'trust': c_t,
                    'kappa_acq': kappa_acq, 'build_time': 0,
                    'best_qps_so_far': best_global_qps,
                    'n_real_evals': n_real_evals,
                    'llm_M': active_region['mu_M'] if llm_regions else '',
                    'llm_efC': active_region['mu_ef'] if llm_regions else '',
                })


        if len(ppo.buffer.rewards) >= update_timestep:
            t2 = time.time()
            ppo.update()
            stats['ppo_update'] += time.time() - t2


        if timestep > 0 and timestep % action_std_decay_freq == 0:
            old_std = ppo.action_std
            ppo.decay_action_std(action_std_decay_rate, min_action_std)
            if ppo.action_std != old_std:
                print(f"    [PPO] action_std: {old_std:.3f} → {ppo.action_std:.3f}")


        if llm_batch_rewards:
            llm_avg = np.mean(llm_batch_rewards)
            if ema_llm_reward == 0.0:
                ema_llm_reward = llm_avg
            else:
                ema_llm_reward = 0.9 * ema_llm_reward + 0.1 * llm_avg
            is_success = 1 if llm_avg >= ema_all_reward or ema_all_reward == 0.0 else 0
        elif source.startswith("LLM-guided"):
            is_success = 0
        else:
            is_success = 0.5

        alpha_trust = gamma_trust * alpha_trust + max(0, is_success)
        beta_trust = gamma_trust * beta_trust + (1 - max(0, is_success))

        real_rewards_step = [r['reward'] for r in history_records[-n_ppo_batch:]
                             if r.get('recall') is not None]
        if real_rewards_step:
            batch_avg = np.mean(real_rewards_step)
            if ema_all_reward == 0.0:
                ema_all_reward = batch_avg
            else:
                ema_all_reward = 0.9 * ema_all_reward + 0.1 * batch_avg


        if llm_regions and len(llm_regions) > 1:
            stagnation_counter = 0 if improved_this_step else stagnation_counter + 1

            if stagnation_counter >= stagnation_threshold:
                old_idx = active_region_idx
                active_region_idx = (
                    active_region_idx + 1
                ) % len(llm_regions)
                stagnation_counter = 0
                print(
                    f"    [Region] R{old_idx+1} stagnated "
                    f"({stagnation_threshold} steps no improvement) "
                    f"-> rotating to R{active_region_idx+1}: "
                    f"M={llm_regions[active_region_idx]['mu_M']}±"
                    f"{llm_regions[active_region_idx]['delta_M']}, "
                    f"efC={llm_regions[active_region_idx]['mu_ef']}±"
                    f"{llm_regions[active_region_idx]['delta_ef']}"
                )


        if timestep % surrogate_retrain_freq == 0 and surrogate.n_samples > n_warmup:
            t3 = time.time()
            surrogate.train(epochs=150, verbose=False)
            train_time = time.time() - t3
            stats['surrogate_train'] += train_time

            recent_real = [r for r in history_records[-20:] if r.get('recall') is not None]
            mae_val = None
            if recent_real:
                errors = []
                for r in recent_real[-5:]:
                    p_rec, _, _, _ = surrogate.predict(r['M'], r['efC'])
                    errors.append(abs(float(p_rec) - float(r['recall'])))
                mae_val = float(np.mean(errors))
                print(f"    [Surrogate] {surrogate.n_samples} samples, "
                      f"recall MAE: {mae_val:.4f}")

            surrogate_train_history.append({
                'step': timestep, 'n_samples': surrogate.n_samples,
                'recall_mae': mae_val, 'train_time': train_time,
            })

        if timestep % 5 == 0:
            print(f"    [Status] Real: {n_real_evals}/{n_total_budget} | "
                  f"Best QPS: {best_global_qps:.0f} | "
                  f"Trust: {c_t:.3f} | LLM calls: {llm_call_count}")

        timestep += 1

        if n_real_evals >= n_total_budget:
            print(f"\n>>> Budget exhausted ({n_real_evals} real evals)")
            break


    if n_real_evals != n_total_budget:
        raise RuntimeError(
            "Experiment ended without consuming the exact real-evaluation "
            f"budget: {n_real_evals} != {n_total_budget}"
        )
    total_time = time.time() - t_start
    final_trust = alpha_trust / (alpha_trust + beta_trust)
    llm_call_count = int(getattr(llm, "api_request_attempts", 0))

    print("\n" + "=" * 65)
    print("ACTuner Complete")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Real evaluations: {n_real_evals} / {n_total_budget} budget")
    print(f"Surrogate evaluations: {total_surrogate_evals}")
    print(f"LLM API calls: {llm_call_count}")
    print(f"Efficiency gain: {total_surrogate_evals/max(n_real_evals,1):.1f}x")
    print(f"Final action_std: {ppo.action_std:.3f}")
    print(f"Final trust: {final_trust:.3f}")
    print(f"Source distribution: {source_stats}")
    if best_result:
        print(f"Selected best: M={best_result['M']}, efC={best_result['efC']}  "
              f"recall={best_result['recall']:.4f}  QPS@R={R_target}={best_result['qps_at_target']:.1f}")
    print("=" * 65)


    if best_result:
        force_final_remeasure = bool(runtime_cfg.get("final_remeasure", False))
        final_mode = "remeasure" if force_final_remeasure else "reuse"
        print(
            "\n>>> Full evaluation (all 1000 queries, 12-point "
            f"curve; mode={final_mode})..."
        )
        final_eval_start = time.time()
        full_results = hnsw_env.full_evaluate(
            best_result['M'],
            best_result['efC'],
            force_remeasure=force_final_remeasure,
        )
        final_eval_s = time.time() - final_eval_start
        stats[
            'final_remeasure' if force_final_remeasure else 'final_curve_reuse'
        ] = final_eval_s
        print(f"\n  efSearch  |  recall@{hnsw_env.recall_k}  |  QPS")
        print("  " + "-" * 36)
        for efS, rec, qps in full_results:
            print(f"  {efS:8d}  |  {rec:.6f}   |  {qps:8.1f}")
        print("\n  Interpolated QPS at recall targets:")
        interp_best, final_qps, final_max_recall = hnsw_env.interpolate_results(
            full_results
        )
        for t, q in sorted(interp_best.items()):
            if q > 0.0:
                print(f"    R={t:.3f} -> QPS={q:.1f}")

        final_meta = dict(
            hnsw_env.get_evaluation(
                best_result['M'], best_result['efC']
            ) or {}
        )
        if force_final_remeasure:
            final_cum_time_min = record_real_evaluation(
                n_real_evals,
                "final_remeasure",
                "final",
                best_result['M'],
                best_result['efC'],
                float(final_max_recall),
                final_qps,
                "final_best_remeasure",
                metrics=final_meta,
                counts_toward_budget=False,
            )
            print(
                "  Final remeasurement cumulative time: "
                f"{final_cum_time_min:.4f}min (not counted as tuning step 31)"
            )
        else:
            final_cum_time_min = best_result.get('cum_time_min')
            print(
                "  Reused the selected configuration's measured curve."
            )
        final_survived = bool(
            final_qps > 0.0 and recall_target_reached(final_max_recall, R_target)
        )
        final_reward = compute_reward(
            final_max_recall, final_qps, R_target, lambda_pen, kappa,
            boundary_beta, np.zeros(2, dtype=np.float32),
        )
        best_result['selection_recall'] = best_result['recall']
        best_result['selection_qps_at_target'] = best_result['qps_at_target']
        best_result['final_validation_mode'] = final_mode
        best_result['final_remeasure_recall'] = final_max_recall
        best_result['final_remeasure_qps'] = final_qps
        best_result['final_remeasure_survived'] = final_survived
        best_result['final_remeasure_cum_time_min'] = final_cum_time_min
        if force_final_remeasure:
            best_result['final_remeasure_metrics'] = final_meta
            best_result['final_remeasure_eval_wall_time_s'] = final_eval_s
        print(
            f"  Final verified: reached={final_survived}, "
            f"max_recall={final_max_recall:.5f}, QPS@R={R_target}={final_qps:.1f}"
        )


        best_result['full_results'] = full_results
        save_param_result(best_result['M'], best_result['efC'],
                          final_max_recall, final_qps,
                          final_reward, interp_best, None, full_results,
                          final_eval_s, source=f'final_{final_mode}',
                          step=best_result.get('step', 'final'),
                          survived=final_survived, target_recall=R_target,
                          cum_time_min=final_cum_time_min,
                          evaluation_metrics=final_meta,
                          global_eval_index=best_result.get(
                              'global_eval_index'
                          ),
                          phase=f'final_{final_mode}')
        if final_survived:
            casebase.add_case(
                dataset_name, N=dataset_N, dim=dataset_dim,
                target_recall=R_target, best_M=best_result['M'],
                best_efC=best_result['efC'], best_qps=final_qps,
            )
        else:
            print(
                "  [WARN] Final curve missed the target; CaseBase was not updated."
            )


    total_time = time.time() - experiment_start


    os.makedirs(log_dir, exist_ok=True)
    llm_usage = llm.get_usage_summary() if hasattr(llm, "get_usage_summary") else {}
    usage_path = os.path.join(log_dir, "llm_usage.json")
    with open(usage_path, 'w') as f:
        json.dump(llm_usage, f, ensure_ascii=False, indent=2)
    if llm_usage.get('records'):
        usage_rows = []
        for r in llm_usage['records']:
            usage_rows.append({
                'Call': r.get('call', ''),
                'Has Usage': r.get('has_usage', False),
                'Prompt Chars': r.get('prompt_chars', 0),
                'Max Tokens Requested': r.get('max_tokens', 0),
                'Prompt Tokens': r.get('prompt_tokens', 0),
                'Completion Tokens': r.get('completion_tokens', 0),
                'Total Tokens': r.get('total_tokens', 0),
            })
        pd.DataFrame(usage_rows).to_excel(os.path.join(log_dir, "llm_usage.xlsx"), index=False)

    log_path = os.path.join(log_dir, "tuning_summary.txt")
    with open(log_path, 'w') as f:
        f.write(f"ACTuner Report\n")
        f.write(f"{'='*40}\n")
        f.write(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)\n")
        f.write(f"Real evaluations: {n_real_evals} / {n_total_budget} budget\n")
        f.write(f"Surrogate evaluations: {total_surrogate_evals}\n")
        f.write(f"LLM API calls: {llm_call_count}\n")
        f.write(f"LLM calls with token usage: {llm_usage.get('calls_with_usage', 0)}\n")
        f.write(f"LLM prompt chars: {llm_usage.get('prompt_chars', 0)}\n")
        f.write(f"LLM prompt tokens: {llm_usage.get('prompt_tokens', 0)}\n")
        f.write(f"LLM completion tokens: {llm_usage.get('completion_tokens', 0)}\n")
        f.write(f"LLM total tokens: {llm_usage.get('total_tokens', 0)}\n")
        f.write(f"LLM max tokens requested: {llm_usage.get('max_tokens_requested', 0)}\n")
        f.write(f"Efficiency gain: {total_surrogate_evals/max(n_real_evals,1):.1f}x\n")
        f.write(f"Final action_std: {ppo.action_std:.3f}\n")
        f.write(f"Final trust: {final_trust:.3f}\n")
        f.write(f"Source distribution: {source_stats}\n")
        if best_result:
            f.write(f"Selected best: M={best_result['M']}, efC={best_result['efC']}, "
                    f"recall={best_result['selection_recall']:.4f}, "
                    f"QPS={best_result['selection_qps_at_target']:.1f}\n")
            f.write(
                "Final remeasurement: "
                f"reached={best_result['final_remeasure_survived']}, "
                f"max_recall={best_result['final_remeasure_recall']:.4f}, "
                f"QPS={best_result['final_remeasure_qps']:.1f}\n"
            )
        f.write(f"\nTime breakdown:\n")
        for k, v in stats.items():
            f.write(f"  {k}: {v:.1f}s\n")
    print(f"\nLog saved to: {log_path}")


    saved_dir = save_aggregated_results(
        all_history=history_records,
        warmup_results=warmup_results,
        best_result=best_result,
        surrogate_train_history=surrogate_train_history,
        llm_suggestions=llm_suggestions,
        source_stats=source_stats,
        stats=stats,
        total_time=total_time,
        llm_call_count=llm_call_count,
        final_trust=final_trust,
        final_action_std=ppo.action_std,
        n_real_evals=n_real_evals,
        total_surrogate_evals=total_surrogate_evals,
        efSearch_train=efSearch_train,
        efSearch_full=efSearch_full,
        llm_usage=llm_usage,
        target_recall=R_target,
    )
    print(f"Aggregated Excel files saved to: {saved_dir}")


    if step_log:
        pd.DataFrame(step_log).to_excel(os.path.join(log_dir, "step_by_step_detail.xlsx"), index=False)
        print(f"Step-by-step detail saved to: {os.path.join(log_dir, 'step_by_step_detail.xlsx')}")
    print(f"Individual param Excel files saved to: {os.path.join(log_dir, 'per_param')}")

    return {
        'best': best_result,
        'n_real_evals': n_real_evals,
        'n_surrogate_evals': total_surrogate_evals,
        'efficiency_gain': total_surrogate_evals / max(n_real_evals, 1),
        'trust': final_trust,
        'time': total_time,
        'history': history_records,
        'llm_calls': llm_call_count,
        'llm_usage': llm_usage,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ACTuner: LLM + Surrogate + PPO')
    parser.add_argument('--llm-cache', type=str, default=None,
                        help='Path to LLM suggestion cache JSON.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed from config.yaml')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Log directory for results')
    parser.add_argument('--target-recall', type=str, default=None,
                        help='Override lrrl.R_target, for example 0.96 (not 096).')
    args = parser.parse_args()
    if args.log_dir is None:
        raw_target = args.target_recall or os.environ.get(
            "TARGET_RECALL", "0.99"
        )
        target_tag = f"{int(round(validate_target_recall(raw_target) * 100)):03d}"
        args.log_dir = os.path.join(
            os.path.dirname(__file__),
            "runs",
            time.strftime(f"actuner_R{target_tag}_%Y%m%d_%H%M%S"),
        )
    args.log_dir = str(project_path(args.log_dir).resolve())

    cache_path = str(project_path(args.llm_cache).resolve()) if args.llm_cache else None
    if cache_path is None:

        cache_path = os.path.join(args.log_dir, "llm_initial_regions.json")

    if os.path.exists(cache_path):
        print(f"[LLM Cache] Will replay initial regions from: {cache_path}")
    else:
        print(f"[LLM Cache] Fresh run — will save initial regions to: {cache_path}")

    result = train_coguided(
        llm_cache_path=cache_path, seed_override=args.seed, log_dir=args.log_dir,
        target_recall_override=args.target_recall)

    if args.log_dir:
        import shutil
        cfg_src = os.path.join(os.path.dirname(__file__), "config.yaml")
        if os.path.exists(cfg_src):
            _cfg_dst = os.path.join(args.log_dir, "config.yaml")
            shutil.copy(cfg_src, _cfg_dst)
            print(f"[Config] Saved to {_cfg_dst}")

    selected_best = result.get("best") if isinstance(result, dict) else None
    if not selected_best:
        print(
            "[FATAL] No feasible configuration survived the requested target; "
            "the run is marked failed.",
            file=sys.stderr,
        )
        raise SystemExit(3)
    if not selected_best.get("final_remeasure_survived", False):
        print(
            "[FATAL] Final validation did not reach the requested target; "
            "the run is marked failed.",
            file=sys.stderr,
        )
        raise SystemExit(4)
