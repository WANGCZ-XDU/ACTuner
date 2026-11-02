import math
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

import lrrl
from test_server_preflight import _check_vecs
from tuner import CoGuidedLLMProcessor

from lrrl import (
    HNSWEnv,
    PPO,
    build_state,
    compute_reward,
    should_update_phase2_best,
    validate_target_recall,
)


class CoreLogicTests(unittest.TestCase):
    def test_target_recall_rejects_run_tags(self):
        self.assertEqual(validate_target_recall("0.96"), 0.96)
        for value in ("096", 96, 0, -0.1, 1.01, True, None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_target_recall(value)

    def test_unreached_target_has_zero_qps(self):
        env = object.__new__(HNSWEnv)
        env.R_target = 0.99
        raw = [(10, 0.80, 200000.0), (20, 0.90, 130000.0)]
        _, qps, max_recall = env.interpolate_results(raw)
        self.assertEqual(qps, 0.0)
        self.assertEqual(max_recall, 0.90)

    def test_target_boundary_tolerates_float_roundoff(self):
        env = object.__new__(HNSWEnv)
        env.R_target = 0.99
        raw = [(10, 0.90, 200000.0), (20, 0.99 - 5e-13, 130000.0)]
        _, qps, _ = env.interpolate_results(raw)
        self.assertEqual(qps, 130000.0)

    def test_gist_qps_state_is_informative(self):
        state = build_state(24, 8, 400, 100, 0.5, 0.99, 1250, 0.2)
        self.assertTrue(math.isclose(state[6], 1250 / 2000, rel_tol=1e-6))
        self.assertLess(state[6], 1.0)

    def test_boundary_penalty_changes_reward(self):
        base = compute_reward(
            0.99, 130104, 0.99, 10, 50, 5, np.array([0.0, 0.0])
        )
        outside = compute_reward(
            0.99, 130104, 0.99, 10, 50, 5, np.array([1.2, 0.0])
        )
        self.assertLess(outside, base)

    def test_first_phase2_result_requires_feasible_qps_improvement(self):
        for recall, qps, expected in (
            (0.80, 5000.0, False),
            (0.90, 2000.0, False),
            (0.90, 3500.0, False),
            (0.90, 4000.0, True),
        ):
            with self.subTest(recall=recall, qps=qps):
                self.assertEqual(
                    should_update_phase2_best(11, 10, recall, 0.90, qps, 3500.0),
                    expected,
                )

    def test_later_phase2_results_require_feasible_qps_improvement(self):
        self.assertFalse(
            should_update_phase2_best(
                12, 10, 0.89, 0.90, 5000.0, 1000.0
            )
        )
        self.assertFalse(
            should_update_phase2_best(
                12, 10, 0.90, 0.90, 900.0, 1000.0
            )
        )
        self.assertTrue(
            should_update_phase2_best(
                12, 10, 0.90, 0.90, 1100.0, 1000.0
            )
        )

    def test_ppo_update_supports_one_transition(self):
        ppo = PPO(8, 2, 3e-4, 2e-3, 0.99, 1, 0.2, 0.6)
        ppo.select_action(np.zeros(8, dtype=np.float32))
        ppo.buffer.rewards.append(1.0)
        ppo.buffer.is_terminals.append(True)
        ppo.update()
        self.assertFalse(ppo.buffer.rewards)
        self.assertTrue(all(parameter.isfinite().all() for parameter in ppo.policy.parameters()))


class PromptTests(unittest.TestCase):
    def test_initial_prompt_preserves_measured_recalls(self):
        processor = CoGuidedLLMProcessor(api_key="test-key")
        measurements = [
            dict(M=24, efC=300, recall=0.9900, qps_at_target=2000, reward=1),
            dict(M=32, efC=400, recall=0.9990, qps_at_target=1500, reward=0.9),
        ]
        for target in (0.90, 0.96, 0.99):
            with self.subTest(target=target), \
                    patch.object(processor, "_load_initial_regions", return_value=None), \
                    patch.object(processor, "_save_initial_regions"), \
                    patch.object(processor, "_call_api", return_value='{"regions": []}') as call, \
                    patch.object(processor, "_parse_regions", return_value=[]):
                processor.get_architect_initial(measurements, target)
                prompt = call.call_args.args[0]
                self.assertIn(f"Recall@10 >= {target}", prompt)
                self.assertIn("| 24 | 300 | 0.9900 |", prompt)
                self.assertIn("| 32 | 400 | 0.9990 |", prompt)
        processor.session.close()

    def test_reflection_preserves_history_best_and_predictions(self):
        processor = CoGuidedLLMProcessor(api_key="test-key")
        measured = dict(M=24, efC=300, recall=0.9900, qps_at_target=2000,
                        reward=1, source="real")
        hot_zones = {"top_points": [dict(M=32, efC=400, pred_recall=0.9990,
                                       pred_qps=1500, pred_reward=0.9, uncertainty=0.1)]}
        with patch.object(processor, "_call_api", return_value='{"regions": []}') as call, \
                patch.object(processor, "_parse_regions", return_value=[]):
            processor.get_architect_reflection([measured], measured, hot_zones,
                                               [measured], target_recall=0.90)
            prompt = call.call_args.args[0]
        self.assertIn("Recall@10 >= 0.9", prompt)
        self.assertIn("recall=0.9900", prompt)
        self.assertIn("recall@10=0.9900", prompt)
        self.assertIn("predicted recall=0.9990", prompt)
        processor.session.close()

    def test_inserted_text_is_not_rendered_as_another_template(self):
        evidence = "Measured 0.9900; literal ${target_recall}"
        prompt = CoGuidedLLMProcessor.INITIAL_PROMPT.substitute(target_recall=0.9, warmup_context=evidence)
        self.assertIn(evidence, prompt)


class RuntimePathTests(unittest.TestCase):
    def test_dataset_paths_resolve_from_config_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            record = struct.pack("<iff", 2, 0.0, 1.0)
            (data / "base.fvecs").write_bytes(record * 3)
            (data / "query.fvecs").write_bytes(record)
            (data / "groundtruth.ivecs").write_bytes(struct.pack("<ii", 1, 0))
            config = {"data": dict(base_path="data/base.fvecs", query_path="data/query.fvecs",
                                    groundtruth_path="data/groundtruth.ivecs", base_limit=3,
                                    query_limit=1, expected_dim=2, expected_gt_dim=1)}
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True), \
                    patch.object(lrrl, "CURRENT_CONFIG", {}), \
                    patch.object(lrrl, "CURRENT_DATA_PATHS", {}), \
                    patch.object(lrrl, "PROJECT_DIR", root):
                base, query, groundtruth = lrrl.load_gist1m(config_path)
                self.assertEqual(base.shape, (3, 2))
                self.assertEqual(query.shape, (1, 2))
                self.assertEqual(groundtruth.shape, (1, 1))
                self.assertEqual(Path(base.path), data / "base.fvecs")
                self.assertEqual(_check_vecs("data/base.fvecs", 2, 3, "base"), 3)

    def test_openmp_defaults_to_available_cpus_and_honors_override(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(lrrl, "available_cpu_count", return_value=8):
            self.assertEqual(lrrl.configure_openmp({"hnswlib_threads": "max"}), 8)
            os.environ["OMP_NUM_THREADS"] = "4"
            self.assertEqual(lrrl.configure_openmp({"hnswlib_threads": "max"}), 4)
            self.assertEqual(os.environ["OMP_DYNAMIC"], "FALSE")
            os.environ["OMP_NUM_THREADS"] = "9"
            with self.assertRaises(ValueError):
                lrrl.configure_openmp({"hnswlib_threads": "max"})


if __name__ == "__main__":
    unittest.main()
