import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hnsw_backend import (
    EFSEARCH_GRID,
    SUPPORTED_TARGETS,
    count_vecs_records,
    interpolate_qps,
    normalize_efsearch_subset,
    read_performance_curve,
)


def _write_curve(path, rows):
    path.write_text(
        "".join(f"{ef_search},{recall},{qps}\n" for ef_search, recall, qps in rows),
        encoding="utf-8",
    )


def _write_fvecs(path, row_count, dim):
    record = struct.pack("<i", dim) + struct.pack(f"<{dim}f", *([0.0] * dim))
    with path.open("wb") as handle:
        for _ in range(row_count):
            handle.write(record)


def _valid_rows():
    recalls = (
        0.70,
        0.78,
        0.84,
        0.88,
        0.90,
        0.93,
        0.95,
        0.96,
        0.97,
        0.98,
        0.985,
        0.995,
    )
    qps = (
        12000.0,
        10000.0,
        8500.0,
        7200.0,
        6500.0,
        5000.0,
        4000.0,
        3300.0,
        2700.0,
        1800.0,
        300.0,
        200.0,
    )
    return list(zip(EFSEARCH_GRID, recalls, qps))


class HNSWBackendTests(unittest.TestCase):
    def test_fixed_12_point_curve_is_parsed_and_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            curve_path = Path(tmp) / "performance.csv"
            rows = _valid_rows()
            _write_curve(curve_path, rows)

            ef_search, recalls, qps, selected_rows, full_rows = (
                read_performance_curve(curve_path)
            )

        np.testing.assert_array_equal(ef_search, np.asarray(EFSEARCH_GRID, dtype=float))
        self.assertEqual(len(recalls), 12)
        self.assertEqual(len(qps), 12)
        self.assertEqual(selected_rows, rows)
        self.assertEqual(full_rows, rows)

    def test_fixed_12_point_curve_rejects_changed_binary_output(self):
        cases = {
            "missing_tail": _valid_rows()[:-1],
            "extra_row": _valid_rows() + [(4800, 0.999, 100.0)],
            "wrong_grid": [(16, 0.70, 12000.0)] + _valid_rows()[1:],
            "reordered": [_valid_rows()[1], _valid_rows()[0]] + _valid_rows()[2:],
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, rows in cases.items():
                with self.subTest(name=name):
                    curve_path = Path(tmp) / f"{name}.csv"
                    _write_curve(curve_path, rows)
                    with self.assertRaisesRegex(ValueError, "efSearch mismatch"):
                        read_performance_curve(curve_path)

    def test_curve_rejects_non_finite_or_invalid_measurements(self):
        cases = {
            "non_finite": (10, float("nan"), 12000.0),
            "bad_recall": (10, 1.1, 12000.0),
            "zero_qps": (10, 0.70, 0.0),
            "fractional_ef": (10.5, 0.70, 12000.0),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, replacement in cases.items():
                with self.subTest(name=name):
                    rows = _valid_rows()
                    rows[0] = replacement
                    curve_path = Path(tmp) / f"{name}.csv"
                    _write_curve(curve_path, rows)
                    with self.assertRaises(ValueError):
                        read_performance_curve(curve_path)

    def test_single_unique_recall_is_handled_without_crashing(self):
        rows = [
            (ef_search, 0.95, float(10000 - index * 100))
            for index, ef_search in enumerate(EFSEARCH_GRID)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            curve_path = Path(tmp) / "single_recall.csv"
            _write_curve(curve_path, rows)
            ef_search, recalls, qps, selected_rows, _ = (
                read_performance_curve(curve_path)
            )
            reached = interpolate_qps(0.94, ef_search, recalls, qps)
            unreached = interpolate_qps(0.99, ef_search, recalls, qps)

        self.assertEqual(len(recalls), 1)
        self.assertEqual(len(selected_rows), 12)
        self.assertTrue(reached["target_reached"])
        self.assertEqual(reached["qps_at_target"], 10000.0)
        self.assertEqual(reached["efSearch_interp"], 10.0)
        self.assertFalse(unreached["target_reached"])
        self.assertEqual(unreached["qps_at_target"], 0.0)
        self.assertEqual(unreached["efSearch_interp"], 0.0)

    def test_full_12_point_tail_supports_high_target_interpolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            curve_path = Path(tmp) / "high_target.csv"
            _write_curve(curve_path, _valid_rows())

            full_ef, full_recalls, full_qps, _, _ = read_performance_curve(
                curve_path, EFSEARCH_GRID
            )
            full_result = interpolate_qps(
                0.99, full_ef, full_recalls, full_qps
            )

            truncated = EFSEARCH_GRID[:10]
            short_ef, short_recalls, short_qps, _, _ = read_performance_curve(
                curve_path, truncated
            )
            short_result = interpolate_qps(
                0.99, short_ef, short_recalls, short_qps
            )

        self.assertTrue(full_result["target_reached"])
        self.assertAlmostEqual(full_result["qps_at_target"], 250.0)
        self.assertAlmostEqual(full_result["efSearch_interp"], 1800.0)
        self.assertFalse(short_result["target_reached"])
        self.assertEqual(short_result["qps_at_target"], 0.0)

    def test_vecs_metadata_reports_rows_and_dimension_without_payload_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            vecs_path = Path(tmp) / "small.fvecs"
            _write_fvecs(vecs_path, row_count=3, dim=4)

            self.assertEqual(count_vecs_records(vecs_path), (3, 4))

    def test_truncated_vecs_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            vecs_path = Path(tmp) / "truncated.fvecs"
            _write_fvecs(vecs_path, row_count=2, dim=4)
            with vecs_path.open("ab") as handle:
                handle.write(b"\x00")

            with self.assertRaisesRegex(ValueError, "Invalid vecs file size"):
                count_vecs_records(vecs_path)

    def test_subset_mismatch_is_rejected_before_evaluation(self):
        invalid_subsets = (
            [10, 20, 16],
            [20, 10],
            [10, 10],
            [],
        )
        for subset in invalid_subsets:
            with self.subTest(subset=subset):
                with self.assertRaises(ValueError):
                    normalize_efsearch_subset(subset)
        self.assertEqual(normalize_efsearch_subset([10]), (10,))

    def test_all_six_target_recalls_are_supported(self):
        expected = (0.90, 0.92, 0.94, 0.96, 0.98, 0.99)
        self.assertEqual(SUPPORTED_TARGETS, expected)

        ef_search = np.asarray(EFSEARCH_GRID, dtype=float)
        recalls = np.linspace(0.85, 0.995, len(EFSEARCH_GRID))
        qps = np.linspace(12000.0, 200.0, len(EFSEARCH_GRID))
        for target in expected:
            with self.subTest(target=target):
                result = interpolate_qps(target, ef_search, recalls, qps)
                self.assertTrue(result["target_reached"])
                self.assertGreater(result["qps_at_target"], 0.0)


if __name__ == "__main__":
    unittest.main()
