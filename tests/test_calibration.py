from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from callverse.calibration.olist import load_olist
from callverse.calibration.policy import build_calibrated_policy
from callverse.calibration.profiles import (
    EmpiricalQuantiles,
    kaplan_meier,
    load_delivery_profile,
    load_support_profile,
)
from callverse.calibration.technion import TECHNION_FIELDS, load_technion
from callverse.scenarios import get_scenario
from callverse.simulation.engine import run_simulation


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]], delimiter: str = ",") -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


class CalibrationTests(unittest.TestCase):
    def test_technion_fixture_parses_and_reports_malformed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = sorted(TECHNION_FIELDS)
            base = {
                "vru+line": "1", "call_id": "10", "customer_id": "20", "priority": "0", "type": "PS",
                "date": "990101", "vru_entry": "08:00:00", "vru_exit": "08:00:10", "vru_time": "10",
                "q_start": "08:00:10", "q_exit": "08:01:10", "q_time": "60", "outcome": "AGENT",
                "ser_start": "08:01:10", "ser_exit": "08:03:10", "ser_time": "120", "server": "1",
            }
            _write_csv(root / "January1999.txt", fields, [base, dict(base, call_id="11", q_time="-1")], "\t")
            result = load_technion(root, require_full_year=False)
            self.assertEqual((result.profile.quality.raw_rows, result.profile.quality.usable_rows), (2, 1))
            self.assertEqual(result.profile.quality.exclusion_reasons["negative_duration"], 1)
            self.assertEqual(result.profile.service_time_minutes.median, 2)
            self.assertEqual((result.profile.patience_events, result.profile.patience_censored), (0, 1))

    def test_olist_features_and_review_join_remain_at_order_grain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            orders_path, reviews_path = root / "orders.csv", root / "reviews.csv"
            order_fields = ["order_id", "order_status", "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]
            _write_csv(orders_path, order_fields, [
                {"order_id": "early", "order_status": "delivered", "order_purchase_timestamp": "2018-01-01 00:00:00", "order_delivered_customer_date": "2018-01-09 00:00:00", "order_estimated_delivery_date": "2018-01-10 00:00:00"},
                {"order_id": "late", "order_status": "delivered", "order_purchase_timestamp": "2018-01-01 00:00:00", "order_delivered_customer_date": "2018-01-13 00:00:00", "order_estimated_delivery_date": "2018-01-10 00:00:00"},
                {"order_id": "missing", "order_status": "delivered", "order_purchase_timestamp": "2018-01-01 00:00:00", "order_delivered_customer_date": "", "order_estimated_delivery_date": "2018-01-10 00:00:00"},
            ])
            review_fields = ["review_id", "order_id", "review_score", "review_creation_date", "review_answer_timestamp"]
            _write_csv(reviews_path, review_fields, [
                {"review_id": "r1", "order_id": "early", "review_score": 5, "review_creation_date": "2018-01-10 00:00:00", "review_answer_timestamp": "2018-01-11 00:00:00"},
                {"review_id": "r2", "order_id": "late", "review_score": 2, "review_creation_date": "2018-01-14 00:00:00", "review_answer_timestamp": "2018-01-15 00:00:00"},
                {"review_id": "r3", "order_id": "late", "review_score": 1, "review_creation_date": "2018-01-14 00:00:00", "review_answer_timestamp": "2018-01-16 00:00:00"},
            ])
            profile = load_olist(orders_path, reviews_path).profile
            self.assertEqual((profile.eligible_delivered_orders, profile.late_orders), (2, 1))
            self.assertEqual(profile.late_delivery_rate, 0.5)
            self.assertEqual(profile.order_quality.exclusion_reasons["missing_required_timestamp"], 1)
            self.assertEqual(profile.review_quality.duplicate_rows, 1)
            self.assertEqual(profile.reviewed_eligible_orders, 2)
            self.assertEqual(profile.reviews_by_delivery_state["late"].score_counts["1"], 1)

    def test_quantile_contract_and_probability_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            EmpiricalQuantiles(probabilities=(0.0, 1.0), values=(2.0, 1.0))
        with self.assertRaises(ValidationError):
            EmpiricalQuantiles(probabilities=(-0.1, 1.0), values=(1.0, 2.0))

    def test_kaplan_meier_treats_served_wait_as_censored(self) -> None:
        points, quantiles = kaplan_meier([(1.0, True), (2.0, False)], (1.0, 2.0))
        self.assertEqual((points[0].events, points[1].censored), (1, 1))
        self.assertEqual(points[0].survival_probability, 0.5)
        self.assertEqual(list(quantiles.values), sorted(quantiles.values))

    def test_built_profile_loads_and_calibrated_simulation_is_reproducible(self) -> None:
        profile = load_support_profile(Path("data/processed/calibration/support_center_profile.json"))
        delivery = load_delivery_profile(Path("data/processed/calibration/delivery_profile.json"))
        self.assertEqual(delivery.eligible_delivered_orders, 96_470)
        policy = build_calibrated_policy(profile)
        scenario = get_scenario("normal_day")
        self.assertEqual(run_simulation(scenario, seed=77, policy=policy), run_simulation(scenario, seed=77, policy=policy))
        self.assertGreater(run_simulation(scenario, seed=77).counts.generated, 0)

    def test_heldout_validation_artifact_has_chronological_split(self) -> None:
        path = Path("data/processed/calibration/heldout_validation.json")
        result = json.loads(path.read_text(encoding="utf-8"))
        self.assertLess(result["split"]["calibration_end"], result["split"]["validation_start"])
        self.assertEqual(result["split"]["calibration_dates"], 290)
        self.assertEqual(result["split"]["validation_dates"], 73)
        self.assertEqual(result["service"]["selected_representation"], "empirical_quantiles")
        self.assertLess(result["arrival"]["normalized_slot_mae"], 0.1)


if __name__ == "__main__":
    unittest.main()
