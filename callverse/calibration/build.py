"""Build compact calibration artifacts from locally downloaded official datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from callverse.scenarios import get_scenario
from callverse.simulation.engine import run_simulation
from callverse.simulation.policies import DEFAULT_POLICY

from .olist import load_olist
from .policy import build_calibrated_policy
from .profiles import CalibrationBundle
from .technion import load_technion
from .validation import validate_technion, write_validation


REPRESENTATIVE_SCENARIOS = ("normal_day", "rainy_peak", "staff_shortage", "perfect_storm")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: tuple[dict[str, object], ...] | list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(technion_dir: Path, orders: Path, reviews: Path, output_dir: Path) -> CalibrationBundle:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_technion(technion_dir)
    support = load_technion(technion_dir)
    delivery = load_olist(orders, reviews)
    bundle = CalibrationBundle(support=support.profile, delivery=delivery.profile)
    _write_json(output_dir / "support_center_profile.json", support.profile.model_dump(mode="json"))
    _write_json(output_dir / "delivery_profile.json", delivery.profile.model_dump(mode="json"))
    _write_csv(output_dir / "calls_by_date.csv", list(support.calls_by_date))
    _write_csv(output_dir / "arrival_profile.csv", list(support.arrival_profile))
    _write_csv(output_dir / "service_wait_statistics.csv", list(support.service_wait_summary))
    _write_csv(output_dir / "patience_survival.csv", list(support.abandonment_summary))
    _write_csv(output_dir / "delivery_lateness.csv", list(delivery.lateness_summary))
    _write_csv(output_dir / "review_lateness.csv", list(delivery.review_summary))
    write_validation(validation, output_dir)

    calibrated = build_calibrated_policy(support.profile)
    comparison: list[dict[str, object]] = []
    for name in REPRESENTATIVE_SCENARIOS:
        scenario = get_scenario(name)
        for policy_name, policy in (("prototype", DEFAULT_POLICY), ("calibrated", calibrated)):
            result = run_simulation(scenario, seed=42, policy=policy)
            comparison.append({
                "scenario": name,
                "policy": policy_name,
                "seed": 42,
                "generated_requests": result.counts.generated,
                "average_wait_minutes": result.kpis.average_waiting_time,
                "sla": result.kpis.sla,
                "abandonment_rate": result.kpis.abandonment_rate,
                "occupancy": result.kpis.occupancy,
                "average_handling_minutes": result.kpis.average_handling_time,
            })
    _write_csv(output_dir / "prototype_vs_calibrated.csv", comparison)
    print(f"Technion: {support.profile.quality.usable_rows}/{support.profile.quality.raw_rows} usable calls")
    print(f"Olist: {delivery.profile.eligible_delivered_orders}/{delivery.profile.order_quality.raw_rows} eligible orders")
    print(f"Artifacts: {output_dir}")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--technion-dir", type=Path, default=Path("data/raw/technion_anonymous_bank"))
    parser.add_argument("--orders", type=Path, default=Path("data/raw/olist/olist_orders_dataset.csv"))
    parser.add_argument("--reviews", type=Path, default=Path("data/raw/olist/olist_order_reviews_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/calibration"))
    args = parser.parse_args()
    build(args.technion_dir, args.orders, args.reviews, args.output_dir)


if __name__ == "__main__":
    main()
