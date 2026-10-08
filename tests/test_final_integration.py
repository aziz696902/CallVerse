from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from callverse.scenarios import SCENARIO_PRESETS
from helppilot import config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_final_smoke_artifact_is_sanitized_and_bounded():
    report = json.loads(
        (PROJECT_ROOT / "data/processed/final/live_llm_smoke.json").read_text(
            encoding="utf-8"
        )
    )
    encoded = json.dumps(report).lower()
    assert report["summary"]["conceptual_cases"] == 6
    assert report["summary"]["successful_live_advisor_executions"] == 5
    assert report["summary"]["completed_live_quality_evaluations"] == 3
    assert report["secrets_stored"] is False
    assert "groq_api_key" not in encoded
    assert "gsk_" not in encoded


def test_final_demo_manifest_matches_all_declared_presets():
    report = json.loads(
        (PROJECT_ROOT / "data/processed/final/demo_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["all_seven_presets"]["status"] == "passed"
    assert report["all_seven_presets"]["seeds"] == [
        scenario.random_seed for scenario in SCENARIO_PRESETS
    ]
    assert set(report["official_scenarios"]) == {
        "normal_day",
        "staff_shortage",
        "perfect_storm",
    }
    assert report["same_seed_what_if"]["after_sla"] > report["same_seed_what_if"][
        "before_sla"
    ]


def test_final_documents_cover_architecture_results_demo_and_boundaries():
    architecture = (PROJECT_ROOT / "docs/CALLVERSE_ARCHITECTURE.md").read_text(encoding="utf-8")
    results = (PROJECT_ROOT / "docs/CALLVERSE_FINAL_RESULTS.md").read_text(encoding="utf-8")
    demo = (PROJECT_ROOT / "docs/CALLVERSE_DEMO_GUIDE.md").read_text(encoding="utf-8")
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Implemented versus deferred" in architecture
    assert "PPO is not adopted" in results
    assert all(name in demo for name in ("normal_day", "staff_shortage", "perfect_storm"))
    assert "Project implementation status: V1 complete" in readme


def test_app_without_groq_key_stays_available_and_labels_live_as_unavailable(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", None)
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception
    assert app.sidebar.radio[0].value == "📊 Manager Control Room"
    app.sidebar.radio[0].set_value("💬 Customer Interaction Demo")
    app.run(timeout=30)
    assert any("Live Groq is unavailable" in item.value for item in app.error)
    app.sidebar.radio[0].set_value("📊 Manager Control Room")
    app.run(timeout=30)
    assert not app.exception
    visible = "\n".join(item.value for item in (*app.info, *app.caption))
    assert "CallVerse V1 research decision-support prototype" in visible
    assert "HISTORICAL ML FORECAST" in visible
    assert "ANALYTICAL ERLANG-C BASELINE" in visible


def test_example_environment_file_contains_placeholders_only():
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "your_groq_api_key_here" in example
    assert "your_langsmith_api_key_here" in example
    assert "gsk_" not in example
    assert "lsv2_" not in example
