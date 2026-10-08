from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _visible_text(app: AppTest) -> str:
    elements = (
        *app.title,
        *app.header,
        *app.subheader,
        *app.markdown,
        *app.caption,
        *app.info,
        *app.success,
        *app.warning,
        *app.error,
    )
    return "\n".join(str(item.value) for item in elements)


def test_final_manager_journey_and_evidence_labels_are_explicit():
    app = AppTest.from_file("app.py", default_timeout=30).run(timeout=30)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Scenario Studio",
        "Twin Monitor",
        "Forecast",
        "Workforce",
        "Compare Decisions",
        "Interaction Lab",
        "Quality",
    ]
    visible = _visible_text(app)
    assert "1 Scenario" in visible and "7 Quality" in visible
    assert "SIMULATED evidence" in visible
    assert "HISTORICAL ML FORECAST" in visible
    assert "ANALYTICAL ERLANG-C BASELINE" in visible
    assert "SIMULATED same-seed comparison" in visible
    assert "EXPERIMENTAL PPO POLICY — NOT ADOPTED" in visible


def test_interaction_and_quality_boundaries_are_clear_without_live_provider(monkeypatch):
    monkeypatch.setattr("callverse.dashboard.manager.config.GROQ_API_KEY", None)
    app = AppTest.from_file("app.py", default_timeout=30).run(timeout=30)

    assert not app.exception
    visible = _visible_text(app)
    assert "Order and customer facts come from structured tools" in visible
    assert "Policies and procedures come from RAG" in visible
    assert "No fake live result will be substituted" in visible
    assert "deterministic layer enforces hard safety/compliance flags" in visible
    assert "LLM scores are not human ground truth" in visible


def test_default_normal_run_is_healthy_and_replay_wording_is_coherent():
    app = AppTest.from_file("app.py", default_timeout=30).run(timeout=30)
    next(button for button in app.button if button.label == "RUN DIGITAL TWIN").click()
    app.run(timeout=30)

    assert not app.exception
    visible = _visible_text(app)
    assert "CENTER STATUS · HEALTHY" in visible
    metrics = {metric.label: metric.value for metric in app.metric}
    assert "Current snapshot pressure" in metrics
    assert "Completed so far" in metrics
    assert "Abandoned so far" in metrics
    assert "cumulative abandonment can reflect earlier stress" in visible


def test_recommended_path_loads_controls_but_does_not_auto_run():
    app = AppTest.from_file("app.py", default_timeout=30).run(timeout=30)
    next(button for button in app.button if button.label == "LOAD RECOMMENDED DEMO").click()
    app.run(timeout=30)

    assert not app.exception
    assert app.selectbox(key="manager_preset").value == "staff_shortage"
    assert not any(metric.label == "Generated contacts" for metric in app.metric)
    assert "The model does not auto-run" in _visible_text(app)
