from __future__ import annotations

from pairalign.finite_sample import (
    FiniteSampleRecord,
    analyze_bootstrap,
    analyze_finite_sample,
    summarize_bootstrap_values,
    summarize_values,
)


def test_analyze_finite_sample_overall_groups() -> None:
    records = [
        FiniteSampleRecord(
            pair_id=f"p{i}",
            model="model-a",
            dataset="dataset-a",
            subset="subset-a",
            scoring_rule="rule-a",
            win=1 if i < 6 else 0,
            delta=0.1 if i < 6 else -0.2,
        )
        for i in range(10)
    ]

    trials, summary = analyze_finite_sample(
        records,
        k_values=[5, 20],
        repeats=4,
        delta=0.05,
        seed=123,
        group_fields=("model", "dataset", "scoring_rule"),
    )

    assert len(trials) == 4
    assert len(summary) == 1
    row = summary[0]
    assert row["model"] == "model-a"
    assert row["dataset"] == "dataset-a"
    assert row["scoring_rule"] == "rule-a"
    assert row["K"] == 5
    assert row["full_K"] == 10
    assert row["full_A_hat"] == 0.6
    assert row["full_mu_hat"] == -0.02
    assert row["repeats"] == 4
    assert "mean_mu_hat" in row


def test_summarize_values_reports_hoeffding_radius() -> None:
    row = summarize_values(
        [0.4, 0.5, 0.6],
        [-0.1, 0.0, 0.1],
        k=100,
        full_k=1000,
        full_a_hat=0.5,
        full_mu_hat=0.0,
        delta=0.05,
        repeats=3,
    )

    assert row["K"] == 100
    assert row["full_K"] == 1000
    assert row["full_A_hat"] == 0.5
    assert row["full_mu_hat"] == 0.0
    assert row["mean_A_hat"] == 0.5
    assert row["mean_mu_hat"] == 0.0
    assert 0 < row["hoeffding_radius"] < 1
    assert row["hoeffding_coverage_vs_full"] == 1.0


def test_analyze_bootstrap_uses_full_group_size_with_replacement() -> None:
    records = [
        FiniteSampleRecord(
            pair_id=f"p{i}",
            model="model-a",
            dataset="dataset-a",
            subset="subset-a",
            scoring_rule="rule-a",
            win=1 if i < 6 else 0,
            delta=0.1 if i < 6 else -0.2,
        )
        for i in range(10)
    ]

    trials, summary = analyze_bootstrap(
        records,
        repeats=5,
        seed=123,
        group_fields=("model", "dataset", "scoring_rule"),
    )

    assert len(trials) == 5
    assert len(summary) == 1
    assert {trial["K"] for trial in trials} == {10}
    assert {trial["sampling"] for trial in trials} == {"with_replacement"}
    assert summary[0]["full_A_hat"] == 0.6
    assert summary[0]["full_mu_hat"] == -0.02
    assert summary[0]["K"] == 10
    assert "bootstrap_mean_mu_hat" in summary[0]


def test_summarize_bootstrap_values_reports_percentile_interval() -> None:
    row = summarize_bootstrap_values([0.4, 0.5, 0.6], [-0.1, 0.0, 0.1], full_k=100, full_a_hat=0.5, full_mu_hat=0.0, repeats=3)

    assert row["K"] == 100
    assert row["full_A_hat"] == 0.5
    assert row["full_mu_hat"] == 0.0
    assert row["bootstrap_mean_A_hat"] == 0.5
    assert row["bootstrap_bias"] == 0
    assert row["bootstrap_mean_mu_hat"] == 0.0
    assert row["bootstrap_mu_bias"] == 0.0
    assert row["bootstrap_ci_low_2p5"] < row["bootstrap_ci_high_97p5"]
    assert row["bootstrap_mu_ci_low_2p5"] < row["bootstrap_mu_ci_high_97p5"]
