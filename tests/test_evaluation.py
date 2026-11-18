from evaluation.reporter import StatisticsReporter

def test_statistics_reporter_returns_mean_median_std_and_ci():
    result = StatisticsReporter.summarize([1, 2, 3, 4])
    assert result["n"] == 4
    assert result["mean"] == 2.5
    assert result["median"] == 2.5
    assert result["std"] > 0
    assert result["ci95_low"] < result["mean"] < result["ci95_high"]
