"""Tests for credit-specific evaluation metrics."""
import numpy as np
import pytest

from src.credit_metrics import (
    gini_coefficient,
    ks_statistic,
    pr_auc_minority_class,
    score_deciles,
    population_stability_index,
    credit_metrics_summary,
)


class TestCreditMetrics:
    """Test credit-specific metrics calculations."""
    
    def test_gini_coefficient_perfect_model(self):
        """Perfect model should have Gini = 1."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_proba = np.array([1.0, 0.9, 0.8, 0.3, 0.2, 0.1])
        gini = gini_coefficient(y_true, y_proba)
        assert gini == pytest.approx(1.0, abs=0.01)
    
    def test_gini_coefficient_random_model(self):
        """Random model should have Gini ≈ 0."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_proba = np.random.rand(100)
        gini = gini_coefficient(y_true, y_proba)
        assert -0.2 < gini < 0.2  # Allow some noise
    
    def test_gini_equals_2auc_minus_1(self):
        """Gini should equal 2*AUC - 1."""
        from sklearn.metrics import roc_auc_score
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_proba = np.array([0.9, 0.8, 0.4, 0.3, 0.7, 0.2])
        
        auc = roc_auc_score(y_true, y_proba)
        gini = gini_coefficient(y_true, y_proba)
        
        assert gini == pytest.approx(2 * auc - 1, abs=1e-10)
    
    def test_ks_statistic_perfect_separation(self):
        """Perfect separation should give KS = 1."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_proba = np.array([1.0, 0.9, 0.8, 0.3, 0.2, 0.1])
        ks = ks_statistic(y_true, y_proba)
        assert ks == pytest.approx(1.0, abs=0.01)
    
    def test_ks_statistic_no_separation(self):
        """No separation should give KS ≈ 0."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_proba = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ks = ks_statistic(y_true, y_proba)
        assert ks < 0.2  # Allow some numerical error
    
    def test_ks_statistic_range(self):
        """KS should be in [0, 1]."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_proba = np.random.rand(100)
        ks = ks_statistic(y_true, y_proba)
        assert 0 <= ks <= 1
    
    def test_pr_auc_minority_class(self):
        """PR-AUC for minority class should be > baseline for good model."""
        y_true = np.array([1, 1, 1, 1, 1, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2])
        
        pr_auc = pr_auc_minority_class(y_true, y_proba)
        baseline = (y_true == 0).mean()
        
        assert pr_auc > baseline  # Better than random
    
    def test_score_deciles_shape(self):
        """Decile table should have correct shape."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_proba = np.random.rand(200)
        
        deciles = score_deciles(y_true, y_proba, n_bins=10)
        
        assert len(deciles) == 10
        assert 'decile' in deciles.columns
        assert 'default_rate' in deciles.columns
        assert 'lift' in deciles.columns
        assert 'cum_default_capture' in deciles.columns
    
    def test_score_deciles_monotonic_scores(self):
        """Deciles should be ordered by score (descending)."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_proba = np.random.rand(200)
        
        deciles = score_deciles(y_true, y_proba, n_bins=10)
        
        # Decile 1 should have highest score_min
        assert deciles.loc[0, 'score_min'] > deciles.loc[9, 'score_min']
    
    def test_score_deciles_cum_capture_reaches_one(self):
        """Cumulative default capture should reach 1.0 in last decile."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_proba = np.random.rand(200)
        
        deciles = score_deciles(y_true, y_proba, n_bins=10)
        
        assert deciles.iloc[-1]['cum_default_capture'] == pytest.approx(1.0, abs=1e-10)
    
    def test_population_stability_index_identical(self):
        """PSI should be ~0 for identical distributions."""
        np.random.seed(42)
        baseline = np.random.rand(1000)
        current = baseline.copy()
        
        psi = population_stability_index(baseline, current)
        
        assert psi == pytest.approx(0.0, abs=0.01)
    
    def test_population_stability_index_shifted(self):
        """PSI should be > 0 for shifted distributions."""
        np.random.seed(42)
        baseline = np.random.rand(1000)
        current = baseline + 0.2  # Shift distribution
        
        psi = population_stability_index(baseline, current)
        
        assert psi > 0.1  # Significant shift
    
    def test_credit_metrics_summary_keys(self):
        """Summary should contain all expected metrics."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_proba = np.array([0.9, 0.8, 0.4, 0.3, 0.7, 0.2])
        
        summary = credit_metrics_summary(y_true, y_proba)
        
        assert 'roc_auc' in summary
        assert 'gini' in summary
        assert 'ks_statistic' in summary
        assert 'pr_auc_minority' in summary
        assert 'baseline_dr' in summary
    
    def test_credit_metrics_summary_values(self):
        """Summary values should be in valid ranges."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_proba = np.random.rand(100)
        
        summary = credit_metrics_summary(y_true, y_proba)
        
        assert 0 <= summary['roc_auc'] <= 1
        assert -1 <= summary['gini'] <= 1
        assert 0 <= summary['ks_statistic'] <= 1
        assert 0 <= summary['pr_auc_minority'] <= 1
        assert 0 <= summary['baseline_dr'] <= 1


class TestCreditMetricsRealisticScenarios:
    """Test credit metrics on realistic credit scoring scenarios."""
    
    def test_weak_model(self):
        """Weak model (AUC ~0.6) should have low-to-moderate Gini and KS."""
        np.random.seed(42)
        # Generate data where scores only weakly predict outcome
        n = 1000
        y_true = np.random.binomial(1, 0.8, n)  # 80% good
        # Add more noise to make it weak
        y_proba = 0.65 + 0.1 * y_true + 0.25 * np.random.rand(n)
        y_proba = np.clip(y_proba, 0, 1)
        
        gini = gini_coefficient(y_true, y_proba)
        ks = ks_statistic(y_true, y_proba)
        
        # Weak-to-moderate model
        assert 0.0 < gini < 0.8
        assert 0.0 < ks < 0.8
    
    def test_imbalanced_credit_data(self):
        """Test on imbalanced data (typical for credit: 80% good, 20% bad)."""
        np.random.seed(42)
        n = 1000
        # 80% good loans
        y_true = np.random.binomial(1, 0.8, n)
        # Model with moderate separation
        y_proba = np.where(y_true == 1, 
                          np.random.beta(8, 2, n),
                          np.random.beta(2, 8, n))
        
        summary = credit_metrics_summary(y_true, y_proba)
        
        # Should have reasonable discrimination
        assert summary['roc_auc'] > 0.7
        assert summary['gini'] > 0.4
        # Baseline should be ~20% (default rate)
        assert 0.15 < summary['baseline_dr'] < 0.25
