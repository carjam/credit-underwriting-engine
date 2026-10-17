"""
Tests for feature guard - ensuring no target leakage.
"""
import pytest

from src.feature_guard import (
    check_for_leakage,
    filter_to_allowed_features,
    get_allowed_features,
    get_blocklist,
    validate_features_before_training,
)


class TestFeatureGuard:
    def test_get_allowed_features_without_pricing(self):
        """Application-time features should be returned."""
        features = get_allowed_features(include_pricing=False)
        
        # Should include application-time features
        assert "loan_amnt" in features
        assert "annual_inc" in features
        assert "fico_range_low" in features
        assert "dti" in features
        
        # Should NOT include pricing features when include_pricing=False
        assert "grade" not in features
        assert "sub_grade" not in features
        assert "int_rate" not in features
    
    def test_get_allowed_features_with_pricing(self):
        """Pricing features should be included when requested."""
        features = get_allowed_features(include_pricing=True)
        
        # Should include application-time features
        assert "loan_amnt" in features
        assert "annual_inc" in features
        
        # Should include pricing features when include_pricing=True
        assert "grade" in features
        assert "sub_grade" in features
        assert "int_rate" in features
    
    def test_get_blocklist(self):
        """Post-outcome features should be in blocklist."""
        blocklist = get_blocklist()
        
        # Payment history features
        assert "total_pymnt" in blocklist
        assert "total_rec_prncp" in blocklist
        assert "recoveries" in blocklist
        
        # Outstanding principal
        assert "out_prncp" in blocklist
        assert "out_prncp_inv" in blocklist
        
        # Post-application bureau data
        assert "last_fico_range_high" in blocklist
        assert "last_fico_range_low" in blocklist
        
        # Hardship features
        assert "hardship_flag" in blocklist
        assert "hardship_type" in blocklist
        
        # Settlement features
        assert "debt_settlement_flag" in blocklist
        assert "settlement_amount" in blocklist
    
    def test_check_for_leakage_clean(self):
        """No leakage should be detected for clean feature list."""
        clean_features = ["loan_amnt", "annual_inc", "fico_range_low", "dti"]
        result = check_for_leakage(clean_features)
        assert result["leaked_features"] == []
    
    def test_check_for_leakage_detected(self):
        """Leakage should be detected when post-outcome features are present."""
        leaky_features = [
            "loan_amnt",
            "annual_inc",
            "total_pymnt",  # POST-OUTCOME
            "recoveries",   # POST-OUTCOME
            "dti",
        ]
        result = check_for_leakage(leaky_features)
        leaked = result["leaked_features"]
        assert len(leaked) == 2
        assert "total_pymnt" in leaked
        assert "recoveries" in leaked
    
    def test_filter_to_allowed_features(self):
        """Should filter columns to only allowed features."""
        all_columns = [
            "loan_amnt",
            "annual_inc",
            "total_pymnt",  # should be filtered out
            "dti",
            "recoveries",   # should be filtered out
            "fico_range_low",
        ]
        
        filtered = filter_to_allowed_features(all_columns, include_pricing=False)
        
        assert "loan_amnt" in filtered
        assert "annual_inc" in filtered
        assert "dti" in filtered
        assert "fico_range_low" in filtered
        assert "total_pymnt" not in filtered
        assert "recoveries" not in filtered
    
    def test_validate_features_clean(self):
        """Validation should pass for clean features."""
        clean_features = ["loan_amnt", "annual_inc", "fico_range_low"]
        # Should not raise
        validate_features_before_training(clean_features, raise_on_leak=True)
    
    def test_validate_features_leaky(self):
        """Validation should raise for leaky features."""
        leaky_features = ["loan_amnt", "total_pymnt", "recoveries"]
        
        with pytest.raises(ValueError, match="TARGET LEAKAGE DETECTED"):
            validate_features_before_training(leaky_features, raise_on_leak=True)
    
    def test_validate_features_no_raise(self):
        """Validation should not raise when raise_on_leak=False."""
        leaky_features = ["loan_amnt", "total_pymnt", "recoveries"]
        # Should not raise
        validate_features_before_training(leaky_features, raise_on_leak=False)


class TestFeatureGuardCriticalLeakage:
    """Critical test cases for the most common leakage scenarios."""
    
    def test_all_payment_fields_blocked(self):
        """All payment-related fields should be blocked."""
        blocklist = set(get_blocklist())
        
        payment_fields = [
            "total_pymnt",
            "total_pymnt_inv",
            "total_rec_prncp",
            "total_rec_int",
            "total_rec_late_fee",
            "recoveries",
            "collection_recovery_fee",
        ]
        
        for field in payment_fields:
            assert field in blocklist, f"{field} should be in blocklist but is not"
    
    def test_outstanding_principal_blocked(self):
        """Outstanding principal fields should be blocked."""
        blocklist = set(get_blocklist())
        assert "out_prncp" in blocklist
        assert "out_prncp_inv" in blocklist
    
    def test_last_fico_blocked(self):
        """Last FICO scores (from bureau refresh) should be blocked."""
        blocklist = set(get_blocklist())
        assert "last_fico_range_high" in blocklist
        assert "last_fico_range_low" in blocklist
        
        # But original FICO should be allowed
        allowed = set(get_allowed_features(include_pricing=False))
        assert "fico_range_high" in allowed
        assert "fico_range_low" in allowed
    
    def test_hardship_fields_blocked(self):
        """All hardship fields should be blocked."""
        blocklist = set(get_blocklist())
        
        hardship_fields = [
            "hardship_flag",
            "hardship_type",
            "hardship_reason",
            "hardship_status",
            "orig_projected_additional_accrued_interest",
        ]
        
        for field in hardship_fields:
            assert field in blocklist, f"{field} should be in blocklist"
    
    def test_settlement_fields_blocked(self):
        """All settlement fields should be blocked."""
        blocklist = set(get_blocklist())
        
        settlement_fields = [
            "debt_settlement_flag",
            "settlement_status",
            "settlement_date",
            "settlement_amount",
        ]
        
        for field in settlement_fields:
            assert field in blocklist, f"{field} should be in blocklist"
