"""
Tests for strategies/helpers/risk_manager.py

No Freqtrade dependency required.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategies.helpers.risk_manager import RiskManager


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

DEFAULT_RISK_PARAMS = {
    "risk_per_trade_pct": 2.0,
    "max_daily_loss_pct": 3.0,
    "circuit_breaker": {
        "yellow_drawdown_pct": 5.0,
        "orange_drawdown_pct": 10.0,
        "red_drawdown_pct": 15.0,
    },
    "position_sizing": {
        "full_confluence_score": 4,
        "half_confluence_score": 3,
        "min_confluence_score": 3,
    },
}


def make_rm(portfolio_value: float = 1000.0) -> RiskManager:
    return RiskManager(portfolio_value=portfolio_value, risk_params=DEFAULT_RISK_PARAMS)


# -----------------------------------------------------------------------
# Position sizing
# -----------------------------------------------------------------------

class TestPositionSizing:
    def test_full_size_at_high_score(self):
        """Score >= 4 → full 2 % risk."""
        rm = make_rm(1000.0)
        # entry=100, stop=95 → stop_distance=5
        # risk_amount = 1000 * 0.02 = 20
        # units = 20 / 5 = 4
        size = rm.calculate_position_size(entry_price=100.0, stop_loss_price=95.0, confluence_score=4)
        assert size == pytest.approx(4.0, rel=1e-6)

    def test_half_size_at_score_3(self):
        """Score == 3 → half 2 % risk (1 %)."""
        rm = make_rm(1000.0)
        # risk_amount = 1000 * 0.02 * 0.5 = 10
        # units = 10 / 5 = 2
        size = rm.calculate_position_size(entry_price=100.0, stop_loss_price=95.0, confluence_score=3)
        assert size == pytest.approx(2.0, rel=1e-6)

    def test_zero_size_below_min_score(self):
        """Score < 3 → no trade."""
        rm = make_rm(1000.0)
        size = rm.calculate_position_size(entry_price=100.0, stop_loss_price=95.0, confluence_score=2)
        assert size == 0.0

    def test_score_5_same_as_4(self):
        """Score >= 4 all use full size."""
        rm = make_rm(1000.0)
        s4 = rm.calculate_position_size(100.0, 95.0, 4)
        s5 = rm.calculate_position_size(100.0, 95.0, 5)
        assert s4 == pytest.approx(s5, rel=1e-6)

    def test_portfolio_200_full_score(self):
        """Test with $200 dry-run wallet."""
        rm = make_rm(200.0)
        # risk = 200 * 0.02 = 4 USDT, stop_distance = 5
        # units = 4 / 5 = 0.8
        size = rm.calculate_position_size(entry_price=100.0, stop_loss_price=95.0, confluence_score=4)
        assert size == pytest.approx(0.8, rel=1e-6)

    def test_portfolio_200_half_score(self):
        """Test with $200 and score=3 (half size)."""
        rm = make_rm(200.0)
        # risk = 200 * 0.02 * 0.5 = 2, stop_distance = 5 → 0.4 units
        size = rm.calculate_position_size(entry_price=100.0, stop_loss_price=95.0, confluence_score=3)
        assert size == pytest.approx(0.4, rel=1e-6)

    def test_zero_stop_distance_returns_zero(self):
        """Entry == stop loss → divide by zero protection."""
        rm = make_rm(1000.0)
        size = rm.calculate_position_size(100.0, 100.0, 4)
        assert size == 0.0

    def test_position_capped_at_portfolio(self):
        """Position size should not exceed portfolio_value / entry_price."""
        rm = make_rm(100.0)
        # Very tight stop → very large theoretical size; must be capped
        size = rm.calculate_position_size(entry_price=50.0, stop_loss_price=49.99, confluence_score=4)
        max_allowed = 100.0 / 50.0  # 2 units
        assert size <= max_allowed


# -----------------------------------------------------------------------
# Circuit breaker
# -----------------------------------------------------------------------

class TestCircuitBreaker:
    def test_green_below_yellow(self):
        rm = make_rm()
        assert rm.check_circuit_breaker(0.049) == "green"

    def test_yellow_at_threshold(self):
        rm = make_rm()
        assert rm.check_circuit_breaker(0.05) == "yellow"

    def test_yellow_just_above_threshold(self):
        rm = make_rm()
        assert rm.check_circuit_breaker(0.051) == "yellow"

    def test_orange_at_threshold(self):
        rm = make_rm()
        assert rm.check_circuit_breaker(0.10) == "orange"

    def test_orange_just_below_red(self):
        rm = make_rm()
        assert rm.check_circuit_breaker(0.149) == "orange"

    def test_red_at_threshold(self):
        rm = make_rm()
        assert rm.check_circuit_breaker(0.15) == "red"

    def test_red_beyond_threshold(self):
        rm = make_rm()
        assert rm.check_circuit_breaker(0.30) == "red"

    def test_accepts_negative_drawdown(self):
        """Function should handle negative drawdown input (absolute value taken)."""
        rm = make_rm()
        assert rm.check_circuit_breaker(-0.10) == "orange"


# -----------------------------------------------------------------------
# Daily loss limit
# -----------------------------------------------------------------------

class TestDailyLossLimit:
    def test_below_limit_trading_allowed(self):
        rm = make_rm()
        assert rm.check_daily_loss_limit(0.029) is False

    def test_at_limit_halt_trading(self):
        rm = make_rm()
        assert rm.check_daily_loss_limit(0.03) is True

    def test_above_limit_halt_trading(self):
        rm = make_rm()
        assert rm.check_daily_loss_limit(0.05) is True

    def test_zero_loss_no_halt(self):
        rm = make_rm()
        assert rm.check_daily_loss_limit(0.0) is False


# -----------------------------------------------------------------------
# Stop distance calculation
# -----------------------------------------------------------------------

class TestStopDistance:
    def test_stop_distance_5pct(self):
        rm = make_rm()
        dist = rm.calculate_stop_distance(entry_price=100.0, stop_pct=0.05)
        assert dist == pytest.approx(5.0, rel=1e-6)

    def test_stop_distance_accepts_negative_pct(self):
        rm = make_rm()
        dist = rm.calculate_stop_distance(entry_price=100.0, stop_pct=-0.05)
        assert dist == pytest.approx(5.0, rel=1e-6)  # abs() applied


# -----------------------------------------------------------------------
# Portfolio update
# -----------------------------------------------------------------------

class TestPortfolioUpdate:
    def test_update_portfolio(self):
        rm = make_rm(1000.0)
        rm.update_portfolio_value(1200.0)
        assert rm.portfolio_value == pytest.approx(1200.0, rel=1e-6)

    def test_update_portfolio_invalid_raises(self):
        rm = make_rm(1000.0)
        with pytest.raises(ValueError):
            rm.update_portfolio_value(-100.0)

    def test_position_size_scales_with_portfolio(self):
        """Larger portfolio should produce proportionally larger position."""
        rm_small = make_rm(500.0)
        rm_large = make_rm(2000.0)
        s_small = rm_small.calculate_position_size(100.0, 95.0, 4)
        s_large = rm_large.calculate_position_size(100.0, 95.0, 4)
        assert s_large == pytest.approx(s_small * 4, rel=1e-6)


# -----------------------------------------------------------------------
# get_position_size_for_score
# -----------------------------------------------------------------------

class TestGetPositionSizeForScore:
    def test_basic_formula(self):
        rm = make_rm(1000.0)
        # risk_amount=20, stop_distance=5, entry=100 → 20/5 = 4 units
        result = rm.get_position_size_for_score(risk_amount=20.0, stop_distance=5.0, entry_price=100.0)
        assert result == pytest.approx(4.0, rel=1e-6)

    def test_zero_stop_returns_zero(self):
        rm = make_rm()
        result = rm.get_position_size_for_score(20.0, 0.0, 100.0)
        assert result == 0.0

    def test_capped_at_max_units(self):
        """Result should be capped at portfolio_value / entry_price."""
        rm = make_rm(100.0)
        # risk_amount=100, stop=0.01 → theoretical 10000 units; max = 100/50 = 2
        result = rm.get_position_size_for_score(100.0, 0.01, 50.0)
        assert result == pytest.approx(2.0, rel=1e-6)
