"""
Risk manager for CryptoEdge – 2% fixed-fractional position sizing
with circuit-breaker and daily-loss-limit enforcement.
"""
from __future__ import annotations


class RiskManager:
    """
    Manages position sizing and risk controls.

    Parameters
    ----------
    portfolio_value : float
        Current total portfolio value in stake currency (e.g. USDT).
    risk_params : dict
        Risk parameters loaded from config/risk_params.json.
    """

    def __init__(self, portfolio_value: float, risk_params: dict) -> None:
        self.portfolio_value = portfolio_value
        self.risk_params = risk_params

        # Extract commonly used params with sensible defaults
        self._risk_pct = risk_params.get("risk_per_trade_pct", 2.0) / 100.0
        self._max_daily_loss_pct = risk_params.get("max_daily_loss_pct", 3.0) / 100.0

        cb = risk_params.get("circuit_breaker", {})
        self._yellow_dd = cb.get("yellow_drawdown_pct", 5.0) / 100.0
        self._orange_dd = cb.get("orange_drawdown_pct", 10.0) / 100.0
        self._red_dd = cb.get("red_drawdown_pct", 15.0) / 100.0

        ps = risk_params.get("position_sizing", {})
        self._full_score = ps.get("full_confluence_score", 4)
        self._half_score = ps.get("half_confluence_score", 3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        confluence_score: int,
    ) -> float:
        """
        Calculate position size in base-currency units.

        Uses 2% fixed-fractional risk.  Full size when confluence_score >= 4,
        half size when confluence_score == 3.  Returns 0 if score < 3.

        Parameters
        ----------
        entry_price : float
            Planned entry price.
        stop_loss_price : float
            Hard stop-loss price.
        confluence_score : int
            Signal quality score from confluence_score().

        Returns
        -------
        float
            Number of base-currency units to buy (can be 0).
        """
        if confluence_score < self._half_score:
            return 0.0

        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance == 0 or entry_price == 0:
            return 0.0

        risk_amount = self.portfolio_value * self._risk_pct

        # Scale by confluence
        if confluence_score >= self._full_score:
            size = self.get_position_size_for_score(risk_amount, stop_distance, entry_price)
        else:
            # Half size for minimum confluence
            size = self.get_position_size_for_score(risk_amount * 0.5, stop_distance, entry_price)

        return max(size, 0.0)

    def check_circuit_breaker(self, current_drawdown_pct: float) -> str:
        """
        Determine the circuit-breaker level based on current drawdown.

        Parameters
        ----------
        current_drawdown_pct : float
            Current drawdown as a positive fraction (e.g. 0.05 = 5 %).

        Returns
        -------
        str
            One of "green", "yellow", "orange", "red".
        """
        dd = abs(current_drawdown_pct)
        if dd >= self._red_dd:
            return "red"
        if dd >= self._orange_dd:
            return "orange"
        if dd >= self._yellow_dd:
            return "yellow"
        return "green"

    def check_daily_loss_limit(self, daily_loss_pct: float) -> bool:
        """
        Check whether the daily loss limit has been breached.

        Parameters
        ----------
        daily_loss_pct : float
            Today's realised loss as a positive fraction (e.g. 0.03 = 3 %).

        Returns
        -------
        bool
            True if trading should be **halted** (limit breached),
            False if trading may continue.
        """
        return abs(daily_loss_pct) >= self._max_daily_loss_pct

    def calculate_stop_distance(self, entry_price: float, stop_pct: float) -> float:
        """
        Calculate stop distance in price units.

        Parameters
        ----------
        entry_price : float
        stop_pct : float
            Stop percentage as a positive fraction (e.g. 0.05 for 5 %).

        Returns
        -------
        float
            Price distance from entry to stop.
        """
        return entry_price * abs(stop_pct)

    def update_portfolio_value(self, new_value: float) -> None:
        """Update the portfolio value used for sizing calculations."""
        if new_value <= 0:
            raise ValueError(f"Portfolio value must be positive, got {new_value}")
        self.portfolio_value = new_value

    def get_position_size_for_score(
        self, risk_amount: float, stop_distance: float, entry_price: float
    ) -> float:
        """
        Core position sizing formula.

        position_size (units) = risk_amount / stop_distance

        Parameters
        ----------
        risk_amount : float
            Dollar amount at risk on this trade.
        stop_distance : float
            Price distance between entry and stop-loss.
        entry_price : float
            Entry price (used to cap size at portfolio value).

        Returns
        -------
        float
            Number of base-currency units.
        """
        if stop_distance <= 0 or entry_price <= 0:
            return 0.0
        units = risk_amount / stop_distance
        # Cap at total portfolio value / entry_price (can't buy more than we own)
        max_units = self.portfolio_value / entry_price
        return min(units, max_units)
