"""
Metrics module for order book analysis.
Tracks and computes various market quality metrics.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from collections import deque
import statistics


@dataclass
class MarketMetrics:
    """Container for market quality metrics."""
    timestamp_ns: int
    
    # Spread metrics
    spread: Optional[float] = None
    spread_bps: Optional[float] = None  # Spread in basis points
    
    # Price metrics
    mid_price: Optional[float] = None
    micro_price: Optional[float] = None
    
    # Volume metrics
    bid_volume: int = 0
    ask_volume: int = 0
    total_volume: int = 0
    
    # Imbalance metrics
    volume_imbalance: Optional[float] = None  # -1 to 1
    order_imbalance: Optional[float] = None   # -1 to 1
    
    # Depth metrics
    bid_depth: int = 0  # Number of bid price levels
    ask_depth: int = 0  # Number of ask price levels
    
    # Order count
    total_orders: int = 0
    bid_orders: int = 0
    ask_orders: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'timestamp_ns': self.timestamp_ns,
            'spread': self.spread,
            'spread_bps': self.spread_bps,
            'mid_price': self.mid_price,
            'micro_price': self.micro_price,
            'bid_volume': self.bid_volume,
            'ask_volume': self.ask_volume,
            'total_volume': self.total_volume,
            'volume_imbalance': self.volume_imbalance,
            'order_imbalance': self.order_imbalance,
            'bid_depth': self.bid_depth,
            'ask_depth': self.ask_depth,
            'total_orders': self.total_orders,
            'bid_orders': self.bid_orders,
            'ask_orders': self.ask_orders
        }


@dataclass
class TradeMetrics:
    """Metrics related to executed trades."""
    timestamp_ns: int
    
    # Trade counts
    total_trades: int = 0
    buy_trades: int = 0  # Market buy (taker buy)
    sell_trades: int = 0  # Market sell (taker sell)
    
    # Volume
    total_volume: int = 0
    buy_volume: int = 0
    sell_volume: int = 0
    
    # Price metrics
    vwap: Optional[float] = None  # Volume-weighted average price
    last_price: Optional[float] = None
    
    # Slippage (for market orders)
    avg_slippage: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'timestamp_ns': self.timestamp_ns,
            'total_trades': self.total_trades,
            'buy_trades': self.buy_trades,
            'sell_trades': self.sell_trades,
            'total_volume': self.total_volume,
            'buy_volume': self.buy_volume,
            'sell_volume': self.sell_volume,
            'vwap': self.vwap,
            'last_price': self.last_price,
            'avg_slippage': self.avg_slippage
        }


class MetricsCollector:
    """
    Collects and computes metrics from order book snapshots and trades.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics collector.
        
        Args:
            window_size: Number of recent snapshots/trades to keep for rolling metrics
        """
        self.window_size = window_size
        
        # Historical data
        self.market_metrics_history: deque = deque(maxlen=window_size)
        self.trade_history: deque = deque(maxlen=window_size)
        
        # Current state
        self.total_trades = 0
        self.total_volume = 0
    
    def compute_market_metrics(self, l1_snapshot, l2_snapshot=None, l3_snapshot=None) -> MarketMetrics:
        """
        Compute market metrics from snapshots.
        
        Args:
            l1_snapshot: L1Snapshot object
            l2_snapshot: Optional L2Snapshot object for deeper metrics
            l3_snapshot: Optional L3Snapshot object for order-level metrics
        """
        metrics = MarketMetrics(timestamp_ns=l1_snapshot.timestamp_ns)
        
        # Basic spread metrics
        metrics.spread = l1_snapshot.spread
        if metrics.spread is not None and l1_snapshot.mid_price is not None and l1_snapshot.mid_price > 0:
            metrics.spread_bps = (metrics.spread / l1_snapshot.mid_price) * 10000
        
        # Price metrics
        metrics.mid_price = l1_snapshot.mid_price
        metrics.micro_price = l1_snapshot.micro_price
        
        # Volume from L1
        if l1_snapshot.best_bid_qty is not None:
            metrics.bid_volume = l1_snapshot.best_bid_qty
        if l1_snapshot.best_ask_qty is not None:
            metrics.ask_volume = l1_snapshot.best_ask_qty
        metrics.total_volume = metrics.bid_volume + metrics.ask_volume
        
        # Volume imbalance
        if metrics.total_volume > 0:
            metrics.volume_imbalance = (metrics.bid_volume - metrics.ask_volume) / metrics.total_volume
        
        # Additional metrics from L2
        if l2_snapshot:
            metrics.bid_depth = len(l2_snapshot.bids)
            metrics.ask_depth = len(l2_snapshot.asks)
            
            # More accurate volume from L2
            metrics.bid_volume = l2_snapshot.total_bid_volume()
            metrics.ask_volume = l2_snapshot.total_ask_volume()
            metrics.total_volume = metrics.bid_volume + metrics.ask_volume
            
            # Volume imbalance from L2
            metrics.volume_imbalance = l2_snapshot.volume_imbalance()
        
        # Order-level metrics from L3
        if l3_snapshot:
            metrics.total_orders = l3_snapshot.total_orders()
            metrics.bid_orders = l3_snapshot.total_bid_orders()
            metrics.ask_orders = l3_snapshot.total_ask_orders()
            
            # Order imbalance
            if metrics.total_orders > 0:
                metrics.order_imbalance = (metrics.bid_orders - metrics.ask_orders) / metrics.total_orders
        
        # Store in history
        self.market_metrics_history.append(metrics)
        
        return metrics
    
    def compute_trade_metrics(self, fills: List, reference_price: Optional[float] = None) -> TradeMetrics:
        """
        Compute trade metrics from a list of fills.
        
        Args:
            fills: List of Fill objects
            reference_price: Optional reference price for slippage calculation
        """
        timestamp_ns = fills[0].time_ns if fills else 0
        metrics = TradeMetrics(timestamp_ns=timestamp_ns)
        
        if not fills:
            return metrics
        
        # Count trades and volume by side
        total_notional = 0.0
        slippages = []
        
        for fill in fills:
            self.total_trades += 1
            metrics.total_trades += 1
            
            # Taker side determines if it's a buy or sell
            from wrapper import Side
            if fill.taker_side == Side.BUY:
                metrics.buy_trades += 1
                metrics.buy_volume += fill.qty
            else:
                metrics.sell_trades += 1
                metrics.sell_volume += fill.qty
            
            metrics.total_volume += fill.qty
            self.total_volume += fill.qty
            
            # Track notional for VWAP
            total_notional += fill.price * fill.qty
            
            # Track slippage if reference price provided
            if reference_price is not None:
                slippage = abs(fill.price - reference_price) / reference_price
                slippages.append(slippage)
            
            # Update last price
            metrics.last_price = fill.price
        
        # Calculate VWAP
        if metrics.total_volume > 0:
            metrics.vwap = total_notional / metrics.total_volume
        
        # Calculate average slippage
        if slippages:
            metrics.avg_slippage = statistics.mean(slippages)
        
        # Store in history
        self.trade_history.append(metrics)
        
        return metrics
    
    def get_rolling_spread(self, window: Optional[int] = None) -> Optional[float]:
        """Get average spread over rolling window."""
        if not self.market_metrics_history:
            return None
        
        window = window or len(self.market_metrics_history)
        recent = list(self.market_metrics_history)[-window:]
        spreads = [m.spread for m in recent if m.spread is not None]
        
        return statistics.mean(spreads) if spreads else None
    
    def get_rolling_volume_imbalance(self, window: Optional[int] = None) -> Optional[float]:
        """Get average volume imbalance over rolling window."""
        if not self.market_metrics_history:
            return None
        
        window = window or len(self.market_metrics_history)
        recent = list(self.market_metrics_history)[-window:]
        imbalances = [m.volume_imbalance for m in recent if m.volume_imbalance is not None]
        
        return statistics.mean(imbalances) if imbalances else None
    
    def get_trade_flow_imbalance(self, window: Optional[int] = None) -> Optional[float]:
        """
        Get trade flow imbalance over rolling window.
        Positive = more buying pressure, Negative = more selling pressure.
        """
        if not self.trade_history:
            return None
        
        window = window or len(self.trade_history)
        recent = list(self.trade_history)[-window:]
        
        total_buy = sum(m.buy_volume for m in recent)
        total_sell = sum(m.sell_volume for m in recent)
        total = total_buy + total_sell
        
        if total == 0:
            return None
        
        return (total_buy - total_sell) / total
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        latest_market = self.market_metrics_history[-1] if self.market_metrics_history else None
        latest_trade = self.trade_history[-1] if self.trade_history else None
        
        return {
            'total_trades': self.total_trades,
            'total_volume': self.total_volume,
            'latest_market_metrics': latest_market.to_dict() if latest_market else None,
            'latest_trade_metrics': latest_trade.to_dict() if latest_trade else None,
            'rolling_spread': self.get_rolling_spread(),
            'rolling_volume_imbalance': self.get_rolling_volume_imbalance(),
            'trade_flow_imbalance': self.get_trade_flow_imbalance(),
            'snapshots_collected': len(self.market_metrics_history),
            'trade_events_collected': len(self.trade_history)
        }
    
    def reset(self):
        """Reset all metrics."""
        self.market_metrics_history.clear()
        self.trade_history.clear()
        self.total_trades = 0
        self.total_volume = 0
