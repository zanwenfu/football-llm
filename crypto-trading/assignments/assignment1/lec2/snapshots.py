"""
Snapshot module for limit order book views.
Provides L1, L2, and L3 book snapshots.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple


@dataclass
class L1Snapshot:
    """
    Level 1 market data - top of book only.
    Shows best bid/ask price and quantity.
    """
    timestamp_ns: int
    best_bid_price: Optional[float] = None
    best_bid_qty: Optional[int] = None
    best_ask_price: Optional[float] = None
    best_ask_qty: Optional[int] = None
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate the bid-ask spread."""
        if self.best_bid_price is not None and self.best_ask_price is not None:
            return self.best_ask_price - self.best_bid_price
        return None
    
    @property
    def spread_bps(self) -> Optional[float]:
        """Calculate the spread in basis points."""
        if self.spread is not None and self.mid_price is not None and self.mid_price > 0:
            return (self.spread / self.mid_price) * 10000
        return None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate the mid price."""
        if self.best_bid_price is not None and self.best_ask_price is not None:
            return (self.best_bid_price + self.best_ask_price) / 2
        return None
    
    @property
    def micro_price(self) -> Optional[float]:
        """
        Calculate the volume-weighted mid price (micro price).
        Gives more weight to the side with more liquidity.
        """
        if (self.best_bid_price is not None and self.best_ask_price is not None and
            self.best_bid_qty is not None and self.best_ask_qty is not None and
            (self.best_bid_qty + self.best_ask_qty) > 0):
            total_qty = self.best_bid_qty + self.best_ask_qty
            return (self.best_bid_price * self.best_ask_qty + 
                   self.best_ask_price * self.best_bid_qty) / total_qty
        return None
    
    def __str__(self) -> str:
        bid = f"{self.best_bid_price}@{self.best_bid_qty}" if self.best_bid_price else "---"
        ask = f"{self.best_ask_price}@{self.best_ask_qty}" if self.best_ask_price else "---"
        spread_str = f"{self.spread:.4f}" if self.spread is not None else "N/A"
        return f"L1[{bid} | {ask}] spread={spread_str}"


@dataclass
class L2Snapshot:
    """
    Level 2 market data - aggregated price levels.
    Shows multiple price levels with total quantity at each level.
    """
    timestamp_ns: int
    bids: List[Tuple[float, int]]  # List of (price, qty) sorted descending by price
    asks: List[Tuple[float, int]]  # List of (price, qty) sorted ascending by price
    
    @property
    def best_bid(self) -> Optional[Tuple[float, int]]:
        """Get best bid (highest bid price)."""
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[Tuple[float, int]]:
        """Get best ask (lowest ask price)."""
        return self.asks[0] if self.asks else None
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate the bid-ask spread."""
        bid = self.best_bid
        ask = self.best_ask
        if bid and ask:
            return ask[0] - bid[0]
        return None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate the mid price."""
        bid = self.best_bid
        ask = self.best_ask
        if bid and ask:
            return (bid[0] + ask[0]) / 2
        return None
    
    def total_bid_volume(self, levels: Optional[int] = None) -> int:
        """Calculate total bid volume for specified levels (or all if None)."""
        bids_to_sum = self.bids[:levels] if levels else self.bids
        return sum(qty for _, qty in bids_to_sum)
    
    def total_ask_volume(self, levels: Optional[int] = None) -> int:
        """Calculate total ask volume for specified levels (or all if None)."""
        asks_to_sum = self.asks[:levels] if levels else self.asks
        return sum(qty for _, qty in asks_to_sum)
    
    def volume_imbalance(self, levels: int = 5) -> Optional[float]:
        """
        Calculate volume imbalance ratio for top N levels.
        Positive means more bid volume, negative means more ask volume.
        Returns value between -1 and 1.
        """
        bid_vol = self.total_bid_volume(levels)
        ask_vol = self.total_ask_volume(levels)
        total = bid_vol + ask_vol
        if total == 0:
            return None
        return (bid_vol - ask_vol) / total
    
    def __str__(self) -> str:
        lines = [f"L2 Snapshot @ {self.timestamp_ns}"]
        spread_str = f"{self.spread:.4f}" if self.spread is not None else "N/A"
        mid_str = f"{self.mid_price:.2f}" if self.mid_price is not None else "N/A"
        lines.append(f"  Spread: {spread_str}")
        lines.append(f"  Mid: {mid_str}")
        lines.append("  Bids:")
        for price, qty in self.bids[:5]:
            lines.append(f"    {price:.2f} x {qty}")
        lines.append("  Asks:")
        for price, qty in self.asks[:5]:
            lines.append(f"    {price:.2f} x {qty}")
        return "\n".join(lines)


@dataclass
class L3Snapshot:
    """
    Level 3 market data - full order-level detail.
    Shows individual orders with their IDs.
    """
    timestamp_ns: int
    bids: Dict[float, List[Dict[str, Any]]]  # price -> list of orders
    asks: Dict[float, List[Dict[str, Any]]]  # price -> list of orders
    
    def total_orders(self) -> int:
        """Count total number of orders in the book."""
        bid_count = sum(len(orders) for orders in self.bids.values())
        ask_count = sum(len(orders) for orders in self.asks.values())
        return bid_count + ask_count
    
    def total_bid_orders(self) -> int:
        """Count total number of bid orders."""
        return sum(len(orders) for orders in self.bids.values())
    
    def total_ask_orders(self) -> int:
        """Count total number of ask orders."""
        return sum(len(orders) for orders in self.asks.values())
    
    def to_l2(self) -> L2Snapshot:
        """Convert L3 to L2 by aggregating orders at each price level."""
        bids = []
        for price in sorted(self.bids.keys(), reverse=True):
            total_qty = sum(order['qty'] for order in self.bids[price])
            bids.append((price, total_qty))
        
        asks = []
        for price in sorted(self.asks.keys()):
            total_qty = sum(order['qty'] for order in self.asks[price])
            asks.append((price, total_qty))
        
        return L2Snapshot(timestamp_ns=self.timestamp_ns, bids=bids, asks=asks)
    
    def to_l1(self) -> L1Snapshot:
        """Convert L3 to L1 by taking best bid/ask."""
        l2 = self.to_l2()
        best_bid = l2.best_bid
        best_ask = l2.best_ask
        
        return L1Snapshot(
            timestamp_ns=self.timestamp_ns,
            best_bid_price=best_bid[0] if best_bid else None,
            best_bid_qty=best_bid[1] if best_bid else None,
            best_ask_price=best_ask[0] if best_ask else None,
            best_ask_qty=best_ask[1] if best_ask else None
        )
    
    def __str__(self) -> str:
        lines = [f"L3 Snapshot @ {self.timestamp_ns}"]
        lines.append(f"  Total orders: {self.total_orders()} ({self.total_bid_orders()} bids, {self.total_ask_orders()} asks)")
        
        # Show top 3 price levels for bids
        lines.append("  Bid orders (top 3 levels):")
        for price in sorted(self.bids.keys(), reverse=True)[:3]:
            orders = self.bids[price]
            lines.append(f"    {price:.2f}: {len(orders)} orders, {sum(o['qty'] for o in orders)} total qty")
        
        # Show top 3 price levels for asks
        lines.append("  Ask orders (top 3 levels):")
        for price in sorted(self.asks.keys())[:3]:
            orders = self.asks[price]
            lines.append(f"    {price:.2f}: {len(orders)} orders, {sum(o['qty'] for o in orders)} total qty")
        
        return "\n".join(lines)


class SnapshotManager:
    """
    Manages snapshot creation from the order book backend.
    """
    
    def __init__(self, backend):
        self.backend = backend
    
    def get_l1(self, timestamp_ns: int) -> L1Snapshot:
        """Get L1 (top-of-book) snapshot."""
        tob = self.backend.top_of_book()
        
        bid = tob.get('bid')
        ask = tob.get('ask')
        
        return L1Snapshot(
            timestamp_ns=timestamp_ns,
            best_bid_price=bid[0] if bid else None,
            best_bid_qty=bid[1] if bid else None,
            best_ask_price=ask[0] if ask else None,
            best_ask_qty=ask[1] if ask else None
        )
    
    def get_l2(self, timestamp_ns: int, levels: int = 10) -> L2Snapshot:
        """Get L2 (price-level aggregated) snapshot."""
        depth = self.backend.depth(levels=levels)
        
        return L2Snapshot(
            timestamp_ns=timestamp_ns,
            bids=depth.get('bids', []),
            asks=depth.get('asks', [])
        )
    
    def get_l3(self, timestamp_ns: int) -> L3Snapshot:
        """Get L3 (order-level) snapshot."""
        l3_data = self.backend.snapshot_l3()
        
        return L3Snapshot(
            timestamp_ns=timestamp_ns,
            bids=l3_data.get('bids', {}),
            asks=l3_data.get('asks', {})
        )
