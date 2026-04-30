"""
Main simulator module - ties everything together.
Provides replay/backtest harness for limit order book simulation.
"""

from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import time

from wrapper import OrderBookBackend, OrderRequest, Fill, Side, OrderType
from events import Event, NewLimitEvent, NewMarketEvent, CancelEvent, AmendEvent
from snapshots import SnapshotManager, L1Snapshot, L2Snapshot, L3Snapshot
from metrics import MetricsCollector


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    total_events: int = 0
    successful_events: int = 0
    failed_events: int = 0
    validation_errors: int = 0
    total_fills: int = 0
    total_volume: int = 0
    
    # Snapshots (optional)
    final_l1: Optional[L1Snapshot] = None
    final_l2: Optional[L2Snapshot] = None
    final_l3: Optional[L3Snapshot] = None
    
    # Metrics summary
    metrics_summary: Optional[Dict[str, Any]] = None
    
    # Event log
    event_log: List[Dict[str, Any]] = field(default_factory=list)
    
    # Fill log
    fill_log: List[Fill] = field(default_factory=list)
    
    def __str__(self) -> str:
        lines = [
            "=== Simulation Results ===",
            f"Events: {self.total_events} ({self.successful_events} success, {self.failed_events} failed, {self.validation_errors} validation errors)",
            f"Fills: {self.total_fills}",
            f"Total Volume: {self.total_volume}",
        ]
        
        if self.final_l1:
            lines.append(f"\nFinal L1: {self.final_l1}")
        
        if self.metrics_summary:
            lines.append(f"\nMetrics Summary:")
            for key, value in self.metrics_summary.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)


class LimitOrderBookSimulator:
    """
    Main simulator class that orchestrates the limit order book simulation.
    Handles event processing, snapshot generation, and metrics collection.
    """
    
    def __init__(self, tick_size: float = 0.01, collect_metrics: bool = True, 
                 record_snapshots: bool = False, snapshot_interval: Optional[int] = None):
        """
        Initialize the simulator.
        
        Args:
            tick_size: Minimum price increment
            collect_metrics: Whether to collect metrics during simulation
            record_snapshots: Whether to record snapshots at intervals
            snapshot_interval: Take snapshot every N events (if record_snapshots=True)
        """
        self.backend = OrderBookBackend(tick_size=tick_size)
        self.snapshot_mgr = SnapshotManager(self.backend)
        
        self.collect_metrics = collect_metrics
        self.metrics_collector = MetricsCollector() if collect_metrics else None
        
        self.record_snapshots = record_snapshots
        self.snapshot_interval = snapshot_interval or 100
        self.snapshots: List[L2Snapshot] = []
        
        # State
        self.current_time_ns = 0
        self.event_count = 0
        
        # Hooks for custom logic
        self.on_fill_callback: Optional[Callable[[Fill], None]] = None
        self.on_event_callback: Optional[Callable[[Event, bool], None]] = None
    
    def process_event(self, event: Event) -> tuple[bool, List[Fill], Optional[str]]:
        """
        Process a single event.
        
        Args:
            event: Event to process
        
        Returns:
            (success: bool, fills: List[Fill], error_msg: Optional[str])
        """
        # Validate the event
        error = event.validate()
        if error:
            return False, [], error
        
        # Update simulator time
        self.current_time_ns = event.timestamp_ns
        self.event_count += 1
        
        fills = []
        success = True
        error_msg = None
        
        try:
            if isinstance(event, NewLimitEvent):
                # Submit limit order
                req = OrderRequest(
                    order_id=event.order_id,
                    side=event.side,
                    qty=event.qty,
                    order_type=OrderType.LIMIT,
                    price=event.price,
                    time_ns=event.timestamp_ns
                )
                fills = self.backend.submit(req)
            
            elif isinstance(event, NewMarketEvent):
                # Submit market order
                req = OrderRequest(
                    order_id=event.order_id,
                    side=event.side,
                    qty=event.qty,
                    order_type=OrderType.MARKET,
                    time_ns=event.timestamp_ns
                )
                fills = self.backend.submit(req)
            
            elif isinstance(event, CancelEvent):
                # Cancel order
                success = self.backend.cancel(event.order_id)
                if not success:
                    error_msg = f"Order {event.order_id} not found"
            
            elif isinstance(event, AmendEvent):
                # Amend order
                fills = self.backend.amend(
                    event.order_id,
                    new_price=event.new_price,
                    new_qty=event.new_qty
                )
        
        except Exception as e:
            success = False
            error_msg = str(e)
        
        # Call fill callback
        if self.on_fill_callback and fills:
            for fill in fills:
                self.on_fill_callback(fill)
        
        # Call event callback
        if self.on_event_callback:
            self.on_event_callback(event, success)
        
        # Collect metrics if enabled
        if success and self.collect_metrics and fills:
            self.metrics_collector.compute_trade_metrics(fills)
        
        # Record snapshot if enabled
        if self.record_snapshots and self.event_count % self.snapshot_interval == 0:
            snapshot = self.snapshot_mgr.get_l2(self.current_time_ns)
            self.snapshots.append(snapshot)
            
            if self.collect_metrics:
                l1 = self.snapshot_mgr.get_l1(self.current_time_ns)
                self.metrics_collector.compute_market_metrics(l1, snapshot)
        
        return success, fills, error_msg
    
    def run(self, events: List[Event], verbose: bool = False) -> SimulationResult:
        """
        Run a simulation with a list of events.
        
        Args:
            events: List of events to process
            verbose: Print progress during simulation
        
        Returns:
            SimulationResult object with results
        """
        result = SimulationResult()
        result.total_events = len(events)
        
        start_time = time.time()
        
        for i, event in enumerate(events):
            if verbose and (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(events)} events...")
            
            success, fills, error_msg = self.process_event(event)
            
            # Track results
            if error_msg and not success:
                result.validation_errors += 1
            elif success:
                result.successful_events += 1
            else:
                result.failed_events += 1
            
            result.total_fills += len(fills)
            result.total_volume += sum(f.qty for f in fills)
            
            # Log event
            result.event_log.append({
                'event_num': i,
                'event_type': event.event_type.value,
                'timestamp_ns': event.timestamp_ns,
                'success': success,
                'fills': len(fills),
                'error': error_msg
            })
            
            # Log fills
            result.fill_log.extend(fills)
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"\nSimulation completed in {elapsed:.2f}s")
            print(f"Events/sec: {len(events) / elapsed:.0f}")
        
        # Capture final state
        result.final_l1 = self.snapshot_mgr.get_l1(self.current_time_ns)
        result.final_l2 = self.snapshot_mgr.get_l2(self.current_time_ns)
        
        # Add metrics summary
        if self.collect_metrics:
            result.metrics_summary = self.metrics_collector.get_summary()
        
        return result
    
    def get_l1_snapshot(self) -> L1Snapshot:
        """Get current L1 snapshot."""
        return self.snapshot_mgr.get_l1(self.current_time_ns)
    
    def get_l2_snapshot(self, levels: int = 10) -> L2Snapshot:
        """Get current L2 snapshot."""
        return self.snapshot_mgr.get_l2(self.current_time_ns, levels=levels)
    
    def get_l3_snapshot(self) -> L3Snapshot:
        """Get current L3 snapshot."""
        return self.snapshot_mgr.get_l3(self.current_time_ns)
    
    def reset(self):
        """Reset the simulator to initial state."""
        self.backend = OrderBookBackend(tick_size=self.backend.ob.tick_size)
        self.snapshot_mgr = SnapshotManager(self.backend)
        
        if self.metrics_collector:
            self.metrics_collector.reset()
        
        self.snapshots.clear()
        self.current_time_ns = 0
        self.event_count = 0


# Utility functions for creating events

def create_limit_order(order_id: str, side: str, price: float, qty: int, timestamp_ns: int) -> NewLimitEvent:
    """Helper to create a limit order event."""
    side_enum = Side.BUY if side.upper() in ['BUY', 'BID', 'B'] else Side.SELL
    return NewLimitEvent(order_id, side_enum, price, qty, timestamp_ns)


def create_market_order(order_id: str, side: str, qty: int, timestamp_ns: int) -> NewMarketEvent:
    """Helper to create a market order event."""
    side_enum = Side.BUY if side.upper() in ['BUY', 'BID', 'B'] else Side.SELL
    return NewMarketEvent(order_id, side_enum, qty, timestamp_ns)


def create_cancel(order_id: str, timestamp_ns: int) -> CancelEvent:
    """Helper to create a cancel event."""
    return CancelEvent(order_id, timestamp_ns)


def create_amend(order_id: str, timestamp_ns: int, new_price: Optional[float] = None, 
                 new_qty: Optional[int] = None) -> AmendEvent:
    """Helper to create an amend event."""
    return AmendEvent(order_id, timestamp_ns, new_price, new_qty)
