# Limit Order Book Simulator

A robust Python-based limit order book simulator built on top of the `dyn4mik3/OrderBook` matching engine. This simulator provides a clean, production-ready API for backtesting trading strategies and analyzing market microstructure.

## Features

### 1. **Unified API Wrapper** ([wrapper.py](wrapper.py))
- Clean, type-safe interface abstracting the underlying orderbook package
- Standardized `Side`, `OrderType`, `OrderRequest`, and `Fill` data structures
- Backend-agnostic `LobBackend` interface for potential engine swaps
- `OrderBookBackend` implementation wrapping dyn4mik3/OrderBook

### 2. **Event Model** ([events.py](events.py))
- Structured event types: `NewLimitEvent`, `NewMarketEvent`, `CancelEvent`, `AmendEvent`
- Built-in validation for all events
- Timestamp tracking for replay consistency

### 3. **Fill/Trade Objects** ([wrapper.py](wrapper.py))
- Consistent trade ID generation
- Detailed fill information (taker/maker order IDs, side, price, quantity, timestamp)
- Trade history tracking

### 4. **Snapshots** ([snapshots.py](snapshots.py))
- **L1 (Top-of-Book)**: Best bid/ask with spread and mid-price calculations
- **L2 (Price-Level Depth)**: Multiple price levels with aggregated quantities
- **L3 (Order-Level)**: Full order book with individual order details
- Volume imbalance and micro-price calculations

### 5. **Replay/Backtest Harness** ([simulator.py](simulator.py))
- `LimitOrderBookSimulator` class for event-driven simulation
- Batch event processing with detailed results
- Custom callbacks for fills and events
- Snapshot recording at configurable intervals
- Reset capability for multiple runs

### 6. **Metrics** ([metrics.py](metrics.py))
- **Market metrics**: spread (absolute and bps), mid price, micro price, volume imbalance
- **Trade metrics**: VWAP, slippage, trade flow imbalance
- **Rolling statistics**: windowed averages for spread, imbalance, etc.
- Comprehensive metrics collector with history tracking

## Installation

1. Ensure you have the `orderbook` package installed:
```bash
pip install orderbook
```

2. Install additional dependencies (if needed):
```bash
pip install sortedcontainers
```

## Quick Start

```python
from simulator import LimitOrderBookSimulator, create_limit_order, create_market_order
from wrapper import Side

# Create simulator
sim = LimitOrderBookSimulator(tick_size=0.01, collect_metrics=True)

# Build order book
events = [
    create_limit_order("buy_1", "BUY", 100.00, 10, timestamp_ns=1000),
    create_limit_order("sell_1", "SELL", 100.50, 10, timestamp_ns=2000),
    create_market_order("mkt_1", "BUY", 5, timestamp_ns=3000),
]

# Run simulation
result = sim.run(events, verbose=True)

# View results
print(result)
l1 = sim.get_l1_snapshot()
print(l1)
```

## Usage Examples

See [example.py](example.py) for comprehensive examples including:
- Basic order flow
- Market order execution
- Order cancellation and amendment
- Metrics collection
- Stress testing with 10,000+ orders

Run examples:
```bash
python example.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 LimitOrderBookSimulator                 │
│  (Orchestrates everything, processes events)            │
└────────────┬───────────────────────────────────────────┘
             │
             ├─> Events (NewLimit, NewMarket, Cancel, Amend)
             │   └─> Validation
             │
             ├─> OrderBookBackend (wraps dyn4mik3/OrderBook)
             │   ├─> Submit orders
             │   ├─> Cancel/Amend orders
             │   └─> Generate fills
             │
             ├─> SnapshotManager
             │   ├─> L1Snapshot (top-of-book)
             │   ├─> L2Snapshot (price levels)
             │   └─> L3Snapshot (order details)
             │
             └─> MetricsCollector
                 ├─> MarketMetrics (spread, imbalance, etc.)
                 └─> TradeMetrics (VWAP, slippage, etc.)
```

## Core Components

### Events
```python
from events import NewLimitEvent, NewMarketEvent, CancelEvent, AmendEvent
from wrapper import Side

# Create a limit order
limit = NewLimitEvent(
    order_id="order_1",
    side=Side.BUY,
    price=100.00,
    qty=10,
    timestamp_ns=1000
)

# Validate
error = limit.validate()  # Returns None if valid, error message if not
```

### Snapshots
```python
# Get different levels of market data
l1 = sim.get_l1_snapshot()  # Best bid/ask
l2 = sim.get_l2_snapshot(levels=10)  # Top 10 levels
l3 = sim.get_l3_snapshot()  # Full order book

# L1 provides spread, mid, micro price
print(f"Spread: {l1.spread}")
print(f"Mid: {l1.mid_price}")

# L2 provides volume imbalance
print(f"Imbalance: {l2.volume_imbalance(levels=5)}")

# L3 can be converted to L2 or L1
l2_from_l3 = l3.to_l2()
l1_from_l3 = l3.to_l1()
```

### Metrics
```python
# Create simulator with metrics enabled
sim = LimitOrderBookSimulator(collect_metrics=True)

# Run simulation
result = sim.run(events)

# Get metrics summary
summary = result.metrics_summary
print(f"Total trades: {summary['total_trades']}")
print(f"Rolling spread: {summary['rolling_spread']}")
print(f"Volume imbalance: {summary['rolling_volume_imbalance']}")
```

## Simulation Results

The `SimulationResult` object provides:
- Event counts (total, successful, failed, validation errors)
- Fill counts and volume
- Final L1/L2/L3 snapshots
- Metrics summary
- Detailed event log
- Full fill log

```python
result = sim.run(events)
print(f"Success rate: {result.successful_events / result.total_events * 100:.1f}%")
print(f"Total fills: {result.total_fills}")
print(f"Total volume: {result.total_volume}")
```

## Advanced Features

### Custom Callbacks
```python
def on_fill(fill):
    print(f"Fill: {fill.qty} @ {fill.price}")

def on_event(event, success):
    print(f"Event {event.event_type}: {'✓' if success else '✗'}")

sim.on_fill_callback = on_fill
sim.on_event_callback = on_event
```

### Snapshot Recording
```python
# Record snapshots every 100 events
sim = LimitOrderBookSimulator(
    record_snapshots=True,
    snapshot_interval=100
)

result = sim.run(events)
# Access recorded snapshots
for snapshot in sim.snapshots:
    print(snapshot)
```

### Reset for Multiple Runs
```python
sim = LimitOrderBookSimulator()

# First run
result1 = sim.run(events_set_1)

# Reset and run again
sim.reset()
result2 = sim.run(events_set_2)
```

## Performance

The simulator can process 10,000+ events per second on typical hardware. The underlying `dyn4mik3/OrderBook` uses:
- Red-black trees for price-time priority
- `sortedcontainers` for efficient price level management
- Doubly-linked lists for orders at each price level

## Limitations & Future Enhancements

Current limitations:
- Single instrument only (no multi-asset support yet)
- No latency modeling
- Amend operations don't cross the spread (simple modify in place)

Potential enhancements (from the design doc):
- Multi-asset support
- Session clock for realistic time simulation
- Latency model (queue delays, network latency)
- Logging and persistence
- Enhanced slippage tracking
- Order queue position approximations

## File Structure

```
lec2/
├── wrapper.py         # Backend wrapper and core data structures
├── events.py          # Event model with validation
├── snapshots.py       # L1/L2/L3 snapshot classes
├── metrics.py         # Metrics collection and computation
├── simulator.py       # Main simulator harness
├── example.py         # Usage examples and demos
└── README.md          # This file
```

## License

This simulator is built on top of the open-source `dyn4mik3/OrderBook` package.

## Contributing

This is an educational project for understanding market microstructure and limit order book dynamics.
