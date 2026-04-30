"""
Example usage of the Limit Order Book Simulator.
Demonstrates basic functionality and features.
"""

from simulator import (
    LimitOrderBookSimulator, 
    create_limit_order, 
    create_market_order,
    create_cancel,
    create_amend
)
from wrapper import Side


def basic_example():
    """Basic example showing simple order flow."""
    print("=" * 60)
    print("BASIC EXAMPLE: Simple Order Flow")
    print("=" * 60)
    
    # Create simulator
    sim = LimitOrderBookSimulator(tick_size=0.01, collect_metrics=True)
    
    # Build order book
    events = [
        # Add some bids
        create_limit_order("buy_1", "BUY", 100.00, 10, timestamp_ns=1000),
        create_limit_order("buy_2", "BUY", 99.50, 15, timestamp_ns=2000),
        create_limit_order("buy_3", "BUY", 99.00, 20, timestamp_ns=3000),
        
        # Add some asks
        create_limit_order("sell_1", "SELL", 100.50, 10, timestamp_ns=4000),
        create_limit_order("sell_2", "SELL", 101.00, 15, timestamp_ns=5000),
        create_limit_order("sell_3", "SELL", 101.50, 20, timestamp_ns=6000),
    ]
    
    # Run simulation
    result = sim.run(events, verbose=True)
    
    # Show L1 snapshot
    print("\n" + "=" * 60)
    print("Order Book (L1):")
    print("=" * 60)
    l1 = sim.get_l1_snapshot()
    print(l1)
    
    # Show L2 snapshot
    print("\n" + "=" * 60)
    print("Order Book (L2):")
    print("=" * 60)
    l2 = sim.get_l2_snapshot(levels=5)
    print(l2)
    
    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)


def market_order_example():
    """Example showing market order execution and fills."""
    print("\n\n" + "=" * 60)
    print("MARKET ORDER EXAMPLE: Execution and Fills")
    print("=" * 60)
    
    # Create simulator
    sim = LimitOrderBookSimulator(tick_size=0.01, collect_metrics=True)
    
    events = [
        # Build the book
        create_limit_order("bid_1", "BUY", 100.00, 10, timestamp_ns=1000),
        create_limit_order("bid_2", "BUY", 99.50, 20, timestamp_ns=2000),
        create_limit_order("ask_1", "SELL", 100.50, 10, timestamp_ns=3000),
        create_limit_order("ask_2", "SELL", 101.00, 20, timestamp_ns=4000),
        
        # Execute a market buy that sweeps the best ask
        create_market_order("mkt_buy_1", "BUY", 5, timestamp_ns=5000),
        
        # Execute a market sell that sweeps the best bid
        create_market_order("mkt_sell_1", "SELL", 7, timestamp_ns=6000),
    ]
    
    # Run simulation
    result = sim.run(events, verbose=False)
    
    # Show fills
    print("\nFills:")
    for i, fill in enumerate(result.fill_log, 1):
        print(f"  {i}. {fill.trade_id}: {fill.qty} @ {fill.price:.2f} "
              f"(taker: {fill.taker_order_id} {fill.taker_side.value})")
    
    # Show final state
    print("\nFinal L1:")
    print(sim.get_l1_snapshot())
    
    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)


def cancel_amend_example():
    """Example showing cancel and amend operations."""
    print("\n\n" + "=" * 60)
    print("CANCEL/AMEND EXAMPLE: Order Management")
    print("=" * 60)
    
    sim = LimitOrderBookSimulator(tick_size=0.01, collect_metrics=True)
    
    events = [
        # Build the book
        create_limit_order("bid_1", "BUY", 100.00, 10, timestamp_ns=1000),
        create_limit_order("bid_2", "BUY", 99.50, 20, timestamp_ns=2000),
        create_limit_order("ask_1", "SELL", 100.50, 10, timestamp_ns=3000),
        create_limit_order("ask_2", "SELL", 101.00, 20, timestamp_ns=4000),
    ]
    
    print("\nInitial book:")
    sim.run(events[:4])
    print(sim.get_l1_snapshot())
    
    # Cancel an order
    print("\nCanceling bid_2...")
    sim.process_event(create_cancel("bid_2", timestamp_ns=5000))
    print(sim.get_l1_snapshot())
    
    # Amend an order (change qty)
    print("\nAmending bid_1 quantity to 25...")
    sim.process_event(create_amend("bid_1", timestamp_ns=6000, new_qty=25))
    l2 = sim.get_l2_snapshot()
    print(f"Best bid: {l2.best_bid}")
    
    # Amend an order (change price)
    print("\nAmending bid_1 price to 100.25...")
    sim.process_event(create_amend("bid_1", timestamp_ns=7000, new_price=100.25))
    print(sim.get_l1_snapshot())


def metrics_example():
    """Example showing metrics collection."""
    print("\n\n" + "=" * 60)
    print("METRICS EXAMPLE: Market Quality Metrics")
    print("=" * 60)
    
    sim = LimitOrderBookSimulator(tick_size=0.01, collect_metrics=True, 
                                   record_snapshots=True, snapshot_interval=2)
    
    events = [
        # Build initial book
        create_limit_order("b1", "BUY", 100.00, 100, timestamp_ns=1000),
        create_limit_order("b2", "BUY", 99.50, 150, timestamp_ns=2000),
        create_limit_order("a1", "SELL", 100.50, 100, timestamp_ns=3000),
        create_limit_order("a2", "SELL", 101.00, 150, timestamp_ns=4000),
        
        # Execute some trades
        create_market_order("m1", "BUY", 50, timestamp_ns=5000),
        create_market_order("m2", "SELL", 30, timestamp_ns=6000),
        
        # Add more liquidity
        create_limit_order("b3", "BUY", 99.00, 200, timestamp_ns=7000),
        create_limit_order("a3", "SELL", 101.50, 200, timestamp_ns=8000),
    ]
    
    result = sim.run(events, verbose=False)
    
    # Show metrics
    print("\nMetrics Summary:")
    if result.metrics_summary:
        for key, value in result.metrics_summary.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            elif value is not None:
                print(f"  {key}: {value}")
    
    # Show L2 with imbalance
    print("\nFinal L2 Snapshot:")
    l2 = sim.get_l2_snapshot(levels=5)
    print(f"  Best bid: {l2.best_bid}")
    print(f"  Best ask: {l2.best_ask}")
    if l2.spread is not None:
        print(f"  Spread: {l2.spread:.2f}")
        print(f"  Mid: {l2.mid_price:.2f}")
    imb = l2.volume_imbalance(5)
    imb_str = f"{imb:.3f}" if imb is not None else "N/A"
    print(f"  Volume imbalance (top 5): {imb_str}")


def stress_test_example():
    """Stress test with many orders."""
    print("\n\n" + "=" * 60)
    print("STRESS TEST: Processing 10,000 Orders")
    print("=" * 60)
    
    sim = LimitOrderBookSimulator(tick_size=0.01, collect_metrics=True)
    
    # Generate many orders
    events = []
    base_time = 1000
    
    # Add 5000 bids around 100.00
    for i in range(5000):
        price = 100.00 - (i % 100) * 0.01
        qty = 10 + (i % 50)
        events.append(create_limit_order(f"bid_{i}", "BUY", price, qty, 
                                        timestamp_ns=base_time + i))
    
    # Add 5000 asks around 100.50
    for i in range(5000):
        price = 100.50 + (i % 100) * 0.01
        qty = 10 + (i % 50)
        events.append(create_limit_order(f"ask_{i}", "SELL", price, qty, 
                                        timestamp_ns=base_time + 5000 + i))
    
    # Run simulation
    import time
    start = time.time()
    result = sim.run(events, verbose=True)
    elapsed = time.time() - start
    
    print(f"\nProcessed {len(events)} events in {elapsed:.2f}s")
    print(f"Throughput: {len(events)/elapsed:.0f} events/sec")
    
    # Show final state
    l1 = sim.get_l1_snapshot()
    spread_str = f"{l1.spread:.4f}" if l1.spread is not None else "N/A"
    mid_str = f"{l1.mid_price:.2f}" if l1.mid_price is not None else "N/A"
    print(f"\nFinal spread: {spread_str}")
    print(f"Final mid: {mid_str}")
    
    l2 = sim.get_l2_snapshot(levels=10)
    print(f"Bid depth: {len(l2.bids)} levels")
    print(f"Ask depth: {len(l2.asks)} levels")


def main():
    """Run all examples."""
    basic_example()
    market_order_example()
    cancel_amend_example()
    metrics_example()
    stress_test_example()
    
    print("\n\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
