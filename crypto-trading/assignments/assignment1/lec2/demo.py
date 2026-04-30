"""
Comprehensive demo showing how the simulator works.
"""

from simulator import (
    LimitOrderBookSimulator,
    create_limit_order,
    create_market_order,
    create_cancel
)

def demo_with_explanations():
    print("=" * 70)
    print("LIMIT ORDER BOOK SIMULATOR DEMO")
    print("=" * 70)
    
    # Create simulator with metrics enabled
    sim = LimitOrderBookSimulator(tick_size=0.01, collect_metrics=True)
    
    print("\n1. BUILDING THE ORDER BOOK")
    print("-" * 70)
    print("Adding limit orders to create a two-sided market...\n")
    
    # Build a realistic order book
    events = [
        # Bids (buy orders) - prices descending
        create_limit_order("B1", "BUY", 100.00, 100, timestamp_ns=1000),
        create_limit_order("B2", "BUY", 99.95, 150, timestamp_ns=2000),
        create_limit_order("B3", "BUY", 99.90, 200, timestamp_ns=3000),
        create_limit_order("B4", "BUY", 99.85, 250, timestamp_ns=4000),
        
        # Asks (sell orders) - prices ascending
        create_limit_order("A1", "SELL", 100.10, 100, timestamp_ns=5000),
        create_limit_order("A2", "SELL", 100.15, 150, timestamp_ns=6000),
        create_limit_order("A3", "SELL", 100.20, 200, timestamp_ns=7000),
        create_limit_order("A4", "SELL", 100.25, 250, timestamp_ns=8000),
    ]
    
    # Run the initial orders
    result = sim.run(events)
    
    print(f"✓ Added {result.successful_events} limit orders")
    print(f"  - Total events processed: {result.total_events}")
    print(f"  - No fills yet (orders don't cross the spread)")
    
    # Show L1 (best bid/ask)
    print("\n2. LEVEL 1 (TOP OF BOOK)")
    print("-" * 70)
    l1 = sim.get_l1_snapshot()
    print(f"Best Bid: {l1.best_bid_price:.2f} @ {l1.best_bid_qty} shares")
    print(f"Best Ask: {l1.best_ask_price:.2f} @ {l1.best_ask_qty} shares")
    print(f"Spread:   {l1.spread:.2f} ({l1.spread_bps:.1f} bps)" if l1.spread else "Spread: N/A")
    print(f"Mid:      {l1.mid_price:.2f}")
    print(f"Micro:    {l1.micro_price:.2f} (volume-weighted mid)")
    
    # Show L2 (price levels)
    print("\n3. LEVEL 2 (MARKET DEPTH)")
    print("-" * 70)
    l2 = sim.get_l2_snapshot(levels=5)
    
    print("BIDS (sorted high to low):")
    for price, qty in l2.bids:
        print(f"  {price:7.2f} | {qty:4d} shares")
    
    print("\nASKS (sorted low to high):")
    for price, qty in l2.asks:
        print(f"  {price:7.2f} | {qty:4d} shares")
    
    imb = l2.volume_imbalance(5)
    if imb is not None:
        imb_pct = imb * 100
        side = "bid" if imb > 0 else "ask"
        print(f"\nVolume Imbalance (top 5): {imb_pct:+.1f}% (more {side} liquidity)")
    
    # Execute a market buy
    print("\n4. EXECUTING MARKET BUY ORDER")
    print("-" * 70)
    print("Submitting market buy for 80 shares...")
    
    success, fills, error = sim.process_event(
        create_market_order("MKT1", "BUY", 80, timestamp_ns=9000)
    )
    
    if success and fills:
        print(f"\n✓ Order executed! Generated {len(fills)} fill(s):")
        total_notional = 0
        for i, fill in enumerate(fills, 1):
            notional = fill.price * fill.qty
            total_notional += notional
            print(f"  Fill {i}: {fill.qty:3d} @ {fill.price:.2f} = ${notional:7.2f}")
            print(f"          (matched against maker order {fill.maker_order_id})")
        
        avg_price = total_notional / sum(f.qty for f in fills)
        print(f"\n  Average execution price: ${avg_price:.4f}")
        print(f"  Total cost: ${total_notional:.2f}")
        
        # Compare to mid price before execution
        slippage = avg_price - 100.10  # best ask was 100.10
        print(f"  Slippage: ${slippage:.4f} (paid {slippage:.4f} above best ask)")
    
    # Show updated book
    print("\n5. UPDATED BOOK AFTER TRADE")
    print("-" * 70)
    l1_after = sim.get_l1_snapshot()
    print(f"Best Bid: {l1_after.best_bid_price:.2f} @ {l1_after.best_bid_qty}")
    print(f"Best Ask: {l1_after.best_ask_price:.2f} @ {l1_after.best_ask_qty}")
    print(f"  → Ask quantity reduced from 100 to {l1_after.best_ask_qty}")
    
    # Execute a market sell
    print("\n6. EXECUTING MARKET SELL ORDER")
    print("-" * 70)
    print("Submitting market sell for 120 shares...")
    
    success, fills, error = sim.process_event(
        create_market_order("MKT2", "SELL", 120, timestamp_ns=10000)
    )
    
    if success and fills:
        print(f"\n✓ Order executed! Generated {len(fills)} fill(s):")
        for i, fill in enumerate(fills, 1):
            print(f"  Fill {i}: {fill.qty:3d} @ {fill.price:.2f}")
        
        if len(fills) > 1:
            print(f"\n  → Order swept through multiple price levels!")
            print(f"    (not enough liquidity at best bid)")
    
    # Cancel an order
    print("\n7. CANCELING AN ORDER")
    print("-" * 70)
    print("Canceling order B4...")
    
    success, fills, error = sim.process_event(
        create_cancel("B4", timestamp_ns=11000)
    )
    
    if success:
        print("✓ Order canceled successfully")
    else:
        print(f"✗ Cancel failed: {error}")
    
    # Show final metrics
    print("\n8. SIMULATION METRICS")
    print("-" * 70)
    
    if sim.metrics_collector:
        summary = sim.metrics_collector.get_summary()
        print(f"Total Trades:     {summary['total_trades']}")
        print(f"Total Volume:     {summary['total_volume']} shares")
        
        if summary['trade_flow_imbalance'] is not None:
            tfi = summary['trade_flow_imbalance'] * 100
            side = "buying" if summary['trade_flow_imbalance'] > 0 else "selling"
            print(f"Flow Imbalance:   {tfi:+.1f}% (more {side} pressure)")
    
    # Show L3 snapshot
    print("\n9. LEVEL 3 (ORDER-LEVEL DETAIL)")
    print("-" * 70)
    l3 = sim.get_l3_snapshot()
    print(f"Total orders in book: {l3.total_orders()}")
    print(f"  Bid orders: {l3.total_bid_orders()}")
    print(f"  Ask orders: {l3.total_ask_orders()}")
    
    # Show some individual orders
    print("\nSample bid orders at best price:")
    best_bid_price = list(l3.bids.keys())[0] if l3.bids else None
    if best_bid_price and best_bid_price in l3.bids:
        for order in l3.bids[best_bid_price][:3]:
            print(f"  Order {order['order_id']}: {order['qty']} shares @ {best_bid_price:.2f}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nKEY TAKEAWAYS:")
    print("• The simulator uses dyn4mik3/OrderBook as the matching engine")
    print("• Orders maintain price-time priority")
    print("• Market orders can sweep multiple price levels")
    print("• L1/L2/L3 snapshots provide different views of the book")
    print("• Metrics track market quality (spread, imbalance, VWAP, etc.)")
    print("=" * 70)

if __name__ == "__main__":
    demo_with_explanations()
