"""
Simple test script to verify the limit order book simulator is working correctly.
"""

from simulator import (
    LimitOrderBookSimulator, 
    create_limit_order,
    create_market_order
)


def test_basic_order_flow():
    """Test basic limit order book operations."""
    print("Testing basic order flow...")
    
    sim = LimitOrderBookSimulator(tick_size=0.01, collect_metrics=True)
    
    # Build a two-sided market
    events = [
        # Bids
        create_limit_order("bid1", "BUY", 99.95, 100, timestamp_ns=1000),
        create_limit_order("bid2", "BUY", 99.90, 150, timestamp_ns=2000),
        create_limit_order("bid3", "BUY", 99.85, 200, timestamp_ns=3000),
        
        # Asks
        create_limit_order("ask1", "SELL", 100.05, 100, timestamp_ns=4000),
        create_limit_order("ask2", "SELL", 100.10, 150, timestamp_ns=5000),
        create_limit_order("ask3", "SELL", 100.15, 200, timestamp_ns=6000),
    ]
    
    result = sim.run(events)
    
    # Verify the book state
    l1 = sim.get_l1_snapshot()
    assert l1.best_bid_price == 99.95, f"Expected best bid 99.95, got {l1.best_bid_price}"
    assert l1.best_ask_price == 100.05, f"Expected best ask 100.05, got {l1.best_ask_price}"
    assert l1.spread == 0.10, f"Expected spread 0.10, got {l1.spread}"
    
    print(f"✓ Book state correct: {l1}")
    
    # Execute a market buy
    success, fills, error = sim.process_event(
        create_market_order("mkt1", "BUY", 50, timestamp_ns=7000)
    )
    
    assert success, f"Market order failed: {error}"
    assert len(fills) == 1, f"Expected 1 fill, got {len(fills)}"
    assert fills[0].price == 100.05, f"Expected fill price 100.05, got {fills[0].price}"
    assert fills[0].qty == 50, f"Expected fill qty 50, got {fills[0].qty}"
    
    print(f"✓ Market order executed correctly: {len(fills)} fills")
    
    # Check updated book
    l1 = sim.get_l1_snapshot()
    assert l1.best_ask_qty == 50, f"Expected best ask qty 50, got {l1.best_ask_qty}"
    
    print("✓ All tests passed!")
    return True


def test_price_time_priority():
    """Test that price-time priority is maintained."""
    print("\nTesting price-time priority...")
    
    sim = LimitOrderBookSimulator(tick_size=0.01)
    
    # Add orders at same price in sequence
    events = [
        create_limit_order("ask1", "SELL", 100.00, 100, timestamp_ns=1000),
        create_limit_order("ask2", "SELL", 100.00, 100, timestamp_ns=2000),
        create_limit_order("ask3", "SELL", 100.00, 100, timestamp_ns=3000),
    ]
    
    sim.run(events)
    
    # Execute market buy - should match ask1 first (earliest timestamp)
    _, fills, _ = sim.process_event(
        create_market_order("mkt1", "BUY", 100, timestamp_ns=4000)
    )
    
    assert fills[0].maker_order_id == "1" or "ask_1" in fills[0].maker_order_id.lower(), \
        f"Expected first order to be matched, got {fills[0].maker_order_id}"
    
    print("✓ Price-time priority maintained")
    return True


def test_metrics_collection():
    """Test metrics collection."""
    print("\nTesting metrics collection...")
    
    sim = LimitOrderBookSimulator(tick_size=0.01, collect_metrics=True)
    
    events = [
        create_limit_order("bid1", "BUY", 100.00, 100, timestamp_ns=1000),
        create_limit_order("ask1", "SELL", 100.10, 100, timestamp_ns=2000),
        create_market_order("mkt1", "BUY", 50, timestamp_ns=3000),
    ]
    
    result = sim.run(events)
    
    assert result.total_fills == 1, f"Expected 1 fill, got {result.total_fills}"
    assert result.total_volume == 50, f"Expected volume 50, got {result.total_volume}"
    assert result.metrics_summary['total_trades'] == 1, "Expected 1 trade in metrics"
    
    print("✓ Metrics collected correctly")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("LIMIT ORDER BOOK SIMULATOR - Test Suite")
    print("=" * 60)
    
    try:
        test_basic_order_flow()
        test_price_time_priority()
        test_metrics_collection()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
