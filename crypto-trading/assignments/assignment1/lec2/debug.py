from simulator import LimitOrderBookSimulator, create_limit_order

sim = LimitOrderBookSimulator()

events = [
    create_limit_order('buy_1', 'BUY', 100.00, 10, timestamp_ns=1000),
    create_limit_order('buy_2', 'BUY', 99.50, 15, timestamp_ns=2000),
    create_limit_order('buy_3', 'BUY', 99.00, 20, timestamp_ns=3000),
    create_limit_order('sell_1', 'SELL', 100.50, 10, timestamp_ns=4000),
    create_limit_order('sell_2', 'SELL', 101.00, 15, timestamp_ns=5000),
    create_limit_order('sell_3', 'SELL', 101.50, 20, timestamp_ns=6000),
]

result = sim.run(events)
print(f'Successful: {result.successful_events}')
print(f'Failed: {result.failed_events}')
print(f'Validation errors: {result.validation_errors}')

# Show which events had errors
for log in result.event_log:
    if log['error']:
        print(f"Event {log['event_num']} ({log['event_type']}): {log['error']}")

l1 = sim.get_l1_snapshot()
print(f'\nBest bid: {l1.best_bid_price} @ {l1.best_bid_qty}')
print(f'Best ask: {l1.best_ask_price} @ {l1.best_ask_qty}')
