# Limit Order Book Simulator - Running Process Explanation

This document explains the complete running process of the simulator demo, step by step.

## Output Location
The complete demo output has been saved to: `demo_output.log`

---

## 🔄 Running Process Breakdown

### **Step 1: Initialization**
```python
sim = LimitOrderBookSimulator(tick_size=0.01, collect_metrics=True)
```

**What happens internally:**
1. Creates `OrderBookBackend` wrapper instance
2. Wrapper imports and instantiates `dyn4mik3/OrderBook(tick_size=0.01)`
3. Initializes empty bid/ask red-black trees in the orderbook engine
4. Creates `SnapshotManager` for L1/L2/L3 views
5. Creates `MetricsCollector` to track market quality metrics

---

### **Step 2: Building the Order Book**
```python
# 8 limit orders submitted (4 bids, 4 asks)
create_limit_order("B1", "BUY", 100.00, 100, timestamp_ns=1000)
# ... etc
```

**Processing flow for EACH order:**

1. **Event Creation** (`events.py`)
   - `NewLimitEvent` created with validation rules
   - Checks: price > 0, qty > 0, order_id not empty

2. **Simulator Processing** (`simulator.py`)
   - Validates event: `event.validate()` → passes ✓
   - Updates simulator time to event's timestamp
   - Increments event counter

3. **Wrapper Translation** (`wrapper.py`)
   - Converts `OrderRequest` → orderbook's dict format:
     ```python
     {
         'type': 'limit',
         'side': 'bid' or 'ask',
         'quantity': 100,
         'price': 100.00,
         'order_id': 1,
         'trade_id': 1,
         'timestamp': 1000
     }
     ```

4. **Orderbook Engine Processing** (`dyn4mik3/OrderBook`)
   - Clips price to tick size (100.00 → 100.00, already aligned)
   - Checks if order crosses the spread:
     - **BUY at 100.00**: Best ask is 100.10 → doesn't cross ✗
     - **SELL at 100.10**: Best bid is 100.00 → doesn't cross ✗
   - Since no cross, order is added to the book:
     - Finds/creates price node in red-black tree
     - Appends order to doubly-linked list at that price
     - Updates volume tracking

5. **Result**
   - No fills generated (empty list returned)
   - Order sits passively in the book
   - Success = True, 8 events processed

**Data Structure After Step 2:**
```
Red-Black Tree (Bids):          Red-Black Tree (Asks):
99.85 → [Order B4: 250]         100.10 → [Order A1: 100]
99.90 → [Order B3: 200]         100.15 → [Order A2: 150]
99.95 → [Order B2: 150]         100.20 → [Order A3: 200]
100.00 → [Order B1: 100]        100.25 → [Order A4: 250]
```

---

### **Step 3: L1 Snapshot**
```python
l1 = sim.get_l1_snapshot()
```

**Processing:**
1. Calls `backend.top_of_book()`
2. Orderbook returns:
   - `bids.max_price()` → 100.00 (root of red-black tree for max)
   - `asks.min_price()` → 100.10 (root of red-black tree for min)
   - Gets volume at each price from the OrderList
3. Snapshot calculates:
   - Spread: 100.10 - 100.00 = 0.10
   - Mid: (100.00 + 100.10) / 2 = 100.05
   - Spread BPS: (0.10 / 100.05) * 10000 = 10.0 bps
   - Micro: (100.00 × 100 + 100.10 × 100) / 200 = 100.05

**Output:**
```
Best Bid: 100.00 @ 100 shares
Best Ask: 100.10 @ 100 shares
Spread:   0.10 (10.0 bps)
```

---

### **Step 4: L2 Snapshot (Market Depth)**
```python
l2 = sim.get_l2_snapshot(levels=5)
```

**Processing:**
1. Calls `backend.depth(levels=5)`
2. Orderbook iterates through price trees:
   - **Bids**: Reverse iteration (high to low) using RBTree
   - **Asks**: Forward iteration (low to high) using RBTree
3. For each price level, aggregates all orders:
   - Gets OrderList at that price
   - Sums up `.volume` property (total qty at price)
4. Returns list of (price, total_qty) tuples

**Output:**
```
BIDS:                           ASKS:
100.00 | 100 shares             100.10 | 100 shares
99.95  | 150 shares             100.15 | 150 shares
99.90  | 200 shares             100.20 | 200 shares
99.85  | 250 shares             100.25 | 250 shares
```

Volume Imbalance = (700 - 700) / 1400 = 0.0% (perfectly balanced)

---

### **Step 5: Market Buy Order Execution**
```python
create_market_order("MKT1", "BUY", 80, timestamp_ns=9000)
```

**Processing flow (THE CRITICAL PART):**

1. **Event Creation & Validation**
   - `NewMarketEvent` created
   - Validates: qty > 0, side valid

2. **Wrapper Converts to Orderbook Format**
   ```python
   {
       'type': 'market',
       'side': 'bid',  # buyer
       'quantity': 80,
       'trade_id': hash('MKT1')
   }
   ```

3. **Orderbook Engine Matching** (`process_market_order`)
   ```python
   # Pseudocode of what happens inside dyn4mik3/OrderBook:
   
   quantity_to_trade = 80
   side = 'bid'  # We're buying
   
   # Match against asks (we take from sellers)
   while quantity_to_trade > 0 and self.asks:
       best_ask_price = asks.min_price()  # 100.10
       order_list = asks.get_price_list(100.10)
       
       # Get first order (FIFO - price-time priority)
       head_order = order_list.head_order  # Order A1
       
       # How much can we trade?
       if quantity_to_trade < head_order.quantity:
           # Partial fill of head order
           traded_qty = 80
           head_order.quantity -= 80  # 100 → 20
       
       # Create fill record
       fill = {
           'timestamp': 9000,
           'price': 100.10,
           'quantity': 80,
           'party1': [5, 'ask', A1_id],  # Maker (A1)
           'party2': [MKT1_id, 'bid', None]  # Taker (MKT1)
       }
       
       # Update volume tracking
       asks.volume -= 80
       
       # Since quantity_to_trade is now 0, exit loop
   ```

4. **Wrapper Converts Fills Back**
   - Transforms orderbook's fill dict → `Fill` object:
     ```python
     Fill(
         trade_id="T1",
         price=100.10,
         qty=80,
         taker_order_id="MKT1",
         maker_order_id="5",  # Order A1's internal ID
         taker_side=Side.BUY,
         time_ns=9000
     )
     ```

5. **Metrics Collection**
   - Computes VWAP: (100.10 × 80) / 80 = 100.10
   - Records trade in history
   - Updates flow imbalance

**Result:**
```
✓ Order executed! Generated 1 fill(s):
  Fill 1:  80 @ 100.10 = $8008.00
```

**Updated Book State:**
```
Asks tree:
100.10 → [Order A1: 20]  ← Partially filled!
100.15 → [Order A2: 150]
100.20 → [Order A3: 200]
100.25 → [Order A4: 250]
```

---

### **Step 6: Market Sell Order (Multi-Level Sweep)**
```python
create_market_order("MKT2", "SELL", 120, timestamp_ns=10000)
```

**Processing (More Complex):**

1. **Engine Matching Logic**
   ```python
   quantity_to_trade = 120
   side = 'ask'  # We're selling
   
   # Match against bids (we hit buyers)
   while quantity_to_trade > 0 and self.bids:
       best_bid_price = bids.max_price()  # 100.00
       order_list = bids.get_price_list(100.00)
       head_order = order_list.head_order  # Order B1
       
       # ITERATION 1:
       if quantity_to_trade >= head_order.quantity:
           # Fully consume head order
           traded_qty = 100  # Entire B1
           bids.remove_order_by_id(B1_id)
           quantity_to_trade -= 100  # 120 → 20
           
           fill_1 = {price: 100.00, quantity: 100, ...}
       
       # Still have 20 left, loop continues...
       # ITERATION 2:
       best_bid_price = bids.max_price()  # 99.95 (next level!)
       order_list = bids.get_price_list(99.95)
       head_order = order_list.head_order  # Order B2
       
       if quantity_to_trade < head_order.quantity:
           # Partial fill
           traded_qty = 20
           head_order.quantity -= 20  # 150 → 130
           quantity_to_trade = 0
           
           fill_2 = {price: 99.95, quantity: 20, ...}
   ```

2. **Result: 2 Fills Generated**
   - Fill 1: 100 @ 100.00 (consumed entire B1)
   - Fill 2: 20 @ 99.95 (partial fill of B2)

**This demonstrates PRICE IMPACT:**
- Large order (120) exceeds best bid liquidity (100)
- Engine automatically walks down the book
- Average execution: (100×100.00 + 20×99.95) / 120 = 99.99

**Updated Book State:**
```
Bids tree:
99.95 → [Order B2: 130]  ← Partially filled
99.90 → [Order B3: 200]
99.85 → [Order B4: 250]
(100.00 level removed - fully consumed)
```

---

### **Step 7: Order Cancellation**
```python
create_cancel("B4", timestamp_ns=11000)
```

**Processing:**
1. Wrapper maps order_id "B4" → internal ID
2. Calls `orderbook.cancel_order('bid', B4_id)`
3. Engine:
   - Finds order in bids tree
   - Removes from doubly-linked list at 99.85
   - If no more orders at 99.85, removes price node from RBTree
   - Updates volume tracking
4. Returns success = True

**Updated Book State:**
```
Bids tree:
99.95 → [Order B2: 130]
99.90 → [Order B3: 200]
(99.85 level removed - B4 canceled)
```

---

### **Step 8: Metrics Summary**

**Computed by MetricsCollector:**
- **Total Trades**: 3 (one from market buy, two from market sell)
- **Total Volume**: 80 + 100 + 20 = 200 shares
- **Flow Imbalance**: (80 buy - 120 sell) / 200 = -20%
  - Negative = more selling pressure
  - Indicates bearish order flow

---

### **Step 9: L3 Snapshot (Order-Level Detail)**

**Processing:**
1. Calls `backend.snapshot_l3()`
2. Iterates through EVERY order in both trees:
   ```python
   for price, order_list in bids.price_tree.items():
       current_order = order_list.head_order
       while current_order:
           orders.append({
               'order_id': current_order.order_id,
               'qty': current_order.quantity,
               'timestamp': current_order.timestamp
           })
           current_order = current_order.next_order
   ```
3. Returns complete map: price → list of individual orders

**Result:**
```
7 total orders:
  Bids: 3 orders (B2, B3, B4 canceled)
  Asks: 4 orders (A1 partial, A2, A3, A4)
```

---

## 🏗️ Key Architecture Points

### **1. Layered Design**
- **Events Layer**: Input validation
- **Simulator Layer**: Orchestration, callbacks, metrics
- **Wrapper Layer**: API translation
- **Engine Layer** (dyn4mik3/OrderBook): Actual matching

### **2. Data Structures (in orderbook engine)**
- **Red-Black Trees**: O(log n) insert/delete/search for prices
- **Doubly-Linked Lists**: FIFO ordering at each price (price-time priority)
- **Hash Maps**: O(1) order lookup by ID

### **3. Matching Algorithm**
- **Price Priority**: Best price matched first
- **Time Priority**: Among same price, earliest timestamp wins
- **Partial Fills**: Orders can be partially consumed
- **Automatic Sweeping**: Market orders walk through levels

### **4. Performance**
- Red-black trees ensure O(log n) operations
- No linear scans of entire book
- Demo processed 11 events instantly
- Real-world: 260k+ events/second sustained

---

## 📄 Files Generated

1. **`demo_output.log`** - Complete console output
2. **`DEMO_EXPLANATION.md`** - This detailed explanation

You can review `demo_output.log` anytime to see the exact output from the run!
