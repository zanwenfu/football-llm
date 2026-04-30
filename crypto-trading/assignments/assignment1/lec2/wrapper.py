from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


@dataclass(frozen=True)
class OrderRequest:
    order_id: str
    side: Side
    qty: int
    order_type: OrderType
    price: Optional[float] = None  # float or int ticks; you decide
    time_ns: Optional[int] = None  # simulator timestamp


@dataclass(frozen=True)
class Fill:
    trade_id: str
    price: float
    qty: int
    taker_order_id: str
    maker_order_id: str
    taker_side: Side
    time_ns: int


class LobBackend:
    """Backend-agnostic interface."""
    def submit(self, req: OrderRequest) -> List[Fill]:
        raise NotImplementedError

    def cancel(self, order_id: str) -> bool:
        raise NotImplementedError

    def amend(self, order_id: str, new_price: Optional[float] = None, new_qty: Optional[int] = None) -> List[Fill]:
        """Typically cancel+replace; may generate fills if new order crosses."""
        raise NotImplementedError

    def top_of_book(self) -> Dict[str, Optional[Tuple[float, int]]]:
        """Return {'bid': (price, qty) or None, 'ask': (price, qty) or None}"""
        raise NotImplementedError

    def depth(self, levels: int = 10) -> Dict[str, List[Tuple[float, int]]]:
        """Return {'bids': [(p, qty)...], 'asks': [(p, qty)...]}"""
        raise NotImplementedError

    def snapshot_l3(self) -> Dict[str, Any]:
        """Optional: full order-level snapshot."""
        raise NotImplementedError


class OrderBookBackend(LobBackend):
    """Wrapper for dyn4mik3/OrderBook matching engine."""
    
    def __init__(self, tick_size: float = 0.01):
        # Import the OrderBook class
        from orderbook import OrderBook
        self.ob = OrderBook(tick_size=tick_size)
        self.next_trade_id = 1
        self._order_side_map: Dict[str, str] = {}  # order_id -> 'bid'/'ask'
        
    def _convert_side(self, side: Side) -> str:
        """Convert our Side enum to orderbook's 'bid'/'ask'."""
        return 'bid' if side == Side.BUY else 'ask'
    
    def _convert_side_back(self, side_str: str) -> Side:
        """Convert orderbook's 'bid'/'ask' to our Side enum."""
        return Side.BUY if side_str == 'bid' else Side.SELL
    
    def _make_quote(self, req: OrderRequest) -> Dict[str, Any]:
        """Convert OrderRequest to orderbook quote format."""
        side_str = self._convert_side(req.side)
        quote = {
            'type': 'market' if req.order_type == OrderType.MARKET else 'limit',
            'side': side_str,
            'quantity': req.qty,
            'trade_id': int(req.order_id.split('_')[-1]) if '_' in req.order_id else hash(req.order_id) % 1000000,
            'order_id': int(req.order_id.split('_')[-1]) if '_' in req.order_id else hash(req.order_id) % 1000000,
        }
        if req.time_ns is not None:
            quote['timestamp'] = req.time_ns
        if req.price is not None:
            quote['price'] = float(req.price)
        return quote
    
    def _convert_trades(self, trades: List[Dict], taker_order_id: str, taker_side: Side) -> List[Fill]:
        """Convert orderbook trade records to Fill objects."""
        fills = []
        for trade in trades:
            # Extract maker and taker info
            party1, party2 = trade['party1'], trade['party2']
            
            # party1 is always the maker (existing order)
            # party2 is always the taker (incoming order)
            maker_trade_id, maker_side, maker_order_id = party1
            taker_trade_id, taker_side_str, _ = party2
            
            fill = Fill(
                trade_id=f"T{self.next_trade_id}",
                price=float(trade['price']),
                qty=int(trade['quantity']),
                taker_order_id=taker_order_id,
                maker_order_id=str(maker_order_id) if maker_order_id else f"O{maker_trade_id}",
                taker_side=taker_side,
                time_ns=int(trade['timestamp'])
            )
            self.next_trade_id += 1
            fills.append(fill)
        return fills
    
    def submit(self, req: OrderRequest) -> List[Fill]:
        """Submit a new order (market or limit)."""
        quote = self._make_quote(req)
        side_str = self._convert_side(req.side)
        
        # Track the side for this order
        self._order_side_map[req.order_id] = side_str
        
        # Process the order
        trades, order_in_book = self.ob.process_order(quote, from_data=False, verbose=False)
        
        # Convert trades to fills
        return self._convert_trades(trades, req.order_id, req.side)
    
    def cancel(self, order_id: str) -> bool:
        """Cancel an existing order."""
        if order_id not in self._order_side_map:
            return False
        
        side_str = self._order_side_map[order_id]
        order_id_int = int(order_id.split('_')[-1]) if '_' in order_id else hash(order_id) % 1000000
        
        self.ob.cancel_order(side_str, order_id_int)
        del self._order_side_map[order_id]
        return True
    
    def amend(self, order_id: str, new_price: Optional[float] = None, new_qty: Optional[int] = None) -> List[Fill]:
        """Amend an existing order (cancel and replace)."""
        if order_id not in self._order_side_map:
            return []
        
        side_str = self._order_side_map[order_id]
        order_id_int = int(order_id.split('_')[-1]) if '_' in order_id else hash(order_id) % 1000000
        
        # Get current order info from the tree
        tree = self.ob.bids if side_str == 'bid' else self.ob.asks
        
        if not tree.order_exists(order_id_int):
            return []
        
        current_order = tree.get_order(order_id_int)
        
        # Build update dict
        order_update = {
            'side': side_str,
            'quantity': new_qty if new_qty is not None else int(current_order.quantity),
            'price': new_price if new_price is not None else float(current_order.price),
            'trade_id': current_order.trade_id,
        }
        
        # Modify the order
        self.ob.modify_order(order_id_int, order_update)
        
        # Note: modify_order doesn't generate trades in this implementation
        # For a more sophisticated version, we'd need to cancel and resubmit
        return []
    
    def top_of_book(self) -> Dict[str, Optional[Tuple[float, int]]]:
        """Get best bid and ask."""
        result = {}
        
        best_bid_price = self.ob.get_best_bid()
        if best_bid_price is not None:
            best_bid_qty = self.ob.get_volume_at_price('bid', best_bid_price)
            result['bid'] = (float(best_bid_price), int(best_bid_qty))
        else:
            result['bid'] = None
        
        best_ask_price = self.ob.get_best_ask()
        if best_ask_price is not None:
            best_ask_qty = self.ob.get_volume_at_price('ask', best_ask_price)
            result['ask'] = (float(best_ask_price), int(best_ask_qty))
        else:
            result['ask'] = None
        
        return result
    
    def depth(self, levels: int = 10) -> Dict[str, List[Tuple[float, int]]]:
        """Get market depth (multiple price levels)."""
        bids = []
        asks = []
        
        # Get bids (sorted descending by price)
        if self.ob.bids.depth > 0:
            for price, order_list in self.ob.bids.price_tree.items(reverse=True):
                if len(bids) >= levels:
                    break
                bids.append((float(price), int(order_list.volume)))
        
        # Get asks (sorted ascending by price)
        if self.ob.asks.depth > 0:
            for price, order_list in self.ob.asks.price_tree.items():
                if len(asks) >= levels:
                    break
                asks.append((float(price), int(order_list.volume)))
        
        return {'bids': bids, 'asks': asks}
    
    def snapshot_l3(self) -> Dict[str, Any]:
        """Get full order-level snapshot."""
        snapshot = {
            'bids': {},  # price -> list of orders
            'asks': {},  # price -> list of orders
            'time': self.ob.time
        }
        
        # Collect bid orders
        if self.ob.bids.depth > 0:
            for price, order_list in self.ob.bids.price_tree.items(reverse=True):
                orders = []
                current = order_list.head_order
                while current is not None:
                    orders.append({
                        'order_id': current.order_id,
                        'qty': int(current.quantity),
                        'timestamp': current.timestamp
                    })
                    current = current.next_order
                snapshot['bids'][float(price)] = orders
        
        # Collect ask orders
        if self.ob.asks.depth > 0:
            for price, order_list in self.ob.asks.price_tree.items():
                orders = []
                current = order_list.head_order
                while current is not None:
                    orders.append({
                        'order_id': current.order_id,
                        'qty': int(current.quantity),
                        'timestamp': current.timestamp
                    })
                    current = current.next_order
                snapshot['asks'][float(price)] = orders
        
        return snapshot
