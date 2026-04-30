"""
Event model for limit order book simulator.
Defines the events that can be sent to the simulator with validation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from wrapper import Side, OrderType


class EventType(str, Enum):
    NEW_LIMIT = "NEW_LIMIT"
    NEW_MARKET = "NEW_MARKET"
    CANCEL = "CANCEL"
    AMEND = "AMEND"


@dataclass
class BaseEvent:
    """Base class for all events."""
    event_type: EventType
    timestamp_ns: int
    
    def validate(self) -> Optional[str]:
        """Validate the event. Returns error message if invalid, None if valid."""
        if self.timestamp_ns < 0:
            return "Timestamp cannot be negative"
        return None


@dataclass
class NewLimitEvent(BaseEvent):
    """Event to submit a new limit order."""
    order_id: str
    side: Side
    price: float
    qty: int
    
    def __init__(self, order_id: str, side: Side, price: float, qty: int, timestamp_ns: int):
        super().__init__(EventType.NEW_LIMIT, timestamp_ns)
        self.order_id = order_id
        self.side = side
        self.price = price
        self.qty = qty
    
    def validate(self) -> Optional[str]:
        """Validate limit order parameters."""
        base_error = super().validate()
        if base_error:
            return base_error
        
        if not self.order_id:
            return "Order ID cannot be empty"
        if self.price <= 0:
            return f"Price must be positive, got {self.price}"
        if self.qty <= 0:
            return f"Quantity must be positive, got {self.qty}"
        if not isinstance(self.side, Side):
            return f"Side must be a Side enum, got {type(self.side)}"
        
        return None


@dataclass
class NewMarketEvent(BaseEvent):
    """Event to submit a new market order."""
    order_id: str
    side: Side
    qty: int
    
    def __init__(self, order_id: str, side: Side, qty: int, timestamp_ns: int):
        super().__init__(EventType.NEW_MARKET, timestamp_ns)
        self.order_id = order_id
        self.side = side
        self.qty = qty
    
    def validate(self) -> Optional[str]:
        """Validate market order parameters."""
        base_error = super().validate()
        if base_error:
            return base_error
        
        if not self.order_id:
            return "Order ID cannot be empty"
        if self.qty <= 0:
            return f"Quantity must be positive, got {self.qty}"
        if not isinstance(self.side, Side):
            return f"Side must be a Side enum, got {type(self.side)}"
        
        return None


@dataclass
class CancelEvent(BaseEvent):
    """Event to cancel an existing order."""
    order_id: str
    
    def __init__(self, order_id: str, timestamp_ns: int):
        super().__init__(EventType.CANCEL, timestamp_ns)
        self.order_id = order_id
    
    def validate(self) -> Optional[str]:
        """Validate cancel event."""
        base_error = super().validate()
        if base_error:
            return base_error
        
        if not self.order_id:
            return "Order ID cannot be empty"
        
        return None


@dataclass
class AmendEvent(BaseEvent):
    """Event to amend an existing order (cancel and replace)."""
    order_id: str
    new_price: Optional[float] = None
    new_qty: Optional[int] = None
    
    def __init__(self, order_id: str, timestamp_ns: int, 
                 new_price: Optional[float] = None, new_qty: Optional[int] = None):
        super().__init__(EventType.AMEND, timestamp_ns)
        self.order_id = order_id
        self.new_price = new_price
        self.new_qty = new_qty
    
    def validate(self) -> Optional[str]:
        """Validate amend event."""
        base_error = super().validate()
        if base_error:
            return base_error
        
        if not self.order_id:
            return "Order ID cannot be empty"
        
        if self.new_price is None and self.new_qty is None:
            return "Must specify at least one of new_price or new_qty"
        
        if self.new_price is not None and self.new_price <= 0:
            return f"New price must be positive, got {self.new_price}"
        
        if self.new_qty is not None and self.new_qty <= 0:
            return f"New quantity must be positive, got {self.new_qty}"
        
        return None


# Type alias for all event types
Event = NewLimitEvent | NewMarketEvent | CancelEvent | AmendEvent
