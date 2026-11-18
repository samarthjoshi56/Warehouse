"""LIFO stack with capacity and compatibility enforcement."""
from .exceptions import CapacityError
class Stack:
    def __init__(self,index:int,group:int,capacity:int):
        self.index=index; self.group=group; self.capacity=capacity; self._items=[]
    @property
    def items(self): return tuple(self._items)
    @property
    def height(self): return len(self._items)
    @property
    def free_slots(self): return self.capacity-len(self._items)
    @property
    def top(self): return self._items[-1] if self._items else None
    def push(self,item):
        if self.height>=self.capacity: raise CapacityError(f"Stack {self.index} is full")
        if not item.is_compatible(self.group): raise ValueError(f"Item {item.id} incompatible with stack group {self.group}")
        self._items.append(item); item.location=self.index; item.status="stored"; self.assert_invariant()
    def pop(self):
        if not self._items: raise IndexError(f"Stack {self.index} is empty")
        item=self._items.pop(); self.assert_invariant(); return item
    def position_of(self,item_id):
        for idx,item in enumerate(self._items):
            if item.id==item_id: return idx
        raise ValueError(f"Item {item_id} is not in stack {self.index}")
    def remove_above(self,position):
        if not 0<=position<self.height: raise IndexError("Target position outside stack")
        return [self.pop() for _ in range(self.height-position-1)]
    def assert_invariant(self):
        if len(self._items)>self.capacity: raise AssertionError("Stack capacity invariant violated")
        if len({x.id for x in self._items})!=len(self._items): raise AssertionError("Duplicate item in stack")
