"""Fixed-capacity random-access temporary storage pool."""
from .exceptions import TemporaryStorageFullError
class TempStorage:
    def __init__(self,capacity:int): self.capacity=capacity; self._items={}
    @property
    def items(self): return tuple(self._items.values())
    @property
    def count(self): return len(self._items)
    @property
    def free_slots(self): return self.capacity-self.count
    def is_full(self): return self.count>=self.capacity
    def add(self,item):
        if self.is_full(): raise TemporaryStorageFullError("Temporary storage is full")
        if item.id in self._items: raise ValueError(f"Duplicate temporary item {item.id}")
        self._items[item.id]=item; item.location="temp"; item.status="in_temp"
    def remove(self,item_id):
        try: return self._items.pop(item_id)
        except KeyError as exc: raise KeyError(f"Item {item_id} is not in temporary storage") from exc
    def get(self,item_id): return self._items.get(item_id)
