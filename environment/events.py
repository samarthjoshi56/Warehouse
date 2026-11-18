"""Immutable schedule event records."""
from dataclasses import dataclass
@dataclass(frozen=True)
class ArrivalEvent:
    due_time:int
    item_id:str
    event_type:str="arrival"
@dataclass(frozen=True)
class RetrievalEvent:
    due_time:int
    item_id:str
    priority:int
    event_type:str="retrieval"
    @property
    def sort_key(self): return (self.due_time,self.priority,self.item_id)
