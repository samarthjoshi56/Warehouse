"""Item model. Actual retrieval time is simulator-only."""
from dataclasses import dataclass,field
@dataclass
class Item:
    id:str
    arrival_time:int
    expected_retrieval_time:int|None
    actual_retrieval_time:int
    priority:int=3
    compatible_groups:frozenset[int]=field(default_factory=lambda:frozenset({0}))
    location:int|str="pending"
    status:str="pending"
    def __post_init__(self):
        if not self.id: raise ValueError("Item id cannot be empty")
        if self.arrival_time<0 or self.actual_retrieval_time<self.arrival_time: raise ValueError("Invalid item timestamps")
        if self.expected_retrieval_time is not None and self.expected_retrieval_time<0: raise ValueError("Expected retrieval time must be non-negative")
        if not 1<=self.priority<=5: raise ValueError("priority must be in [1, 5]")
        if not self.compatible_groups: raise ValueError("Item must be compatible with at least one stack group")
    def is_compatible(self,stack_group:int)->bool: return stack_group in self.compatible_groups
    @property
    def urgency_key(self):
        tau=float("inf") if self.expected_retrieval_time is None else self.expected_retrieval_time
        return (tau,self.priority,self.id)
    def public_dict(self):
        return {"id":self.id,"arrival_time":self.arrival_time,"expected_retrieval_time":self.expected_retrieval_time,
                "priority":self.priority,"compatible_groups":tuple(sorted(self.compatible_groups)),
                "location":self.location,"status":self.status}
