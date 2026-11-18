"""Episode metrics. Reshuffle count is independent of reward."""
from dataclasses import asdict,dataclass
@dataclass
class MetricsSummary:
    reshuffles:int=0
    placements:int=0
    restores:int=0
    retrieved:int=0
    failed:int=0
    unplaced:int=0
    invalid_actions:int=0
    deferred_retrievals:int=0
    total_retrieval_delay:float=0.0
    elapsed_steps:int=0
    runtime_seconds:float=0.0
    @property
    def reshuffles_per_retrieved_item(self): return self.reshuffles/self.retrieved if self.retrieved else 0.0
    def to_dict(self):
        data=asdict(self); data["reshuffles_per_retrieved_item"]=self.reshuffles_per_retrieved_item; return data
class MetricsTracker:
    def __init__(self): self.reset()
    def reset(self): self.data=MetricsSummary()
    def get_summary(self): return self.data.to_dict()
