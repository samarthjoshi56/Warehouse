"""Reproducible episode configuration and schedule generation."""
from dataclasses import dataclass
import numpy as np
from .config import WarehouseConfig
from .item import Item
from .events import ArrivalEvent,RetrievalEvent
@dataclass
class EpisodeConfig:
    items:list[Item]
    arrivals:list[ArrivalEvent]
    retrievals:list[RetrievalEvent]
    seed:int
class EpisodeGenerator:
    def __init__(self,config:WarehouseConfig): self.config=config
    def generate(self,seed:int|None=None)->EpisodeConfig:
        seed=self.config.seed if seed is None else seed
        if seed is None: seed=0
        rng=np.random.default_rng(seed); items=[]; arrivals=[]; retrievals=[]
        for i in range(self.config.num_items):
            item_id=f"ITEM_{i+1:03d}"; arrival=i*self.config.arrival_interval
            dwell=int(rng.integers(self.config.min_dwell_time,self.config.max_dwell_time+1)); actual=arrival+dwell
            if self.config.scenario=="A": expected=None
            elif self.config.scenario=="B": expected=max(arrival,int(round(actual+rng.normal(0,self.config.retrieval_noise_std))))
            else: expected=actual
            priority=int(rng.integers(1,6)); group=int(rng.integers(0,self.config.num_groups))
            items.append(Item(item_id,arrival,expected,actual,priority,frozenset({group})))
            arrivals.append(ArrivalEvent(arrival,item_id)); retrievals.append(RetrievalEvent(actual,item_id,priority))
        return EpisodeConfig(items,arrivals,retrievals,int(seed))
