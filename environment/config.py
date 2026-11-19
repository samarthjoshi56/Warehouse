"""Typed configuration for the warehouse simulator."""
from dataclasses import dataclass, field

@dataclass(frozen=True)
class WarehouseConfig:
    num_stacks:int=10
    stack_height:int=5
    temp_capacity:int=10
    num_items:int=100
    max_episode_steps:int=1200
    num_groups:int=1
    scenario:str="B"
    arrival_interval:int=1
    min_dwell_time:int=8
    max_dwell_time:int=60
    retrieval_noise_std:float=5.0
    seed:int|None=None
    safe_temp_threshold:int=6
    max_feasible_dispatch_failures:int=3
    lambda_place:float=0.2
    lambda_delay:float=0.05
    lambda_reshuffle:float=1.0
    lambda_unplaced:float=10.0
    lambda_invalid:float=10.0
    lambda_unretrieved:float=20.0
    lambda_temp:float=2.0
    success_reward:float=2.0
    lookahead_window:int=20
    stack_groups:tuple[int,...]=field(default_factory=tuple)
    observation_version:int=1
    def __post_init__(self):
        if self.observation_version not in {1,2}: raise ValueError("observation_version must be 1 or 2")
        if self.num_stacks<1 or self.stack_height<1: raise ValueError("num_stacks and stack_height must be positive")
        if self.temp_capacity<0 or self.num_items<1: raise ValueError("temp_capacity must be non-negative and num_items positive")
        if self.max_episode_steps<1 or self.num_groups<1: raise ValueError("max_episode_steps and num_groups must be positive")
        if self.scenario not in {"A","B","C"}: raise ValueError("scenario must be A, B, or C")
        if self.arrival_interval<1 or self.min_dwell_time<1 or self.max_dwell_time<self.min_dwell_time: raise ValueError("Invalid timing configuration")
        if self.stack_groups and len(self.stack_groups)!=self.num_stacks: raise ValueError("stack_groups must contain one group per stack")
        if self.stack_groups and any(g<0 or g>=self.num_groups for g in self.stack_groups): raise ValueError("stack group outside configured range")
