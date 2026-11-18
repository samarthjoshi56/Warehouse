"""Read-only state projections used by agents and diagnostics."""
import numpy as np
class WarehouseState:
    def __init__(self,env): self.env=env
    def to_full_state(self):
        return {"stacks":[[x.public_dict() for x in s.items] for s in self.env.stacks],
                "temp_storage":[x.public_dict() for x in self.env.temp_storage.items],
                "sim_time":self.env.sim_time,"reshuffles":self.env.metrics.data.reshuffles,
                "pending_items":[x.id for x in self.env.items.values() if x.status=="pending"]}
    def to_per_slot_tensor(self):
        env,cfg=self.env,self.env.config; x=np.zeros((cfg.num_stacks,cfg.stack_height,6),dtype=np.float32)
        for k,stack in enumerate(env.stacks):
            for h,item in enumerate(stack.items):
                x[k,h,0]=1.0
                x[k,h,1]=-1.0 if item.expected_retrieval_time is None else np.clip(item.expected_retrieval_time/cfg.max_episode_steps,-1,1)
                x[k,h,2]=item.priority/5.0; x[k,h,3]=stack.group/max(1,cfg.num_groups)
                x[k,h,4]=1.0 if h==stack.height-1 else 0.0; x[k,h,5]=h/cfg.stack_height
        return x
    def to_compact_vector(self):
        env,cfg=self.env,self.env.config
        heights=[s.height/cfg.stack_height for s in env.stacks]; free=[s.free_slots/cfg.stack_height for s in env.stacks]
        top_tau=[-1.0 if s.top is None or s.top.expected_retrieval_time is None else s.top.expected_retrieval_time/cfg.max_episode_steps for s in env.stacks]
        incoming=env.current_item
        item_tau=-1.0 if incoming is None or incoming.expected_retrieval_time is None else incoming.expected_retrieval_time/cfg.max_episode_steps
        priority=0.0 if incoming is None else incoming.priority/5.0
        group=0.0 if incoming is None else min(incoming.compatible_groups)/max(1,cfg.num_groups)
        return np.asarray(heights+free+top_tau+[env.temp_storage.count/max(1,cfg.temp_capacity),item_tau,priority,group,env.sim_time/cfg.max_episode_steps],dtype=np.float32)
