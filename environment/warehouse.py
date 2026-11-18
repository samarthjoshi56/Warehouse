"""Gymnasium-compatible warehouse simulator."""
from __future__ import annotations
import time
import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    class _Env:
        metadata={}
        def reset(self,seed=None,options=None): self.np_random=np.random.default_rng(seed)
    class _Box:
        def __init__(self,low,high,shape,dtype): self.low,self.high,self.shape,self.dtype=low,high,shape,dtype
        def contains(self,x):
            a=np.asarray(x); return a.shape==self.shape and np.all(a>=self.low) and np.all(a<=self.high)
    class _Discrete:
        def __init__(self,n): self.n=n
    class _Dict:
        def __init__(self,spaces): self.spaces=spaces
        def contains(self,obs): return set(obs)==set(self.spaces) and all(self.spaces[k].contains(obs[k]) for k in obs)
    class _Gym: Env=_Env
    class _Spaces: Box=_Box; Discrete=_Discrete; Dict=_Dict
    gym,spaces=_Gym(),_Spaces()
from .config import WarehouseConfig
from .episode import EpisodeGenerator,EpisodeConfig
from .stack import Stack
from .temp_storage import TempStorage
from .metrics import MetricsTracker
from .retrieval import RetrievalEngine
from .state import WarehouseState
from .exceptions import InvalidActionError,ItemConservationError

class WarehouseEnv(gym.Env):
    metadata={"render_modes":["human","ansi"]}
    def __init__(self,config:WarehouseConfig|None=None,render_mode=None):
        self.config=config or WarehouseConfig(); self.render_mode=render_mode; cfg=self.config
        self.stack_groups=cfg.stack_groups or tuple(k%cfg.num_groups for k in range(cfg.num_stacks))
        self.action_space=spaces.Discrete(cfg.num_stacks)
        self.observation_space=spaces.Dict({
            "grid":spaces.Box(-1.0,1.0,(cfg.num_stacks,cfg.stack_height,6),np.float32),
            "incoming_item":spaces.Box(-1.0,1.0,(4,),np.float32),
            "temp_storage":spaces.Box(-1.0,1.0,(3,),np.float32),
            "stack_summaries":spaces.Box(0.0,1.0,(cfg.num_stacks,3),np.float32),
            "schedule_lookahead":spaces.Box(-1.0,1.0,(cfg.lookahead_window,4),np.float32),
            "action_mask":spaces.Box(0,1,(cfg.num_stacks,),np.int8),
            "sim_time":spaces.Box(0.0,1.0,(1,),np.float32)})
        self.reset(seed=cfg.seed)

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.seed=self.config.seed if seed is None else seed
        self.episode:EpisodeConfig=EpisodeGenerator(self.config).generate(self.seed)
        self.items={item.id:item for item in self.episode.items}
        self.stacks=[Stack(k,self.stack_groups[k],self.config.stack_height) for k in range(self.config.num_stacks)]
        self.temp_storage=TempStorage(self.config.temp_capacity); self.metrics=MetricsTracker()
        self.retrieval_engine=RetrievalEngine(self.stacks,self.temp_storage,self.metrics,self.config.safe_temp_threshold)
        self.state=WarehouseState(self); self.sim_time=0; self._arrival_index=0
        self._retrieval_events=list(self.episode.retrievals); self._deferred={}; self.current_item=None
        self._terminated=False; self._terminal_reward_paid=False; self._async_reward=0.0
        self._episode_wall_start=time.perf_counter()
        self._prepare_next_arrival(initial=True); self._assert_conservation()
        return self._observation(),{"seed":self.seed}

    def action_masks(self):
        item=self.current_item
        if item is None: return np.zeros(self.config.num_stacks,dtype=np.int8)
        return np.asarray([int(s.free_slots>0 and item.is_compatible(s.group)) for s in self.stacks],dtype=np.int8)

    def _mask_for(self,item):
        return np.asarray([int(s.free_slots>0 and item.is_compatible(s.group)) for s in self.stacks],dtype=np.int8)

    def _prepare_next_arrival(self,initial=False):
        while self._arrival_index<len(self.episode.arrivals):
            event=self.episode.arrivals[self._arrival_index]
            self.sim_time=max(self.sim_time,event.due_time); self._process_due_retrievals()
            item=self.items[event.item_id]; self._arrival_index+=1
            if item.status!="pending": continue
            if not self._mask_for(item).any():
                item.status,item.location="failed","failed"
                self.metrics.data.failed+=1; self.metrics.data.unplaced+=1
                self._async_reward-=self.config.lambda_unplaced
                continue
            self.current_item=item; return
        self.current_item=None
        while self._retrieval_events:
            next_due=min(e.due_time for e in self._retrieval_events)
            self.sim_time=max(self.sim_time,next_due); before=len(self._retrieval_events)
            self._process_due_retrievals()
            if len(self._retrieval_events)==before: break
        self._check_terminal()

    def _process_due_retrievals(self):
        while True:
            if self.sim_time>=self.config.max_episode_steps: break
            due=[e for e in self._retrieval_events if e.due_time<=self.sim_time]
            deferred=[e for e in due if e.item_id in self._deferred]
            due.sort(key=lambda e:(e.due_time,e.priority,e.item_id))
            deferred.sort(key=lambda e:(e.due_time,e.priority,e.item_id))
            candidates=deferred+[e for e in due if e not in deferred]; selected=None
            for event in candidates:
                item=self.items[event.item_id]
                if item.status in {"retrieved","failed"}: selected=event; break
                if item.status=="pending":
                    if item.id not in self._deferred: self.metrics.data.deferred_retrievals+=1
                    self._deferred[item.id]=event; continue
                blockers=self.retrieval_engine.blocker_count(item.id)
                if item.location=="temp": blockers=0
                if self.temp_storage.free_slots>=blockers: selected=event; break
            if selected is None:
                for event in due:
                    if event.item_id not in self._deferred: self.metrics.data.deferred_retrievals+=1
                    self._deferred[event.item_id]=event
                break
            self._retrieval_events.remove(selected); item=self.items[selected.item_id]
            self._deferred.pop(item.id,None)
            if item.status in {"retrieved","failed"}: continue
            blockers,duration=self.retrieval_engine.retrieve(item,self.sim_time)
            completion_time=self.sim_time+duration
            delay=max(0,completion_time-item.expected_retrieval_time) if item.expected_retrieval_time is not None else 0
            self._async_reward+=self.config.success_reward-self.config.lambda_reshuffle*blockers-self.config.lambda_delay*delay
            self.sim_time=completion_time; self._deferred.pop(item.id,None); self._assert_conservation()

    def step(self,action):
        if self._terminated: raise RuntimeError("Episode is terminated. Call reset().")
        if self.current_item is None:
            self._check_terminal(); return self._observation(),0.0,self._terminated,False,self.get_metrics()
        item=self.current_item; mask=self.action_masks()
        if not isinstance(action,(int,np.integer)) or action<0 or action>=self.config.num_stacks or not mask[action]:
            self.metrics.data.invalid_actions+=1
            raise InvalidActionError(f"Action {action} is invalid for item {item.id}; valid mask={mask.tolist()}")
        reward=self._placement_reward(item,self.stacks[int(action)])
        self.stacks[int(action)].push(item); self.metrics.data.placements+=1
        self.current_item=None; self.sim_time+=1; self.metrics.data.elapsed_steps=self.sim_time
        self._process_due_retrievals(); self._prepare_next_arrival()
        reward+=self._async_reward; self._async_reward=0.0; self._assert_conservation()
        truncated=self.sim_time>=self.config.max_episode_steps and not self._terminated
        if truncated: self._terminated=True
        terminated=self._terminated and not truncated
        if (terminated or truncated) and not self._terminal_reward_paid: reward+=self._terminal_reward()
        self.metrics.data.elapsed_steps=self.sim_time
        return self._observation(),float(reward),terminated,truncated,self.get_metrics()

    def _placement_reward(self,item,stack):
        top=stack.top
        if top is None or item.expected_retrieval_time is None or top.expected_retrieval_time is None: return 0.0
        if item.urgency_key<=top.urgency_key: return 0.0
        return -self.config.lambda_place*(stack.height/self.config.stack_height)

    def _terminal_reward(self):
        self._terminal_reward_paid=True
        unretrieved=sum(i.status!="retrieved" for i in self.items.values())
        return -self.config.lambda_unretrieved*unretrieved-self.config.lambda_temp*self.temp_storage.count

    def _check_terminal(self):
        if self._arrival_index>=len(self.episode.arrivals) and not self._retrieval_events:
            self._terminated=True
            self.metrics.data.runtime_seconds=time.perf_counter()-self._episode_wall_start
        elif self._arrival_index>=len(self.episode.arrivals) and self._retrieval_events:
            self._retrieval_events.clear(); self._deferred.clear(); self._terminated=True
            self.metrics.data.runtime_seconds=time.perf_counter()-self._episode_wall_start

    def _assert_conservation(self):
        stored=sum(s.height for s in self.stacks); temp=self.temp_storage.count
        retrieved=sum(i.status=="retrieved" for i in self.items.values())
        failed=sum(i.status=="failed" for i in self.items.values())
        pending=sum(i.status=="pending" for i in self.items.values())
        total=stored+temp+retrieved+failed+pending
        if total!=len(self.items): raise ItemConservationError(f"Item conservation failed: {total} != {len(self.items)}")

    def _observation(self):
        cfg=self.config; grid=self.state.to_per_slot_tensor(); item=self.current_item
        if item is None: incoming=np.zeros(4,dtype=np.float32)
        else:
            tau=-1.0 if item.expected_retrieval_time is None else np.clip(item.expected_retrieval_time/cfg.max_episode_steps,-1,1)
            incoming=np.asarray([tau,item.priority/5.0,min(item.compatible_groups)/max(1,cfg.num_groups),1.0],dtype=np.float32)
        taus=[x.expected_retrieval_time/cfg.max_episode_steps for x in self.temp_storage.items if x.expected_retrieval_time is not None]
        temp=np.asarray([self.temp_storage.count/max(1,cfg.temp_capacity),min(taus) if taus else -1.0,max(taus) if taus else -1.0],dtype=np.float32)
        summaries=np.zeros((cfg.num_stacks,3),dtype=np.float32)
        for k,stack in enumerate(self.stacks):
            summaries[k]=[stack.height/cfg.stack_height,stack.free_slots/cfg.stack_height,float(item is not None and item.is_compatible(stack.group))]
        lookahead=np.zeros((cfg.lookahead_window,4),dtype=np.float32)
        if cfg.scenario=="C":
            events=[]
            for e in self._retrieval_events:
                if e.item_id not in self._deferred: events.append((e.due_time,0,e.item_id,1.0,e))
            for e in self.episode.arrivals[self._arrival_index:]:
                events.append((e.due_time,1,e.item_id,0.0,e))
            events.sort(key=lambda x:(x[0],x[1],x[2]))
            for row,(due,_,item_id,event_type,event) in enumerate(events[:cfg.lookahead_window]):
                scheduled=self.items[item_id]
                tau=-1.0 if event_type==0.0 or scheduled.expected_retrieval_time is None else scheduled.expected_retrieval_time/cfg.max_episode_steps
                priority=scheduled.priority/5.0 if event_type==1.0 else 0.0
                lookahead[row]=[event_type,np.clip((due-self.sim_time)/cfg.max_episode_steps,0,1),tau,priority]
        return {"grid":grid,"incoming_item":incoming,"temp_storage":temp,"stack_summaries":summaries,
                "schedule_lookahead":lookahead,"action_mask":self.action_masks(),
                "sim_time":np.asarray([np.clip(self.sim_time/cfg.max_episode_steps,0,1)],dtype=np.float32)}

    def get_metrics(self): return self.metrics.get_summary()
    def render(self):
        lines=[f"Warehouse time={self.sim_time} reshuffles={self.metrics.data.reshuffles}"]
        for stack in self.stacks:
            lines.append(f"Stack {stack.index} (group={stack.group}): "+" | ".join(i.id for i in stack.items))
        lines.append("Temp: "+", ".join(i.id for i in self.temp_storage.items))
        if self.current_item: lines.append(f"Incoming: {self.current_item.id}")
        output="\n".join(lines)
        if self.render_mode=="human": print(output)
        return output
