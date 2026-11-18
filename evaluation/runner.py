"""Run an agent over reproducible episode seeds."""
import time
class EpisodeRunner:
    def __init__(self,env_factory): self.env_factory=env_factory
    def run(self,agent,seeds):
        results=[]
        for seed in seeds:
            env=self.env_factory(); observation,_=env.reset(seed=int(seed)); agent.reset()
            terminated=truncated=False; start=time.perf_counter()
            while not (terminated or truncated):
                mask=observation["action_mask"]
                if not mask.any():
                    observation,_,terminated,truncated,_=env.step(0); continue
                action=agent.select_action(observation,mask)
                observation,_,terminated,truncated,_=env.step(action)
            metrics=env.get_metrics(); metrics["seed"]=int(seed)
            metrics["wall_clock_seconds"]=time.perf_counter()-start; results.append(metrics)
        return results
