"""Statistical summaries required by the experiment protocol."""
from __future__ import annotations
import math
import statistics

class StatisticsReporter:
    @staticmethod
    def summarize(values):
        values=[float(v) for v in values]; n=len(values)
        if not n: return {"n":0,"mean":None,"median":None,"std":None,"ci95_low":None,"ci95_high":None}
        mean=statistics.mean(values); std=statistics.stdev(values) if n>1 else 0.0
        half=1.96*std/math.sqrt(n)
        return {"n":n,"mean":mean,"median":statistics.median(values),"std":std,
                "ci95_low":mean-half,"ci95_high":mean+half}
    def compare(self,results_by_algorithm,metric="reshuffles"):
        return {name:self.summarize([row[metric] for row in rows]) for name,rows in results_by_algorithm.items()}
    @staticmethod
    def paired_wilcoxon(values_a,values_b):
        try: from scipy.stats import wilcoxon
        except ImportError: return None
        if len(values_a)!=len(values_b) or len(values_a)<2: raise ValueError("Paired samples must have equal length >= 2")
        return float(wilcoxon(values_a,values_b).pvalue)
