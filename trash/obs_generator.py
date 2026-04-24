"""兼容入口：请优先使用 ``marl.obs.generator``。"""
from marl.obs.generator import CandidateColumns, TODCObservationGenerator, alive_pid_eid_sets

__all__ = ["TODCObservationGenerator", "alive_pid_eid_sets", "CandidateColumns"]
