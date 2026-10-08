"""
Reward calculation:  Reward = ServiceBenefit - OperatorCost - lambda * C_transition

Every component is returned separately; nothing is hidden.

ServiceBenefit (dimensionless, usually negative):
    + W_DIRECT    * direct_share
    - W_TRAVEL    * avg_generalised_time / TRAVEL_TIME_NORMALISER
        avg_generalised_time = (IVT + wait + transfer penalties
                                + unserved * UNSERVED_PENALTY_MINUTES) / total demand
    - W_UNSERVED  * unserved_share
    - W_TRANSFER  * transfer_share
    - W_OVERLOAD  * overload_share       (passengers above BUS_CAPACITY * f)
OperatorCost:
    W_FLEET * buses_used / MAX_FLEET + W_ROUTE_TIME * sum(round-trip time) / normaliser
TransitionComponent:
    lambda * C_transition(P_{t-1}, P_t)

ASSUMPTION: an hour with zero total demand has ServiceBenefit = 0.
"""

from __future__ import annotations

from typing import Dict, Sequence

from ..config import ExperimentConfig
from .passenger_assignment import AssignmentResult
from .transition_cost import TransitionResult


class RewardCalculator:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

    def service_terms(self, a: AssignmentResult) -> Dict[str, float]:
        cfg, D = self.cfg, a.total_demand
        if D <= 0:
            return {"direct_share": 0.0, "avg_generalised_time_incl_unserved": 0.0,
                    "unserved_share": 0.0, "transfer_share": 0.0, "overload_share": 0.0,
                    "service_component": 0.0}
        gen_time = (a.total_in_vehicle_time + a.total_waiting_time + a.total_transfer_penalty_time
                    + a.unserved_passengers * cfg.unserved_penalty_minutes) / D
        terms = {
            "direct_share": a.direct_passengers / D,
            "avg_generalised_time_incl_unserved": gen_time,
            "unserved_share": a.unserved_passengers / D,
            "transfer_share": a.transfer_passengers / D,
            "overload_share": a.overload_passengers / D,
        }
        terms["service_component"] = (
            cfg.w_service_direct * terms["direct_share"]
            - cfg.w_service_travel_time * gen_time / cfg.travel_time_normaliser_minutes
            - cfg.w_service_unserved * terms["unserved_share"]
            - cfg.w_service_transfer * terms["transfer_share"]
            - cfg.w_service_overload * terms["overload_share"]
        )
        return terms

    def operator_terms(self, fleet_used: int, round_trip_times: Sequence[float]) -> Dict[str, float]:
        cfg = self.cfg
        total_rt = float(sum(round_trip_times))
        return {
            "fleet_used": fleet_used,
            "total_round_trip_minutes": total_rt,
            "operator_component": cfg.w_operator_fleet * fleet_used / cfg.max_fleet
            + cfg.w_operator_route_time * total_rt / cfg.route_time_normaliser_minutes,
        }

    def compute(self, assignment: AssignmentResult, fleet_used: int,
                round_trip_times: Sequence[float], transition: TransitionResult) -> Dict[str, float]:
        s = self.service_terms(assignment)
        o = self.operator_terms(fleet_used, round_trip_times)
        transition_component = self.cfg.transition_penalty_lambda * transition.total
        total = s["service_component"] - o["operator_component"] - transition_component
        return {
            **s,
            **o,
            "transition_cost": transition.total,
            "transition_component": transition_component,
            "service_component": s["service_component"],
            "operator_component": o["operator_component"],
            "total_reward": total,
        }
