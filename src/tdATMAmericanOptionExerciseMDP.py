from dataclasses import dataclass
from typing import Callable, Tuple, Iterator, Sequence, List
import numpy as np
from src.rl_lib.dynamic_programming import V
from src.rl_lib.markov_decision_process import Terminal, NonTerminal
from src.rl_lib.policy import FiniteDeterministicPolicy
from src.rl_lib.distribution import Constant, Categorical
from src.rl_lib.finite_horizon import optimal_vf_and_policy


@dataclass(frozen=True)
class OptimalExerciseBinTree:

    spot_price: float
    payoff: Callable[[float, float], float]
    expiry: float
    rate: float
    vol: float
    num_steps: int

    def dt(self) -> float:
        return self.expiry / self.num_steps

    def state_price(self, i: int, j: int) -> float:
        return self.spot_price * np.exp((2 * j - i) * self.vol * np.sqrt(self.dt()))

    def get_opt_vf_and_policy(self) -> Iterator[Tuple[V[int], FiniteDeterministicPolicy[int, bool]]]:
        dt: float = self.dt()
        up_factor: float = np.exp(self.vol * np.sqrt(dt))
        up_prob: float = (np.exp(self.rate * dt) * up_factor - 1) / (up_factor * up_factor - 1)
        return optimal_vf_and_policy(
            steps=[
                {NonTerminal(j): {
                    True: Constant(
                        (
                            Terminal(-1),
                            self.payoff(i * dt, self.state_price(i, j))
                        )
                    ),
                    False: Categorical(
                        {
                            (NonTerminal(j + 1), 0.): up_prob,
                            (NonTerminal(j), 0.): 1 - up_prob
                        }
                    )
                } for j in range(i + 1)}
                for i in range(self.num_steps + 1)
            ],
            gamma=np.exp(-self.rate * dt)
        )

    def option_exercise_boundary(
        self,
        policy_seq: Sequence[FiniteDeterministicPolicy[int, bool]],
        is_call: bool
    ) -> Sequence[Tuple[float, float]]:
        dt: float = self.dt()
        ex_boundary: List[Tuple[float, float]] = []
        for i in range(self.num_steps + 1):
            ex_points = [j for j in range(i + 1)
                         if policy_seq[i].action_for[j] and
                         self.payoff(i * dt, self.state_price(i, j)) > 0]
            if len(ex_points) > 0:
                boundary_pt = min(ex_points) if is_call else max(ex_points)
                ex_boundary.append(
                    (i * dt, self.state_price(i, boundary_pt))
                )
        return ex_boundary


if __name__ == '__main__':
    from src.rl_lib.plot_functions import plot_list_of_curves
    spot_price_val: float = 100.0
    strike: float = 100.0
    is_call: bool = False
    expiry_val: float = 10
    rate_val: float = 0.03
    vol_val: float = 0.25
    num_steps_val: int = 10

    if is_call:
        opt_payoff = lambda _ , x: max ( x - strike , 0 )
    else:
        opt_payoff = lambda _ , x: max ( strike - x , 0 )

    opt_ex_bin_tree: OptimalExerciseBinTree = OptimalExerciseBinTree (
        spot_price = spot_price_val ,
        payoff = opt_payoff ,
        expiry = expiry_val ,
        rate = rate_val ,
        vol = vol_val ,
        num_steps = num_steps_val
        )

    vf_seq , policy_seq = zip ( *opt_ex_bin_tree.get_opt_vf_and_policy () )
    ex_boundary: Sequence [ Tuple [ float , float ] ] = opt_ex_bin_tree.option_exercise_boundary ( policy_seq , is_call )
    time_pts , ex_bound_pts = zip ( *ex_boundary )
    label = ("Call" if is_call else "Put") + " Option Exercise Boundary"
    plot_list_of_curves (
        list_of_x_vals = [ time_pts ] ,
        list_of_y_vals = [ ex_bound_pts ] ,
        list_of_colors = [ "b" ] ,
        list_of_curve_labels = [ label ] ,
        x_label = "Time" ,
        y_label = "Underlying Price" ,
        title = label
        )

    am_price: float = vf_seq [ 0 ] [ NonTerminal ( 0 ) ]
    print ( f"American Price = {am_price:.3f}" )

    is_call: bool = True
    if is_call:
        opt_payoff = lambda _ , x: max ( x - strike , 0 )
    else:
        opt_payoff = lambda _ , x: max ( strike - x , 0 )

    opt_ex_bin_tree: OptimalExerciseBinTree = OptimalExerciseBinTree (
        spot_price = spot_price_val ,
        payoff = opt_payoff ,
        expiry = expiry_val ,
        rate = rate_val ,
        vol = vol_val ,
        num_steps = num_steps_val
        )

    vf_seq , policy_seq = zip ( *opt_ex_bin_tree.get_opt_vf_and_policy () )
    ex_boundary: Sequence [ Tuple [ float , float ] ] = opt_ex_bin_tree.option_exercise_boundary ( policy_seq , is_call )
    time_pts , ex_bound_pts = zip ( *ex_boundary )
    label = ("Call" if is_call else "Put") + " Option Exercise Boundary"
    plot_list_of_curves (
        list_of_x_vals = [ time_pts ] ,
        list_of_y_vals = [ ex_bound_pts ] ,
        list_of_colors = [ "b" ] ,
        list_of_curve_labels = [ label ] ,
        x_label = "Time" ,
        y_label = "Underlying Price" ,
        title = label
        )

    am_price: float = vf_seq [ 0 ] [ NonTerminal ( 0 ) ]
    print ( f"American Price = {am_price:.3f}" )
