import unittest

from src.rl_lib.markov_process import NonTerminal
from typing import Callable , Sequence , Tuple
from scipy.stats import norm
import numpy as np

from src.rl_lib.plot_functions import plot_list_of_curves
from src.tdATMAmericanOptionExerciseMDP import OptimalExerciseBinTree


class TestMDPFormation ( unittest.TestCase ):

    spot_price: float
    payoff: Callable[[float, float], float]
    expiry: float
    rate: float
    vol: float
    num_steps: int
    is_call: bool
    strike: float

    def european_price(self, is_call: bool, strike: float) -> float:
        sigma_sqrt: float = self.vol * np.sqrt(self.expiry)
        d1: float = (np.log(self.spot_price / strike) +
                     (self.rate + self.vol ** 2 / 2.) * self.expiry) / sigma_sqrt
        d2: float = d1 - sigma_sqrt
        if is_call:
            ret = self.spot_price * norm.cdf(d1) - strike * np.exp(-self.rate * self.expiry) * norm.cdf(d2)
        else:
            ret = strike * np.exp(-self.rate * self.expiry) * norm.cdf(-d2) - self.spot_price * norm.cdf(-d1)
        return ret

    def setUp ( self ) -> None:
        from src.rl_lib.plot_functions import plot_list_of_curves
        self.spot_price: float = 100.0
        self.strike: float = 100.0
        self.is_call: bool = False
        self.expiry: float = 10
        self.rate: float = 0.03
        self.vol: float = 0.25
        self.num_steps: int = 10

    def testRiskyAssetReturn ( self ):
        if self.is_call:
            opt_payoff = lambda _ , x: max ( x - self.strike , 0 )
        else:
            opt_payoff = lambda _ , x: max ( self.strike - x , 0 )

        opt_ex_bin_tree: OptimalExerciseBinTree = OptimalExerciseBinTree (
            spot_price = self.spot_price ,
            payoff = opt_payoff ,
            expiry = self.expiry ,
            rate = self.rate ,
            vol = self.vol ,
            num_steps = self.num_steps
            )

        vf_seq , policy_seq = zip ( *opt_ex_bin_tree.get_opt_vf_and_policy () )
        ex_boundary: Sequence [ Tuple [ float , float ] ] = opt_ex_bin_tree.option_exercise_boundary ( policy_seq , self.is_call )
        time_pts , ex_bound_pts = zip ( *ex_boundary )
        label = ("Call" if self.is_call else "Put") + " Option Exercise Boundary"
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
            opt_payoff = lambda _ , x: max ( x - self.strike , 0 )
        else:
            opt_payoff = lambda _ , x: max ( self.strike - x , 0 )

        opt_ex_bin_tree: OptimalExerciseBinTree = OptimalExerciseBinTree (
            spot_price = self.spot_price ,
            payoff = opt_payoff ,
            expiry = self.expiry ,
            rate = self.rate ,
            vol = self.vol ,
            num_steps = self.num_steps
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