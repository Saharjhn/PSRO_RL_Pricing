"""
This module implements the meta-game and its extensions using the PSRO (Policy-Space Response Oracles) framework.

The meta-game is modeled as a Bimatrix Game where:
- The row player (Player 0) represents low-cost agents.
- The column player (Player 1) represents high-cost agents.

Both players consist of reinforcement learning agents that have passed a threshold level of return 
(based on equilibrium payoffs) in the pricing game environment.

As new agents are trained and meet the payoff threshold, they are added to the meta-game as new strategies 
corresponding to their cost level. The meta-game is extended iteratively in this manner.

Nash equilibria of the evolving bimatrix game are computed using the Lemke algorithm, 
with the implementation by Prof. Bernhard von Stengel (source files located in the `equilibrium_solver` folder).

New agents are then trained as best responses to the updated equilibrium strategy distributions. 
If these new agents reach the expected equilibrium payoff, they are included in the meta-game, continuing the PSRO loop.

Author: Sahar Jahani
"""

import numpy as np
import time
from typing import List
import os
import pickle

import src.globals as gl
import src.equilibrium_solver.bimatrix as bimatrix
import src.utils as ut
from src.strategy import StrategyType, Strategy, MixedStrategy
from src.envs import str_to_envclass

import logging
logger = logging.getLogger(__name__) 

class BimatrixGame():
    """
    This class represents the bimatrix meta-game between low-cost and high-cost agents. Strategies play against each other in the pricing game and fill the matrices of payoffs.
    Then the equilibria are computed using the Lemke algorithm.
    """

    def __init__(self, low_cost_strategies, high_cost_strategies, env_class, game_name) -> None:
        self.low_strategies = low_cost_strategies
        self.high_strategies = high_cost_strategies
        self.env_class = env_class
        self.game_name = game_name

    def size(self):
        return (len(self.low_strategies), len(self.high_strategies))

    def get_subgame(self, num_row, num_col):
        sub_game = BimatrixGame(
            low_cost_strategies=self.low_strategies[:num_row],
            high_cost_strategies=self.high_strategies[:num_col],
            env_class=self.env_class
        )
        sub_game.matrix_A = self.matrix_A[:num_row, :num_col]
        sub_game.matrix_B = self.matrix_B[:num_row, :num_col]
        return sub_game

    def to_dict(self):
        return {
            'low_strategies': [strt.to_dict() for strt in self.low_strategies],
            'high_strategies': [strt.to_dict() for strt in self.high_strategies],
            'env_class': self.env_class.__name__,
            'matrix_A': self.matrix_A,
            'matrix_B': self.matrix_B,
            'game_name':self.game_name
        }

    @classmethod
    def from_dict(cls, data_dict):
        env_class = str_to_envclass(data_dict['env_class'])
        low_strategies = [Strategy.from_dict(strt_data) for strt_data in data_dict['low_strategies']]
        high_strategies = [Strategy.from_dict(strt_data) for strt_data in data_dict['high_strategies']]

        obj = cls(
            low_cost_strategies=low_strategies,
            high_cost_strategies=high_strategies,
            env_class=env_class,
            game_name=data_dict['game_name']
        )
        obj.matrix_A = data_dict['matrix_A']
        obj.matrix_B = data_dict['matrix_B']
        return obj

    def save_game(self, file_name):
        with open(f"{file_name}.pickle", "wb") as file:
            pickle.dump(self.to_dict(), file)

    def load_game(file_name):
        name = f"{file_name}.pickle"
        if os.path.exists(name):
            with open(name, "rb") as file:
                instance_data = pickle.load(file)
                return BimatrixGame.from_dict(instance_data)
        else:
            return None

    def reset_matrix(self):
        self.matrix_A = np.zeros((len(self.low_strategies), len(self.high_strategies)))
        self.matrix_B = np.zeros((len(self.low_strategies), len(self.high_strategies)))

    def fill_matrix(self):
        self.reset_matrix()
        for low in range(len(self.low_strategies)):
            for high in range(len(self.high_strategies)):
                self.update_matrix_entry(low, high)

    def update_matrix_entry(self, low_index, high_index):
        strt_L = self.low_strategies[low_index]
        strt_H = self.high_strategies[high_index]

        strt_L.reset()
        strt_H.reset()

        env = self.env_class(
            tuple_costs=(gl.LOW_COST, gl.HIGH_COST),
            adversary_mixed_strategy=strt_H.to_mixed_strategy(),
            memory=strt_L.memory
        )
        payoffs = [strt_L.play_against(env, strt_H) for _ in range(gl.NUM_MATRIX_ITER)]
        mean_payoffs = np.mean(np.array(payoffs), axis=0)

        self.matrix_A[low_index][high_index] = mean_payoffs[0]
        self.matrix_B[low_index][high_index] = mean_payoffs[1]

    def write_all_matrix(self):
        output = f"{len(self.matrix_A)} {len(self.matrix_A[0])}\n\n"
        int_output = f"{len(self.matrix_A)} {len(self.matrix_A[0])}\n\n"
        game_name = f"game_{self.game_name}"

        self.save_game(game_name)

        for matrix in [self.matrix_A, self.matrix_B]:
            for i in range(len(self.matrix_A)):
                for j in range(len(self.matrix_A[0])):
                    output += f"{matrix[i][j]:8.3f} "
                    int_output += f"{matrix[i][j]:5.0f} "
                output += "\n"
                int_output += "\n"
            output += "\n"
            int_output += "\n"

        with open(f"{game_name}.txt", "w") as out:
            out.write(output)

        int_output += "\nlow-cost strategies: \n"
        for strt in self.low_strategies:
            int_output += f" {strt.name} "

        int_output += "\nhigh-cost strategies: \n"
        for strt in self.high_strategies:
            int_output += f" {strt.name} "

        with open(f"games/game_{self.game_name}_{int(time.time())}.txt", "w") as out:
            out.write(int_output)

    def add_low_cost_row(self, row_A, row_B):
        self.matrix_A = np.append(self.matrix_A, [row_A], axis=0)
        self.matrix_B = np.append(self.matrix_B, [row_B], axis=0)

    def add_high_cost_col(self, colA, colB):
        self.matrix_A = np.hstack((self.matrix_A, np.atleast_2d(colA).T))
        self.matrix_B = np.hstack((self.matrix_B, np.atleast_2d(colB).T))

    def compute_equilibria(self, num_trace=100, write_matrix=True, prt_progress=True) -> List[bimatrix.Equi]:
        """
        Returns a list of Equi — all the equilibria found for the bimatrix game.
        """
        if write_matrix:
            self.write_all_matrix()

        game = bimatrix.bimatrix(f"game_{self.game_name}.txt")
        equilibria_all = game.tracing(num_trace)

        if prt_progress:
            self.prt(f"\nall equilibria: {len(equilibria_all)}")
            for i, eq in enumerate(equilibria_all):
                self.prt(f"{i} - {eq}")

        return equilibria_all
    def prt(self,string):
        """
        Writing the progres into a file 
        """
        with open(f'progress_{self.game_name}.txt', 'a') as file:
            file.write("\n"+string)
