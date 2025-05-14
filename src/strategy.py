"""
This module includes the `Strategy` class, which represents a policy in the reinforcement learning pricing game. 
It serves both as an individual policy for training agents and as a pure strategy for either the low-cost or high-cost 
player in the meta-game.

This module also includes the `MixedStrategy` class, which represents a probability distribution over a set of strategies. 
This class is used to model the mixed equilibrium strategies in the meta-game.

Author: Sahar Jahani
"""

import numpy as np
import pandas as pd
from collections import namedtuple
import time
from typing import List
from enum import Enum
import src.globals as gl
import src.utils as ut
import copy


import logging
logger = logging.getLogger(__name__) 

class Strategy:
    """
    Represents a pricing strategy for the pricing game (environment).
    
    A strategy can either be static (a function) or learned (e.g., trained using SB3).
    """

    def __init__(self, strategy_type, model_or_func, name, first_price=132, memory=0, action_step=None) -> None:
        """
        Parameters:
        - strategy_type: Type of the strategy (static or SB3 model).
        - model_or_func: Function for static strategies or the SB3 model class for learned strategies.
        - name: Name of the strategy.
        - first_price: Starting price used in static strategies.
        - memory: Number of previous stages considered in the strategy.
        - action_step: Step size used when translating discrete actions to price deltas (SB3 only).
        """
        self.type = strategy_type
        self.name = name
        self.memory = memory
        self.action_step = action_step
        self.first_price = first_price

        if strategy_type == StrategyType.sb3_model:
            self.dir = f"{gl.MODELS_DIR}/{name}"
            self.model = model_or_func
            self.policy = None
        else:
            self.policy = model_or_func
            self.model = None

    def __str__(self) -> str:
        return f"{self.name}:{self.memory},{self.action_step}"

    def reset(self):
        pass

    def to_dict(self):
        return {
            'type': self.type,
            'name': self.name,
            'model_or_func': self.model if self.type == StrategyType.sb3_model else self.policy,
            'first_price': self.first_price,
            'memory': self.memory,
            'action_step': self.action_step
        }

    @classmethod
    def from_dict(cls, data_dict):
        return cls(
            strategy_type=data_dict['type'],
            model_or_func=data_dict['model_or_func'],
            name=data_dict['name'],
            first_price=data_dict['first_price'],
            memory=data_dict['memory'],
            action_step=data_dict['action_step']
        )

    def play(self, env, player=1):
        """
        Returns the price to play at the current stage of env. The environment is not updated.
        """

        if self.type == StrategyType.sb3_model:
            if self.policy is None:
                if env.memory != self.memory:
                    env_new = env.__class__(
                        tuple_costs=env.costs,
                        adversary_mixed_strategy=env.adversary_mixed_strategy,
                        memory=self.memory
                    )
                    self.policy = self.model.load(self.dir, env=env_new).predict
                else:
                    self.policy = self.model.load(self.dir, env=env).predict

            state = env.get_state(stage=env.stage, player=player, memory=self.memory)
            action, _ = self.policy(state)

            price = (env.myopic(player) - action[0]) if self.action_step is None else (
                env.myopic(player) - self.action_step * action)

            if player == 0:
                env.actions[env.stage] = action[0] if self.action_step is None else self.action_step * action

            return price

        else:
            return self.policy(env, player, self.first_price)

    def play_against(self, env, adversary: 'Strategy'):
        """
        Plays a full episode of the environment against the adversary.
        - self is player 0, adversary is player 1.
        - action_step must be set for SB3 strategies.
        
        Returns:
            Tuple: (payoff of self, payoff of adversary)
        """
        env.adversary_mixed_strategy = adversary.to_mixed_strategy()
        state, _ = env.reset()

        while env.stage < env.T:
            prices = [self.play(env, 0), adversary.play(env, 1)]
            env.update_game_variables(prices)
            env.stage += 1

        return [sum(env.profit[0]), sum(env.profit[1])]

    def to_mixed_strategy(self):
        """
        Converts this pure strategy into a mixed strategy with probability 1.
        """
        return MixedStrategy(strategies_lst=[self], probablities_lst=[1])


class MixedStrategy:
    """
    A probabilistic mixture over multiple strategies.
    """

    def __init__(self, strategies_lst, probablities_lst) -> None:
        self.strategies = strategies_lst
        self.strategy_probs = probablities_lst
        self.support_size = ut.support_count(probablities_lst)

    def choose_strategy(self):
        """
        Randomly selects a strategy based on the defined probabilities.
        """
        if self.strategies:
            strategy_ind = np.random.choice(len(self.strategies), size=1, p=self.strategy_probs)
            return self.strategies[strategy_ind[0]]
        else:
            print("Adversary's strategy could not be selected (empty strategy list).")
            return None

    def play_against(self, env, adversary):
        
        pass

    def __str__(self) -> str:
        return ",".join(
            f"{self.strategies[i].name}-{self.strategy_probs[i]:.2f}"
            for i in range(len(self.strategies)) if self.strategy_probs[i] > 0
        )
        
    def reduce(self):
        """
        Removes strategies with zero probability.
        Returns a new reduced MixedStrategy.
        """
        strts = []
        probs = []
        for i in range(len(self.strategies)):
            if self.strategy_probs[i] > 0:
                strts.append(self.strategies[i])
                probs.append(self.strategy_probs[i])
        return MixedStrategy(strategies_lst=strts, probablities_lst=probs)

    def copy_unload(self):
        """
        Returns a copy of the MixedStrategy with SB3 models unloaded.
        """
        strts = []
        probs = []
        for i in range(len(self.strategies)):
            strt = copy.deepcopy(self.strategies[i])
            if self.strategies[i].type == StrategyType.sb3_model:
                strt.policy = None
            strts.append(strt)
            probs.append(self.strategy_probs[i])

        return MixedStrategy(strategies_lst=strts, probablities_lst=probs)


class StrategyType(Enum):
    static = 0
    neural_net = 1
    sb3_model = 2