"""
This module defines the pricing game environments used in the reinforcement learning framework, 
including both continuous and discrete variants. These environments inherit from and conform to the 
standard `Env` interface in the Gymnasium package, as required by reinforcement learning algorithms 
in the Stable-Baselines3 framework. They serve as the core environments for training agents.

Author: Sahar Jahani
"""

from enum import Enum
import numpy as np 
import gymnasium as gym
from gymnasium import spaces
import src.globals as gl
import src.utils as ut

import logging
logger = logging.getLogger(__name__) 

class ConPricingGame(gym.Env):
    """
    Continuous Pricing Game environment for a two-player pricing simulation.
    The agent competes against an adversary in a multi-stage game.
    """

    def __init__(self, tuple_costs, adversary_mixed_strategy, memory):
        super().__init__()

        self.action_step = None  # Only used by discrete version

        self.total_demand = gl.TOTAL_DEMAND
        self.costs = tuple_costs
        self.T = gl.TOTAL_STAGES
        self.demand_potential = None  # Demand potential for both players over time
        self.prices = None            # Price history for both players
        self.profit = None            # Profit history for both players
        self.stage = None             # Current stage in the game
        self.done = False

        self.adversary_mixed_strategy = adversary_mixed_strategy
        self.memory = memory  # Number of past stages included in the observation
        self.reward_division = gl.REWARDS_DIVISION_CONST

        self.action_space = spaces.Box(low=0, high=gl.CON_ACTIONS_RANGE, shape=(1,))

        # Observation includes: [stage, own demand] + own price history + adversary price history
        self.observation_space = spaces.Box(
            low=0, high=self.total_demand, shape=(2 + 2 * memory,)
        )

    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state.
        """
        super().reset(seed=seed)
        self.resetGame()
        self.adversary_strategy = self.adversary_mixed_strategy.choose_strategy()
        observation = self.get_state(stage=0)
        return observation, {}

    def resetGame(self):
        """
        Initializes memory for prices, profits, and demand potential.
        """
        self.episodesMemory = []
        self.stage = 0
        self.done = False
        self.demand_potential = [[0] * (self.T + 1), [0] * (self.T + 1)]
        self.prices = [[0] * self.T, [0] * self.T]
        self.myopic_prices = [[0] * self.T, [0] * self.T]
        self.profit = [[0] * self.T, [0] * self.T]
        self.demand_potential[0][0] = self.demand_potential[1][0] = self.total_demand / 2
        self.actions = [0] * self.T

    def get_state(self, stage, player=0, memory=None):
        """
        Constructs the current observation state for a given player.
        Returns: [stage, own demand, own price memory, adversary price memory]
        """
        mem_len = memory if memory is not None else self.memory
        stage_part = [stage]
        self_mem, adv_mem = [], []

        if stage == 0:
            if mem_len > 0:
                self_mem = [0] * mem_len
                adv_mem = [0] * mem_len
        else:
            self_mem = [0] * mem_len
            adv_mem = [0] * mem_len
            j = mem_len - 1
            for i in range(stage - 1, max(-1, stage - 1 - mem_len), -1):
                adv_mem[j] = self.prices[1 - player][i]
                self_mem[j] = self.prices[player][i]
                j -= 1

        observation = stage_part + [self.demand_potential[player][stage]] + self_mem + adv_mem
        return np.array(observation)

    def step(self, action):
        """
        Performs one step in the environment using the agent's and adversary's actions.
        """
        self.actions[self.stage] = action[0]
        adversary_action = self.adversary_strategy.play(env=self, player=1)
        self.update_game_variables([self.myopic() - action[0], adversary_action])

        done = self.stage == self.T - 1
        reward = self.profit[0][self.stage]
        self.stage += 1

        return self.get_state(stage=self.stage), reward, done, False, {}

    def update_game_variables(self, price_pair):
        """
        Updates the environment with the new prices and calculates profits and next demand.
        """
        for player in [0, 1]:
            price = price_pair[player]
            # Ensure price is within bounds
            price = max(price, self.costs[player])
            price = min(price, self.demand_potential[player][self.stage])
            self.prices[player][self.stage] = price
            profit = (self.demand_potential[player][self.stage] - price) * (price - self.costs[player])
            self.profit[player][self.stage] = profit / self.reward_division

        for player in [0, 1]:
            if self.stage < self.T - 1:
                delta = (self.prices[1 - player][self.stage] - self.prices[player][self.stage]) / 2
                self.demand_potential[player][self.stage + 1] = self.demand_potential[player][self.stage] + delta

    def myopic(self, player=0):
        """
        Calculates the myopic price for the player.
        """
        return (self.demand_potential[player][self.stage] + self.costs[player]) / 2

    def render(self):
        pass

    def close(self):
        pass


class DisPricingGame(ConPricingGame):
    """
    Discrete action version of the pricing game.
    """

    def __init__(self, tuple_costs, adversary_mixed_strategy, memory):
        super().__init__(tuple_costs, adversary_mixed_strategy, memory)
        self.action_step = gl.ACTION_STEP
        self.action_space = spaces.Discrete(gl.NUM_ACTIONS)

    def step(self, action):
        """
        Performs one step using discretized pricing actions.
        """
        self.actions[self.stage] = action
        adversary_action = self.adversary_strategy.play(env=self, player=1)
        agent_price = self.myopic() - (action * self.action_step)
        self.update_game_variables([agent_price, adversary_action])

        done = self.stage == self.T - 1
        reward = self.profit[0][self.stage]
        self.stage += 1

        return self.get_state(stage=self.stage), reward, done, False, {}

def str_to_envclass(s: str):
    """Convert string to corresponding environment class."""
    if s == ConPricingGame.__name__:
        return ConPricingGame
    elif s == DisPricingGame.__name__:
        return DisPricingGame
    else:
        return None