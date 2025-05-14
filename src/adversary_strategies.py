"""
This module defines deterministic strategies that act as adversaries in the training of reinforcement learning agents.

These predefined strategies serve as opponents in the pricing game environment, playing against the agents being trained. 
They vary in complexity to expose the training agents to a diverse range of scenarios and strategic behaviours.

Author: Sahar Jahani
"""

import numpy as np

import src.globals as gl

import logging
logger = logging.getLogger(__name__)


def myopic(env, player:int, first_price:float=0):
    """
    Adversary follows the Myopic strategy.
    """
    return env.myopic(player)


def const(env, player:int, first_price:float):
    """
    Adversary follows a Constant strategy.
    """
    if env.stage == env.T - 1:
        return env.myopic(player)
    return first_price


def imit(env, player:int, first_price:float):
    """
    Adversary imitates the opponent's previous price.
    """
    if env.stage == 0:
        return first_price
    if env.stage == env.T - 1:
        return env.myopic(player)
    return env.prices[1 - player][env.stage - 1]


def fight(env, player:int, first_price:float):
    """
    Simplified fighting strategy that adjusts price based on aspiration level.
    """
    if env.stage == 0:
        return first_price
    if env.stage == env.T - 1:
        return env.myopic(player)

    aspire = [(env.total_demand - env.costs[0] + env.costs[1]) / 2,
              (env.total_demand - env.costs[1] + env.costs[0]) / 2]

    demand = env.demand_potential[player][env.stage]
    target = aspire[player]

    if demand >= target:
        return env.prices[player][env.stage - 1]

    opponent_price = env.prices[1 - player][env.stage - 1]
    new_price = opponent_price + 2 * (demand - target)

    max_aspire_price = int(0.95 * (env.total_demand + env.costs[0] + env.costs[1]) / 4)
    return min(new_price, max_aspire_price)


def fight_lb(env, player:int, first_price:float):
    """
    Lower-bound version of the fight strategy — never prices below cost.
    """
    price = env.fight(player, first_price)
    return max(price, env.costs[player])


def guess(env, player:int, first_price:float):
    """
    Predictive fighting strategy that estimates opponent's sales.
    """
    if env.stage == 0:
        env.aspire_demand = [(env.total_demand / 2 + env.costs[1] - env.costs[0]),
                             (env.total_demand / 2 + env.costs[0] - env.costs[1])]
        env.aspire_price = (env.total_demand + env.costs[0] + env.costs[1]) / 4
        env.sale_guess = [env.aspire_demand[0] - env.aspire_price,
                          env.aspire_demand[1] - env.aspire_price]
        return first_price

    if env.stage == env.T - 1:
        return env.myopic(player)

    demand = env.demand_potential[player][env.stage]
    target = env.aspire_demand[player]

    if demand >= target:
        p_mono = env.myopic(player)
        p_current = env.prices[player][env.stage - 1]
        if p_current > p_mono:
            return p_mono
        elif p_current > p_mono - 7:
            return p_current
        return 0.6 * p_current + 0.4 * (p_mono - 7)

    prev_sales = env.demand_potential[1 - player][env.stage - 1] - env.prices[1 - player][env.stage - 1]
    alpha = 0.5
    new_guess = alpha * env.sale_guess[player] + (1 - alpha) * prev_sales
    env.sale_guess[player] = new_guess

    guessed_opp_price = env.total_demand - demand - new_guess
    price = guessed_opp_price + 2 * (demand - target)

    return min(price, 125 if player == 0 else 130)


def guess2(env, player:int, first_price:float):
    """
    More cooperative variant of the guess strategy.
    """
    if env.stage == 0:
        env.aspire_demand = [(env.total_demand / 2 + env.costs[1] - env.costs[0]),
                             (env.total_demand / 2 + env.costs[0] - env.costs[1])]
        env.aspire_price = (env.total_demand + env.costs[0] + env.costs[1]) / 4
        env.sale_guess = [env.aspire_demand[0] - env.aspire_price,
                          env.aspire_demand[1] - env.aspire_price]
        return first_price

    if env.stage == env.T - 1:
        return env.myopic(player)

    demand = env.demand_potential[player][env.stage]
    target = env.aspire_demand[player]
    allowed_range = 15

    if demand >= target:
        p_mono = env.myopic(player)
        p_current = env.prices[player][env.stage - 1]
        if p_current > p_mono:
            return p_mono
        elif p_current > p_mono - allowed_range:
            return p_current
        return 0.6 * p_current + 0.4 * (p_mono - allowed_range)

    prev_sales = env.demand_potential[1 - player][env.stage - 1] - env.prices[1 - player][env.stage - 1]
    alpha = 0.5
    new_guess = alpha * env.sale_guess[player] + (1 - alpha) * prev_sales
    env.sale_guess[player] = new_guess

    guessed_opp_price = env.total_demand - demand - new_guess
    price = guessed_opp_price + 2 * (demand - target)

    return min(price, 135 if player == 0 else 140)


def guess3(env, player:int, first_price:float):
    """
    Less cooperative variant of the guess strategy.
    """
    if env.stage == 0:
        env.aspire_demand = [(env.total_demand / 2 + env.costs[1] - env.costs[0]),
                             (env.total_demand / 2 + env.costs[0] - env.costs[1])]
        env.aspire_price = (env.total_demand + env.costs[0] + env.costs[1]) / 4
        env.sale_guess = [env.aspire_demand[0] - env.aspire_price,
                          env.aspire_demand[1] - env.aspire_price]
        return first_price

    if env.stage == env.T - 1:
        return env.myopic(player)

    demand = env.demand_potential[player][env.stage]
    target = env.aspire_demand[player]
    allowed_range = 3

    if demand >= target:
        p_mono = env.myopic(player)
        p_current = env.prices[player][env.stage - 1]
        if p_current > p_mono:
            return p_mono
        elif p_current > p_mono - allowed_range:
            return p_current
        return 0.6 * p_current + 0.4 * (p_mono - allowed_range)

    prev_sales = env.demand_potential[1 - player][env.stage - 1] - env.prices[1 - player][env.stage - 1]
    alpha = 0.5
    new_guess = alpha * env.sale_guess[player] + (1 - alpha) * prev_sales
    env.sale_guess[player] = new_guess

    guessed_opp_price = env.total_demand - demand - new_guess
    price = guessed_opp_price + 2 * (demand - target)

    return min(price, 125 if player == 0 else 130)

def spe(env, player, firstprice=0):
    """
    Adversary follows the subgame perfect equilibrium strategy.
    """
    t = env.stage
    P = gl.SPE_a[t]*(env.demand_potential[player][t]-200) + gl.SPE_b[t] + gl.SPE_k[t]*(env.costs[player]-64)
    return P
