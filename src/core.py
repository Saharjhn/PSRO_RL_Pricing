"""
This module contains general-purpose functions that simplify the implementation of training workflows.

Author: Sahar Jahani
"""

import numpy as np
import time
from collections import namedtuple
from stable_baselines3 import SAC, PPO
import copy
from typing import List, Dict

import src.envs as envs
import src.equilibrium_solver.bimatrix as bimatrix
import src.globals as gl
from src.bimatrix import BimatrixGame
from src.strategy import StrategyType, Strategy, MixedStrategy
from src.database import DataBase, AgentRow, IterRow
import src.utils as ut

import logging
logger = logging.getLogger(__name__) 


def find_base_agents(db, alg, memory: int, cost: float, mix_strt, size: int) -> list:
    """
    Select a list of base agents to train from.
    The strategies must match the training agent class.
    Lower-cost strategies should match low-cost training cases.
    Includes some chance of selecting existing similar agents from the database.
    """
    strats = copy.deepcopy(mix_strt.strategies)
    probs = copy.deepcopy(mix_strt.strategy_probs)
    candidates = [None]
    candidate_weights = [1.00]

    # Sort strategies by descending probability
    for i in range(len(probs) - 1):
        for j in range(i + 1, len(probs)):
            if probs[i] < probs[j]:
                strats[i], strats[j] = strats[j], strats[i]
                probs[i], probs[j] = probs[j], probs[i]

    for st in strats:
        if st.type == StrategyType.sb3_model and st.memory == memory and (alg is st.model):
            candidates.append(st.name)
            candidate_weights.append(2.00)

    query = (
        f'SELECT name FROM {DataBase.AGENTS_TABLE} '
        f'WHERE cost={cost} AND memory={memory} AND alg="{str(alg)}" AND added=1 '
        f'ORDER BY id DESC LIMIT {(size - len(candidates))}'
    )

    results = db.execute_select_query(query=query)
    if results is not None:
        for row in results:
            candidates.append(row[0])
            candidate_weights.append(1.00)

    weights = np.array(candidate_weights)
    weights /= np.sum(weights)
    selected_agents = np.random.choice(np.array(candidates), size, replace=True, p=weights)
    return selected_agents


def find_base_agent(db, alg, cost: float, own_mix_strt) -> str:
    """
    Select a single base agent to train from.
    Similar to `find_base_agents` but returns one agent.
    """
    strats = copy.deepcopy(own_mix_strt.strategies)
    probs = copy.deepcopy(own_mix_strt.strategy_probs)
    candidates = [None]
    candidate_weights = [1.00]

    # Sort strategies by descending probability
    for i in range(len(probs) - 1):
        for j in range(i + 1, len(probs)):
            if probs[i] < probs[j]:
                strats[i], strats[j] = strats[j], strats[i]
                probs[i], probs[j] = probs[j], probs[i]

    for st in strats:
        if st.type == StrategyType.sb3_model and (alg is st.model):
            candidates.append(st.name)
            candidate_weights.append(1.00)

    query = (
        f'SELECT name FROM {DataBase.AGENTS_TABLE} '
        f'WHERE cost={cost} AND alg="{name_of(alg)}" AND added=1'
    )

    results = db.execute_select_query(query=query)
    if results is not None:
        for row in results:
            candidates.append(row[0])
            candidate_weights.append(1.00)

    weights = np.array(candidate_weights)
    weights /= np.sum(weights)
    selected_agent = np.random.choice(np.array(candidates), p=weights)
    return selected_agent


def read_matrices_from_file(file_name: str) -> tuple:
    """Read two matrices (A and B) from a text file."""
    lines = [*open(file=file_name)]
    size = tuple(int(num) for num in lines[0].split())
    matrix_A = np.zeros(size)
    matrix_B = np.zeros(size)

    for i in range(2, 2 + size[0]):
        matrix_A[i - 2] = [float(num) for num in lines[i].split()]
    for i in range(3 + size[0], 3 + 2 * size[0]):
        matrix_B[i - 3 - size[0]] = [float(num) for num in lines[i].split()]

    return matrix_A, matrix_B




TrainInputRow = namedtuple(
    "TrainInputRow",
    "id, seed, job_name, env, base_agent, alg, alg_params, adv_mixed_strategy, target_payoff, db, num_ep_coef, equi_id"
)


def new_train(inputs: TrainInputRow):
    """
    Train one agent against the adversary. If its expected payoff is higher than the target NE payoff,
    return the new strategy and payoff to be added to the game.
    """
    id, seed, job_name, env, base_agent, alg, alg_params, adv_mixed_strategy, target_payoff, db, num_ep_coef, equi_id = inputs

    gl.initialize()

    model_name = f"{job_name}-{str(seed)}"
    models_dir = f"{gl.MODELS_DIR}/{model_name}"
    log_dir = f"{gl.LOG_DIR}/{model_name}"
    
    alg_ep_coef = 3 if name_of(alg) == name_of(PPO) else 1

    acceptable = False
    if base_agent is None:
        number_episodes = int(gl.N_EPISODES_BASE * num_ep_coef) * alg_ep_coef
        model = alg(
            'MlpPolicy', env, tensorboard_log=log_dir, seed=seed, gamma=gl.GAMMA, **alg_params
        )
    else:
        number_episodes = int(gl.N_EPISODES_LOAD * num_ep_coef) * alg_ep_coef
        base_agent_dir = f"{gl.MODELS_DIR}/{base_agent}"
        model = alg.load(
            base_agent_dir, env, tensorboard_log=log_dir, gamma=gl.GAMMA, seed=seed, **alg_params
        )

    start = time.time()
    model.learn(total_timesteps=number_episodes, tb_log_name=model_name)
    model.save(models_dir)
    running_time = time.time() - start

    model_strategy = Strategy(
        strategy_type=StrategyType.sb3_model,
        model_or_func=alg,
        name=model_name,
        action_step=env.action_step,
        memory=env.memory
    )

    iter_rows = []
    agent_payoffs = np.zeros(len(adv_mixed_strategy.strategies))
    adv_payoffs = np.zeros(len(adv_mixed_strategy.strategies))
    expected_payoff = 0
    expected_payoff_std = 0

    for strategy_index in range(len(adv_mixed_strategy.strategies)):
        if adv_mixed_strategy.strategy_probs[strategy_index] > 0:
            payoffs = []
            for _ in range(gl.NUM_STOCHASTIC_ITER):
                payoffs.append(model_strategy.play_against(
                    env=env, adversary=adv_mixed_strategy.strategies[strategy_index])
                )

                iter_row = IterRow(
                    agent_id=None,
                    adv=env.adversary_strategy.name,
                    agent_return=sum(env.profit[0]),
                    adv_return=sum(env.profit[1]),
                    agent_rewards=str(env.profit[0]),
                    adv_rewards=str(env.profit[1]),
                    actions=str(env.actions),
                    agent_prices=str(env.prices[0]),
                    adv_prices=str(env.prices[1]),
                    agent_demands=str(env.demand_potential[0]),
                    adv_demands=str(env.demand_potential[1])
                )

                iter_rows.append(iter_row)

            std_payoffs = np.array(payoffs).std(axis=0)
            mean_payoffs = np.array(payoffs).mean(axis=0)

            agent_payoffs[strategy_index] = mean_payoffs[0]
            adv_payoffs[strategy_index] = mean_payoffs[1]
            expected_payoff += agent_payoffs[strategy_index] * adv_mixed_strategy.strategy_probs[strategy_index]
            expected_payoff_std += std_payoffs[0] * adv_mixed_strategy.strategy_probs[strategy_index]

    acceptable = expected_payoff > target_payoff

    agent_id = db.insert_new_agent(AgentRow(
        model_name, base_agent, number_episodes, env.costs[0], str(adv_mixed_strategy),
        name_of(alg), seed, 1, running_time, expected_payoff_std, [expected_payoff,
        target_payoff], acceptable, equi_id
    ))

    if acceptable:
        tuple_list = []
        for row in iter_rows:
            if len(tuple_list) < gl.DB_ITER_LIMIT:
                row.agent_id=agent_id
                tuple_list.append(row)
        db.insert_many_new_iters(tuple_list)

    return (id, acceptable, model_strategy.name, agent_payoffs, adv_payoffs, expected_payoff)


def equi_sort_social_welfare(equis: List[bimatrix.Equi]) -> List[bimatrix.Equi]:
    for i in range(len(equis)):
        for j in range(i + 1, len(equis)):
            if (equis[i].row_payoff + equis[i].col_payoff) < (equis[j].row_payoff + equis[j].col_payoff):
                equis[i], equis[j] = equis[j], equis[i]
    return equis


def get_coop_equilibria(bimatrix_game: BimatrixGame, num_trace: int, db: DataBase) -> Dict[bimatrix.Equi, int]:
    """
    Computes the equilibria of bimatrix game and selects most cooperative ones,
    also writes all of them in db. Returns a dictionary of equis with key as equi and value as id in db.
    """
    all_equilibria = bimatrix_game.compute_equilibria(num_trace=num_trace)
    num_selected_equis = min(len(all_equilibria), gl.NUM_TRACE_EQUILIBRIA)
    all_equilibria = equi_sort_social_welfare(all_equilibria)

    equi_ids = {}
    for i, equi in enumerate(all_equilibria):
        iid = db.insert_new_equi(
            game_size=bimatrix_game.size(),
            freq=(equi.found / float(num_trace)),
            low_strategy_txt=str(equi.row_probs),
            high_strategy_txt=str(equi.col_probs),
            low_payoff=equi.row_payoff,
            high_payoff=equi.col_payoff,
            used=0,
            num_new_low=0,
            num_new_high=0
        )
        if i < num_selected_equis:
            equi_ids[equi] = iid

    return equi_ids


def name_of(alg) -> str:
    """Returns name of training model"""
    if alg is SAC:
        return 'SAC'
    elif alg is PPO:
        return 'PPO'
    else:
        return str(alg)


def match_updated_size(main_game: BimatrixGame, old_adv_mixed_strt: MixedStrategy, own_cost: int, own_payoff: np.array, adv_payoff: np.array):
    """
    When the mixed_adv strategies can be expanded by adding other strategies, this method adds zeros to
    payoff lists to make them same size as rows or cols in main_game and also updates the adv_mixed_strategy
    by adding prob 0 for new ones.

    Attention: payoffs should be computed later.

    Returns new_adv_mixed_strt, new_own_payoff, new_adv_payoff that are ready to be added as rows or columns to the main_game.
    """
    if own_cost == gl.LOW_COST:  # the adv is high_cost
        if len(main_game.high_strategies) == len(old_adv_mixed_strt.strategy_probs):
            pass
        elif (extra_strts := len(main_game.high_strategies) - len(old_adv_mixed_strt.strategy_probs)) > 0:
            old_adv_mixed_strt.strategies = main_game.high_strategies.copy()
            for _ in range(extra_strts):
                old_adv_mixed_strt.strategy_probs.append(0)
        else:
            ut.prt("Error in match updated size, more strategies in adv_strategy than the game_high_strts")
            raise ValueError
    elif own_cost == gl.HIGH_COST:  # the adv is low_cost
        if len(main_game.low_strategies) == len(old_adv_mixed_strt.strategy_probs):
            pass
        elif (extra_strts := len(main_game.low_strategies)-len(old_adv_mixed_strt.strategy_probs)) > 0:
            old_adv_mixed_strt.strategies = main_game.low_strategies.copy()
            for _ in range(extra_strts):
                old_adv_mixed_strt.strategy_probs.append(0)
        else:
            ut.prt("Error in match updated size, more strategies in adv_strategy than the game_low_strts")
            raise ValueError
    if (extra:=len(old_adv_mixed_strt.strategies)-len(own_payoff))>0:
        own_payoff=np.append(own_payoff,[0]*(extra))
        adv_payoff=np.append(adv_payoff,[0]*(extra))
    if len(own_payoff)!= len(old_adv_mixed_strt.strategies):
        print("Error in match updated size ")
    return old_adv_mixed_strt, own_payoff, adv_payoff


 