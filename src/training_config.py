
import numpy as np
from stable_baselines3 import SAC, PPO

import src.globals as gl
import src.core as cl

from src.strategy import StrategyType, Strategy, MixedStrategy
from src.bimatrix import BimatrixGame
import src.adversary_strategies as ad
import src.utils as ut


import os
import multiprocessing as mp

from enum import Enum

import logging
logger = logging.getLogger(__name__) 

alg_params = {
    'SAC': {
        'learning_rate': 0.0003,
        'target_entropy': 'auto',
        'ent_coef': 'auto',
        'tau': 0.010,
        'train_freq': 1,
        'gradient_steps': 1,
        'verbose': 0,
        'buffer_size': 200_000
    },
    'PPO': {
        'learning_rate': 0.00016,
        'n_epochs': 10,
        'clip_range': 0.3,
        'clip_range_vf': None,
        'ent_coef': 0.010,
        'vf_coef': 0.5,
        'verbose': 0
    }
}



class ProcessInd(Enum):
    SAClow = 0
    PPOlow = 1
    SAChigh = 2
    PPOhigh = 3


class StartMode(Enum):
    """ double oracle game strting point, start from myopic-const-guess or from a random model or different strategies similar to guess"""
    myopicConstGuess = 0
    random = 1
    multiGuess = 2
    allVsSpe=3
    
def len_initial_game(start_mode:StartMode)->int:
    if start_mode==StartMode.myopicConstGuess or start_mode==StartMode.multiGuess:
        return 3
    elif start_mode==StartMode.random:
        return 1
    elif start_mode==StartMode.allVsSpe:
        return 7
    else:
        raise ValueError("len of start_mode not implemented!")
        


def initial_matrix(env_class, start_mode, database, game_name):
    """ returns double oracle game with strategies from last stopping point but the matrix and strategies are not loaded , creates the base matrix and adds the trained strategies"""
    
    
    
    if start_mode == StartMode.myopicConstGuess:

        strt1 = Strategy(
            StrategyType.static, model_or_func=ad.myopic, name="myopic")
        strt2 = Strategy(
            StrategyType.static, model_or_func=ad.const, name="const", first_price=132)
        strt3 = Strategy(
            StrategyType.static, model_or_func=ad.guess, name="guess", first_price=132)

        init_low = [strt1, strt2, strt3]
        init_high = [strt1, strt2, strt3]
    elif start_mode == StartMode.allVsSpe:
        strt0 = Strategy(
            StrategyType.static, model_or_func=ad.spe, name="spe", first_price=132)
        strt1 = Strategy(
            StrategyType.static, model_or_func=ad.myopic, name="myopic")
        strt2 = Strategy(
            StrategyType.static, model_or_func=ad.const, name="const", first_price=132)
        strt3 = Strategy(
            StrategyType.static, model_or_func=ad.imit, name="imit", first_price=132)
        
        strt4 = Strategy(
            StrategyType.static, model_or_func=ad.guess, name="normal_guess",first_price=132)
        strt5 = Strategy(
            StrategyType.static, model_or_func=ad.guess2, name="coop_guess", first_price=132)
        strt6 = Strategy(
            StrategyType.static, model_or_func=ad.guess3, name="compete_guess", first_price=132)

        init_low = [strt0,strt1, strt2, strt3,strt4,strt5,strt6]
        init_high = [strt0,strt1, strt2, strt3,strt4,strt5,strt6]
    elif start_mode == StartMode.random:
        model_name = f"rndstart_{game_name}"
        log_dir = f"{gl.LOG_DIR}/{model_name}"
        model_dir = f"{gl.MODELS_DIR}/{model_name}"
        if not os.path.exists(f"{model_dir}.zip"):
            # tuple_costs and others are none just to make sure no play is happening here
            train_env = env_class(tuple_costs=None, adversary_mixed_strategy=None, memory=gl.MEMORY)
            model = SAC('MlpPolicy', train_env,
                        verbose=0, tensorboard_log=log_dir, gamma=gl.GAMMA, target_entropy=0)
            model.save(model_dir)

        strt_rnd = cl.Strategy(strategy_type=cl.StrategyType.sb3_model,
                               model_or_func=SAC, name=model_name, action_step=None, memory=gl.MEMORY)

        init_low = [strt_rnd]
        init_high = [strt_rnd]
    elif start_mode==StartMode.multiGuess:
        strt1 = cl.Strategy(
            cl.StrategyType.static, model_or_func=ad.guess, name="normal_guess",first_price=132)
        strt2 = cl.Strategy(
            cl.StrategyType.static, model_or_func=ad.guess2, name="coop_guess", first_price=132)
        strt3 = cl.Strategy(
            cl.StrategyType.static, model_or_func=ad.guess3, name="compete_guess", first_price=132)

        init_low = [strt1, strt2, strt3]
        init_high = [strt1, strt2, strt3]
    else:
        raise("Error: initial_matrix mode not implemented!")
    

    low_strts, high_strts = database.get_list_of_added_strategies(action_step=None, memory=gl.MEMORY)
    return cl.BimatrixGame(
        low_cost_strategies=init_low+low_strts, high_cost_strategies=init_high+high_strts, env_class=env_class,game_name=game_name)


def get_proc_input(seed, proc_ind: ProcessInd, low_mixed_strt, high_mixed_strt, target_payoffs, job_name, env_class,num_ep_coef,equi_id,db) -> cl.TrainInputRow:
    """
    Creates the input tuple for new_train method, to use in multiprocessing
    """
    # input=(id, seed, job_name, env, base_agent, alg, alg_params, adv_mixed_strategy,target_payoff, db)
    if proc_ind == ProcessInd.PPOlow or proc_ind == ProcessInd.SAClow:
        costs = [gl.LOW_COST, gl.HIGH_COST]
        own_strt = low_mixed_strt.copy_unload()
        adv_strt = high_mixed_strt.copy_unload()
        payoff = target_payoffs[0]
    elif proc_ind == ProcessInd.PPOhigh or proc_ind == ProcessInd.SAChigh:
        costs = [gl.HIGH_COST, gl.LOW_COST]
        own_strt = high_mixed_strt.copy_unload()
        adv_strt = low_mixed_strt.copy_unload()
        payoff = target_payoffs[1]

    if proc_ind == ProcessInd.SAChigh or proc_ind == ProcessInd.SAClow:
        alg = SAC
    elif proc_ind == ProcessInd.PPOhigh or proc_ind == ProcessInd.PPOlow:
        alg = PPO

    iid = proc_ind.value
    env = env_class(tuple_costs=costs, adversary_mixed_strategy=adv_strt, memory=gl.MEMORY)
    base_agent = cl.find_base_agent(db, alg, costs[0], own_strt)
    return cl.TrainInputRow(iid, seed+iid, job_name, env, base_agent, alg, alg_params[cl.name_of(alg)], adv_strt, payoff, db,num_ep_coef,equi_id)


def create_directories() -> None:
    """Create necessary directories if they do not already exist."""
    if not os.path.exists(gl.MODELS_DIR):
        os.makedirs(gl.MODELS_DIR)
    if not os.path.exists(gl.LOG_DIR):
        os.makedirs(gl.LOG_DIR)
    if not os.path.exists(gl.GAMES_DIR):
        os.makedirs(gl.GAMES_DIR)

    
def load_latest_game(game_data_name: str, new_game) -> BimatrixGame:
    """
    Load the game from saved data and update it with any new strategies.
    Preserves trained strategies that were not saved.
    """
    old_game = BimatrixGame.load_game(game_data_name)
    if old_game is None:
        new_game.reset_matrix()
        new_game.fill_matrix()
        return new_game

    new_lows = new_game.low_strategies
    new_highs = new_game.high_strategies

    # Identify start of extra low strategies
    low_trained_i = next((i for i, s in enumerate(old_game.low_strategies) if s.type == StrategyType.sb3_model), len(old_game.low_strategies))
    low_new_trained_i = next((i for i, s in enumerate(new_lows) if s.type == StrategyType.sb3_model), len(new_lows))
    low_extra_start = len(old_game.low_strategies) - low_trained_i - 1 + low_new_trained_i
    while low_extra_start >= 0 and new_lows[low_extra_start].name != old_game.low_strategies[-1].name:
        low_extra_start -= 1

    for i in range(low_extra_start + 1, len(new_lows)):
        old_game.low_strategies.append(new_lows[i])
        n = len(old_game.high_strategies)
        old_game.add_low_cost_row(np.zeros(n), np.zeros(n))
        for j in range(n):
            old_game.update_matrix_entry(len(old_game.low_strategies) - 1, j)

    # Identify start of extra high strategies
    high_trained_i = next((i for i, s in enumerate(old_game.high_strategies) if s.type == StrategyType.sb3_model), len(old_game.high_strategies))
    high_new_trained_i = next((i for i, s in enumerate(new_highs) if s.type == StrategyType.sb3_model), len(new_highs))
    high_extra_start = len(old_game.high_strategies) - high_trained_i - 1 + high_new_trained_i
    while high_extra_start >= 0 and new_highs[high_extra_start].name != old_game.high_strategies[-1].name:
        high_extra_start -= 1

    for i in range(high_extra_start + 1, len(new_highs)):
        old_game.high_strategies.append(new_highs[i])
        n = len(old_game.low_strategies)
        old_game.add_high_cost_col(np.zeros(n), np.zeros(n))
        for j in range(n):
            old_game.update_matrix_entry(j, len(old_game.high_strategies) - 1)

    return old_game

