"""
This project investigates the existence of a stable equilibrium with higher returns than the Subgame Perfect Equilibrium (SPE) 
for players in an asymmetric duopoly pricing game with demand inertia. This class of games was originally studied by 
Reinhard Selten in 1965, who defined and computed the Subgame Perfect Equilibria. Later, in 1992, Claudia Keser conducted 
an experimental study using this game, where game theorists submitted their strategies via flowcharts to compete in the setting.

In this project, we leverage reinforcement learning (RL) to train low-cost and high-cost agents to play the 25-stage 
pricing game, as studied in Keser’s work. To approximate the infinite game over all possible pricing strategies, we use 
the Policy-Space Response Oracles (PSRO) method to construct a meta-game that serves as an evolving approximation.

The meta-game begins with a set of initial strategies, which may be a random strategy or predefined deterministic ones. 
We compute the equilibrium of the current meta-game and then train new RL agents as approximate best responses to the 
equilibrium strategies. If these agents achieve returns higher than the equilibrium payoff, they are added to the meta-game. 
This iterative process expands the strategy space and continually updates the meta-game.

The resulting meta-game approximates the infinite strategy space of the original pricing game, and its computed equilibria approximate the equilibria of the full game.

For reinforcement learning, we use the Stable-Baselines3 framework. The pricing game is implemented as a custom environment compatible with the Gymnasium API, making it usable with a wide range of RL algorithms. In our experiments, we mainly use Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC) to train the agents.

We use multiprocessing to speed up training and evaluation. The `BimatrixGame` class represents the meta-game, and all data related to trained agents, meta-game dynamics, and computed equilibria are stored using the custom-designed `DataBase` class for efficient data management. 

Running this Python file initiates the training process, during which all games, trained models, and logs are automatically saved to the database and corresponding directories. A paused game can be resumed from the last round of the meta-game, as the most recent state is preserved using serializable classes that can be reloaded without the need for recomputation.

Please refer to the README file for detailed usage instructions.

Author: Sahar Jahani
"""



import numpy as np
import time
import multiprocessing as mp
from typing import  Dict

from src.envs import ConPricingGame
import src.globals as gl
import src.core as cl
from src.database import DataBase
import src.utils as ut
import src.training_config as tc
from src.strategy import StrategyType, Strategy, MixedStrategy

import logging
# Configure logging
logging.basicConfig(filename='error.log',
                    level=logging.ERROR,
                    format='%(asctime)s %(name)s [%(levelname)s]: %(message)s')


if __name__ == "__main__":
    try:
        gl.initialize()

        env_class = ConPricingGame

        num_rounds = 10
        num_procs = 4  # Works best with num_procs = 1 or >= 4
        start_mode = tc.StartMode.random

        game_name = 'test'

        db_name = f"{game_name}.db"
        db = DataBase(db_name)
    
        tc.create_directories()

        equilibrium_id_map = []

        initial_game = tc.initial_matrix(env_class=env_class, start_mode=start_mode, database=db, game_name=game_name)
        bimatrix_game = tc.load_latest_game(game_data_name=f"game_{game_name}", new_game=initial_game)

        bimatrix_game.prt("\n" + time.ctime(time.time()) + "\n" + ("-" * 50) + "\n")

        equilibrium_id_map = cl.get_coop_equilibria(bimatrix_game=bimatrix_game, num_trace=100, db=db)
        game_size = bimatrix_game.size()

        episode_scaling_factor = 1

        for round_idx in range(num_rounds):
            bimatrix_game.prt(f"\n\tRound {round_idx + 1} of {num_rounds}")

            added_low, added_high = 0, 0

            for equilibrium in equilibrium_id_map:
                db.updates_equi_average_probs(equilibrium_id_map[equilibrium], equilibrium)

                new_low_strategies = 0
                new_high_strategies = 0

                bimatrix_game.prt(
                    f"Equilibrium: {equilibrium.row_support}, {equilibrium.col_support}\n"
                    f"Payoffs = {equilibrium.row_payoff:.2f}, {equilibrium.col_payoff:.2f}"
                )

                high_mixed_strategy = MixedStrategy(
                    strategies_lst=bimatrix_game.high_strategies,
                    probablities_lst=equilibrium.col_probs + ([0] * added_high if added_high > 0 else []),
                )
                low_mixed_strategy = MixedStrategy(
                    strategies_lst=bimatrix_game.low_strategies,
                    probablities_lst=equilibrium.row_probs + ([0] * added_low if added_low > 0 else []),
                )

                proc_inputs = []
                input_map: Dict[int, cl.TrainInputRow] = {}
                seed = int(time.time())

                for proc_id in tc.ProcessInd:
                    train_input = tc.get_proc_input(
                        seed=seed,
                        proc_ind=proc_id,
                        low_mixed_strt=low_mixed_strategy,
                        high_mixed_strt=high_mixed_strategy,
                        target_payoffs=[equilibrium.row_payoff, equilibrium.col_payoff],
                        job_name=game_name,
                        env_class=env_class,
                        num_ep_coef=episode_scaling_factor,
                        equi_id=equilibrium_id_map[equilibrium],
                        db=db
                    )
                    proc_inputs.append(train_input)
                    input_map[train_input.id] = train_input

                if num_procs > 1:
                    extra_procs = num_procs - len(tc.ProcessInd)
                    if extra_procs > 0:
                        for extra_idx in range(extra_procs):
                            random_proc_id = np.random.choice(list(tc.ProcessInd))
                            new_seed = seed + 1 + (extra_idx + 1) * len(tc.ProcessInd)

                            extra_input = tc.get_proc_input(
                                seed=new_seed,
                                proc_ind=random_proc_id,
                                low_mixed_strat=low_mixed_strategy,
                                high_mixed_strat=high_mixed_strategy,
                                target_payoffs=[equilibrium.row_payoff, equilibrium.col_payoff],
                                job_name=game_name,
                                env_class=env_class,
                                num_ep_coef=episode_scaling_factor,
                                equilibrium_id=equilibrium_id_map[equilibrium],
                                db=db
                            )
                            proc_inputs.append(extra_input)
                            input_map[extra_input.id] = extra_input

                    with mp.Pool(processes=num_procs) as pool:
                        outputs = pool.imap_unordered(cl.new_train, proc_inputs)
                        pool.close()
                        pool.join()
                else:
                    outputs = [cl.new_train(inp) for inp in proc_inputs]

                for output in outputs:
                    strategy_id, is_acceptable, strategy_name, agent_payoffs, adv_payoffs, expected_payoff = output
                    train_input = input_map[strategy_id]
                    pricing_env = train_input.env

                    model_strategy =Strategy(
                        strategy_type=StrategyType.sb3_model,
                        model_or_func=train_input.alg,
                        name=strategy_name,
                        action_step=pricing_env.action_step,
                        memory=pricing_env.memory
                    )

                    if is_acceptable:
                        updated_adv, agent_payoffs, adv_payoffs = cl.match_updated_size(
                            bimatrix_game, train_input.adv_mixed_strategy,
                            pricing_env.costs[0], agent_payoffs, adv_payoffs
                        )

                        for idx, adv_strategy in enumerate(updated_adv.strategies):
                            if updated_adv.strategy_probs[idx] == 0:
                                sampled_payoffs = [
                                    model_strategy.play_against(pricing_env, adversary=adv_strategy)
                                    for _ in range(gl.NUM_STOCHASTIC_ITER)
                                ]
                                mean_payoffs = np.mean(sampled_payoffs, axis=0)
                                agent_payoffs[idx] = mean_payoffs[0]
                                adv_payoffs[idx] = mean_payoffs[1]

                        if pricing_env.costs[0] == gl.LOW_COST:
                            new_low_strategies += 1
                            added_low += 1
                            bimatrix_game.low_strategies.append(model_strategy)
                            bimatrix_game.add_low_cost_row(agent_payoffs, adv_payoffs)
                            bimatrix_game.prt(f"Low-cost strategy {model_strategy.name} added with payoff {expected_payoff:.2f}")

                        elif pricing_env.costs[0] == gl.HIGH_COST:
                            new_high_strategies += 1
                            added_high += 1
                            bimatrix_game.high_strategies.append(model_strategy)
                            bimatrix_game.add_high_cost_col(adv_payoffs, agent_payoffs)
                            bimatrix_game.prt(f"High-cost strategy {model_strategy.name} added with payoff {expected_payoff:.2f}")

                db.update_equi(
                    id=equilibrium_id_map[equilibrium],
                    used=(new_low_strategies > 0 or new_high_strategies > 0),
                    num_new_low=new_low_strategies,
                    num_new_high=new_high_strategies
                )

                if new_high_strategies > 0:
                    high_mixed_strategy.strategy_probs += [0] * new_high_strategies

            if added_low == 0 and added_high == 0:
                episode_scaling_factor *= gl.EPISODE_INCREASE_COEF
            else:
                equilibrium_id_map = cl.get_coop_equilibria(bimatrix_game=bimatrix_game, num_trace=100, db=db)
                game_size = bimatrix_game.size()
                episode_scaling_factor = 1

        all_equilibria = bimatrix_game.compute_equilibria()
        equilibrium_id_map = all_equilibria[:min(len(all_equilibria), gl.NUM_TRACE_EQUILIBRIA)]

    except Exception as e:
        logging.error("An error occurred: %s", str(e),exc_info=True)
        print("An error occurred:", e)
