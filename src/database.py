"""
This module defines the database tables used to record various characteristics of both the pricing game 
(as the reinforcement learning environment) and the meta-game (constructed via the PSRO framework). 

It tracks all trained agents, their gameplay records in the pricing environment, computed equilibria of the meta-game, 
and the average strategy probabilities within those equilibria. The structure is designed for extensibility — 
future projects can inherit from the base database class and implement custom tables as needed.

Author: Sahar Jahani
"""

import json
import sqlite3 as sql

import numpy as np
import pandas as pd
from collections import namedtuple
from stable_baselines3 import SAC, PPO
import time
from typing import List
import src.globals as gl
import src.equilibrium_solver.bimatrix as bimatrix
from src.strategy import Strategy, MixedStrategy, StrategyType
import src.utils as ut

import logging
logger = logging.getLogger(__name__) 


class BaseDataBase():
    """ Basic database class with defined methods for insertion and selection queries. 
        This class should be inherited and extended for each project.
    """
    def __init__(self, name="data.db") -> None:
        self.db_name = name
        self.reset()

    def reset(self) -> None:
        # This should be overridden in child classes to (re)create tables if needed
        pass

    def execute_insert_query(self, query) -> int:
        """
        Executes an INSERT/UPDATE/DELETE query and commits the changes.
        Returns the ID of the last inserted row.
        """
        connection = sql.connect(self.db_name, timeout=100)
        cursor = connection.cursor()
        cursor.execute(query)
        connection.commit()
        last_id = cursor.lastrowid
        connection.close()
        return last_id

    def batch_insert_to_table(self, table, values_list):
        """
        Performs a batch insert into a given table using a list of value strings.
        Each item in values_list must be a string formatted as "(val1, val2, ...)".
        """
        conn = sql.connect(self.db_name, timeout=300)
        res = None
        try:
            conn.execute('BEGIN TRANSACTION')  # Start a transaction
            cursor = conn.cursor()
            for values in values_list:
                cursor.execute(f'INSERT INTO {table} VALUES {values}')
            conn.commit()
            res = cursor.lastrowid
        except Exception:
            conn.rollback()  # Roll back the transaction in case of error
            logger.exception("Error in batch insert")  # Log the error
        finally:
            conn.close()
        return res

    def execute_select_query(self, query, fetch_one=False):
        """
        Executes a SELECT query and fetches the result.
        Set fetch_one=True to fetch a single row, otherwise fetch all rows.
        """
        connection = sql.connect(self.db_name, timeout=100)
        cursor = connection.cursor()
        cursor.execute(query)
        result = cursor.fetchone() if fetch_one else cursor.fetchall()
        return result

    def dataframe_select(self, query) -> pd.DataFrame:
        """
        Executes a SELECT query and returns the result as a pandas DataFrame.
        """
        connection = sql.connect(self.db_name, timeout=100)
        df = pd.read_sql_query(query, connection)
        connection.close()
        return df

    def delete_extra_rows(self, table_name, group_col_name, limit, id_col_name='id'):
        """
        Groups rows by group_col_name and deletes rows beyond the 'limit' for each group.
        Keeps only the first 'limit' rows per group based on current ordering.
        """
        df = self.dataframe_select(f"SELECT * FROM {table_name};")
        if not df.empty:
            connection = sql.connect(self.db_name, timeout=100)
            cursor = connection.cursor()
            gr_df = df.groupby(group_col_name)
            for name, data in gr_df:
                if len(data) > limit:
                    del_ids = []
                    for ind in range(limit, len(data)):
                        del_ids.append(data.at[data.index[ind], id_col_name])
                    cursor.execute(f"DELETE FROM {table_name} WHERE {id_col_name} IN {tuple(del_ids)}")
            connection.commit()
            cursor.execute("VACUUM")  # Optimize database size
            connection.close()


class IterRow:
    def __init__(self, agent_id, adv, agent_return, adv_return, agent_rewards, adv_rewards, actions, agent_prices, adv_prices, agent_demands, adv_demands):
        """ A structure to hold (for DB insertion) added trained agents' record of playing the pricing game against each adversary """
        self.agent_id=agent_id
        self.adv = adv
        self.agent_return = agent_return
        self.adv_return = adv_return
        self.agent_rewards = agent_rewards
        self.adv_rewards = adv_rewards
        self.actions = actions
        self.agent_prices = agent_prices
        self.adv_prices = adv_prices
        self.agent_demands = agent_demands
        self.adv_demands = adv_demands
class AgentRow:
    def __init__(self, name, base_agent,  n_ep, cost, mixed_adv_txt, alg, seed, num_process, running_time, return_std,  expected_and_threshold_payoffs, added, equi_id):
        """ A structure to hold (for DB insertion) all trained agent's performance record """
        self.name=name
        self.base_agent = base_agent
        self.n_ep = n_ep
        self.cost = cost
        self.mixed_adv_txt = mixed_adv_txt
        self.alg = alg
        self.seed = seed
        self.num_process = num_process
        self.running_time = running_time
        self.return_std = return_std
        self.expected_payoff = expected_and_threshold_payoffs[0]
        self.threshold_payoff = expected_and_threshold_payoffs[1]
        self.added=added
        self.equi_id=equi_id



class DataBase(BaseDataBase):
    """
    DataBase class to record trained agents, their performance in the pricing game and the equilibria of the meta-game.
    """


    AGENTS_TABLE = "trained_agents"
    ITERS_TABLE = "agents_iters"
    EQUI_TABLE = "game_equilibria"
    PROBS_TABLE = "strategy_average_probs"

    AGENTS_COLS = (
        "id integer PRIMARY  key AUTOINCREMENT,"
        "time text,"
        "name text NOT NULL,"
        "base_agent text DEFAULT NULL,"
        "n_ep integer NOT NULL,"
        "cost integer NOT NULL,"
        "mixed_adv text NOT NULL,"
        "alg text NOT NULL,"
        "seed integer,"
        "num_procs integer DEFAULT 1,"
        "running_time  integer,"
        "return_std real,"
        "expected_payoff real,"
        "payoff_treshhold real,"
        "added integer,"
        "equi_id integer"
    )

    ITERS_COLS = (
        "id integer PRIMARY key AUTOINCREMENT,"
        "agent_id integer NOT NULL,"
        "adv text  NOT NULL,"
        "agent_return text,"
        "adv_return text,"
        "agent_rewards text,"
        "adv_rewards text,"
        "actions text,"
        "agent_prices text,"
        "adv_prices text,"
        "agent_demands text,"
        "adv_demands text"
    )
   
    EQUI_COLS = (
        "id integer PRIMARY key AUTOINCREMENT,"
        "time text,"
        "game_size text NOT NULL,"
        "freq real,"
        "low_strategy text NOT NULL,"
        "high_strategy text NOT NULL,"
        "low_payoff real NOT NULL,"
        "high_payoff real NOT NULL,"
        "used INTEGER DEFAULT 0,"
        "num_new_low integer DEFAULT 0,"
        "num_new_high integer DEFAULT 0"
    )
  
    PROBS_COLS = (
        "id integer PRIMARY key AUTOINCREMENT,"
        "time text,"
        "game_size text,"
        "equi_count integer,"
        "last_equi_id integer,"
        "cost integer,"
        "strategy_probs text"
    )


    def reset(self):
        """
        Creates all tables in the database if they don't already exist.
        """
        connection = sql.connect(self.db_name, timeout=100)
        cursor = connection.cursor()
        cursor.execute(f'CREATE TABLE IF NOT EXISTS {self.AGENTS_TABLE}({self.AGENTS_COLS});')
        cursor.execute(f'CREATE TABLE IF NOT EXISTS {self.ITERS_TABLE}({self.ITERS_COLS});')
        cursor.execute(f'CREATE TABLE IF NOT EXISTS {self.EQUI_TABLE}({self.EQUI_COLS}, UNIQUE(game_size,low_strategy,high_strategy));')
        cursor.execute(f'CREATE TABLE IF NOT EXISTS {self.PROBS_TABLE}({self.PROBS_COLS}, UNIQUE(game_size,last_equi_id,cost));')
        connection.close()

    def insert_new_agent(self, row: AgentRow):
        """
        Inserts a new agent record and returns its ID.
        """
        query = f'INSERT INTO {self.AGENTS_TABLE} VALUES (NULL,\'{time.ctime(time.time())}\',\'{row.name}\',' + \
            ('NULL' if (row.base_agent is None) else f'\'{row.base_agent}\'') + \
            f',{row.n_ep},{row.cost},\'{row.mixed_adv_txt}\',\"{row.alg}\",{row.seed},{row.num_process},{row.running_time},{row.return_std},{row.expected_payoff},{row.threshold_payoff},{int(row.added)},{row.equi_id})'
        return self.execute_insert_query(query=query)

    def insert_new_iteration(self, agent_id, adv_txt, agent_return, adv_return, agent_rewards_txt, adv_rewards_txt, actions_txt, agent_prices_txt, adv_prices_txt, agent_demands_txt, adv_demands_txt):
        """
        Inserts a new iteration record and returns its ID.
        """
        query = f'INSERT INTO {self.ITERS_TABLE} VALUES (NULL,{agent_id},\'{adv_txt}\',{agent_return},{adv_return},\'{agent_rewards_txt}\',\'{adv_rewards_txt}\',\
            \'{actions_txt}\',\'{agent_prices_txt}\',\'{adv_prices_txt}\',\'{agent_demands_txt}\',\'{adv_demands_txt}\')'
        return self.execute_insert_query(query=query)
    def insert_many_new_iters(self, iters: list[IterRow]):
        """
        inserts many iteration rows into the database
        iters: list of IterRow namedtuples
        """
        values_list = []
        for r in iters:
            values = f"(NULL,{r.agent_id},'{r.adv}',{r.agent_return},{r.adv_return},'{r.agent_rewards}','{r.adv_rewards}',\
                '{r.actions}','{r.agent_prices}','{r.adv_prices}','{r.agent_demands}','{r.adv_demands}')"
            values_list.append(values)
        return self.batch_insert_to_table(self.ITERS_TABLE, values_list)

    def insert_new_equi(self, game_size, freq, low_strategy_txt, high_strategy_txt, low_payoff, high_payoff, used, num_new_low, num_new_high):
        """
        Adds a new equilibrium to the database and returns the inserted row ID.
        If the equilibrium already exists, returns its existing ID.
        """
        # Check if the equilibrium already exists in the database
        row = self.execute_select_query(
            f"SELECT id FROM {self.EQUI_TABLE} "
            f"WHERE game_size = '{str(game_size)}' "
            f"AND low_strategy = '{low_strategy_txt}' "
            f"AND high_strategy = '{high_strategy_txt}';",
            fetch_one=True
        )
        
        if row:
            return row[0]  # Return existing ID
        else:
            # Insert new equilibrium
            query = (
                f"INSERT INTO {self.EQUI_TABLE} VALUES ("
                f"NULL, '{time.ctime(time.time())}', '{str(game_size)}', {freq}, "
                f"'{low_strategy_txt}', '{high_strategy_txt}', "
                f"{low_payoff}, {high_payoff}, {int(used)}, {num_new_low}, {num_new_high})"
            )
            return self.execute_insert_query(query=query)

    def update_equi(self, id, used, num_new_low, num_new_high):
        """
        Updates the 'used' status and new strategy counts for a given equilibrium ID.
        """
        query = (
            f"UPDATE {self.EQUI_TABLE} "
            f"SET used = {used}, num_new_low = {num_new_low}, num_new_high = {num_new_high} "
            f"WHERE id = {id};"
        )
        return self.execute_insert_query(query=query)

    def insert_new_average_probs(self, game_size, equi_count, last_equi_id, cost, strategy_probs_str):
        """
        Adds a new row of average strategy probabilities to the database.
        Each equilibrium should result in two rows: one for each cost.
        """
        # Check if this average probability entry already exists
        row = self.execute_select_query(
            f"SELECT id FROM {self.PROBS_TABLE} "
            f"WHERE game_size = '{str(game_size)}' "
            f"AND last_equi_id = {last_equi_id} "
            f"AND cost = {cost};",
            fetch_one=True
        )
        
        if row:
            return row[0]  # Return existing ID
        else:
            # Insert new average strategy probability
            query = (
                f"INSERT INTO {self.PROBS_TABLE} VALUES ("
                f"NULL, '{time.ctime(time.time())}', '{str(game_size)}', "
                f"{equi_count}, {last_equi_id}, {cost}, \"{strategy_probs_str}\")"
            )
            return self.execute_insert_query(query=query)

    def updates_equi_average_probs(self, equi_id: int, equi: bimatrix.Equi):
        """
        Updates average strategy probabilities using a running average based on new equilibrium data.
        Adds one entry per cost type (LOW_COST and HIGH_COST).
        """
        game_size = (len(equi.row_probs), len(equi.col_probs))
        
        for cost in [gl.LOW_COST, gl.HIGH_COST]:
            # Select row or column probabilities based on the cost
            new_probs = equi.row_probs if cost == gl.LOW_COST else equi.col_probs
            
            # Get the latest average for the given cost
            query = f"SELECT * FROM {self.PROBS_TABLE} WHERE cost = {cost} ORDER BY id DESC LIMIT 1;"
            df = self.dataframe_select(query)
            
            if df.empty:
                # No previous averages exist
                equi_count = 1
                avg_probs = new_probs
            else:
                row = df.iloc[0]
                equi_count = row['equi_count'] + 1
                
                # Load previous probabilities and match length if needed
                prev_probs = json.loads(row['strategy_probs'])
                if (extra := len(new_probs) - len(prev_probs)) > 0:
                    prev_probs.extend([0] * extra)
                
                # Compute new running average
                avg_probs = (
                    (np.array(new_probs) + (np.array(prev_probs) * (equi_count - 1))) / equi_count
                ).tolist()
            
            self.insert_new_average_probs(game_size, equi_count, equi_id, cost, str(avg_probs))
    
    def get_list_of_added_strategies(self,memory, action_step):
        
        """ Returns lists of low_cost and high_cost agents added to the game  """
        
        low_q = f"SELECT name, alg FROM {self.AGENTS_TABLE} WHERE (added=1 and cost={gl.LOW_COST})"
        high_q = f"SELECT name, alg FROM {self.AGENTS_TABLE} WHERE (added=1 and cost={gl.HIGH_COST})"
        low_lst = []
        high_lst = []

        connection = sql.connect(self.db_name, timeout=100)
        cursor = connection.cursor()

        cursor.execute(low_q)
        low_all = cursor.fetchall()
        for tup in low_all:
            model=alg_classes[tup[1]]
            low_lst.append(Strategy(strategy_type=StrategyType.sb3_model, model_or_func=model,
                           name=tup[0], memory=memory, action_step=action_step))

        cursor.execute(high_q)
        high_all = cursor.fetchall()
        for tup in high_all:
            model=alg_classes[tup[1]] 
            high_lst.append(Strategy(strategy_type=StrategyType.sb3_model, model_or_func=model,
                            name=tup[0], memory=memory, action_step=action_step))
        connection.close()
        return low_lst, high_lst

alg_classes = {
    'SAC': SAC,
    'PPO': PPO
 }