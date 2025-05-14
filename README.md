# Pricing Game Strategy Evolution via PSRO and Multi-Agent Reinforcement Learning 

This project investigates the existence of a stable equilibrium with higher returns than the Subgame Perfect Equilibrium (SPE) for players in an asymmetric duopoly pricing game with demand inertia.

This class of games was originally studied by Reinhard Selten in 1965, who defined and computed the Subgame Perfect Equilibria. Later, in 1992, Claudia Keser conducted an experimental study using this game, where game theorists submitted their strategies via flowcharts to compete in the setting.

In this project, we utilise reinforcement learning (RL) to train low-cost and high-cost agents to play the 25-stage pricing game, as studied in Keser’s work. To approximate the infinite game over all possible pricing strategies, we use the **Policy-Space Response Oracles (PSRO)** method to construct a meta-game that works as an evolving approximation.

The meta-game begins with a set of initial strategies, which may be random or predefined deterministic strategies. We compute the equilibrium of the current meta-game and then train new RL agents as approximate best responses to the equilibrium strategies. If these agents achieve returns higher than the equilibrium payoff, they are added to the meta-game. This iterative process expands the strategy space and continually extends the meta-game.

The resulting meta-game approximates the infinite strategy space of the original pricing game, and its computed equilibria approximate the equilibria of the full game.

## 🧠 Reinforcement Learning Framework

- We use the [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) framework.
- The pricing game is defined as a custom environment compatible with the **Gymnasium API**, implemented in the `ConPricingGame` class.
- We primarily use the following algorithms:
  - **Proximal Policy Optimization (PPO)**
  - **Soft Actor-Critic (SAC)**

## ⚙️ Multiprocessing & Data Management

- We use **multiprocessing** to speed up training and evaluation.
- The `BimatrixGame` class represents the meta-game.
- All data (trained agents, meta-games, equilibria, logs) is saved using custom-designed classes and databases for efficient data management.

## 🎯 Equilibrium Enumeration in PSRO

Nash equilibria of the evolving bimatrix meta-game—central to the **Policy Space Response Oracles (PSRO)** framework—are computed using the **Lemke algorithm**, implemented by Prof. **Bernhard von Stengel**.

📁 The source code for the solver is located in the `src/equilibrium_solver/` directory.

---
## 🛠️ Installation

Before running the training script, make sure you have the required packages installed.

### Install Dependencies

Install [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3):
```bash
pip install stable-baselines3
```

Install [TensorBoard]() for monitoring training progress:
```bash
pip install tensorboard
```
## 🚀 How to Run

To start the training process, set the following parameters in `main_psro_rl.py`, then run the script.

### Parameters to Set:

- **`game_name`**  
  The prefix used for all files and folders generated during the experiment.  
  ⚠️ **Use the same name to resume an existing experiment.**

- **`num_procs`**  
  The number of parallel processes for training, based on your system’s available CPU cores.

### Run the Script

```bash
python main_psro_rl.py
```

### ⚙️ Configuration

All constants and hyperparameters for both the pricing game and training procedure are defined in `globals.py`.

This includes:

- Agent production costs  
- Total demand potential  
- Number of time steps in the pricing game  
- Agents' memory of prices in the observations  
- Number of episodes for training each agent  
- Number of equilibria to follow in each round of the meta-game  
- Hyperparameters for the SAC and PPO algorithms  

The hyperparameters were determined through prior experimentation to ensure efficient and stable learning in this setting.

---
## 📁 Project Structure and Output Files

When you start running `main_psro_rl.py`, the following directories and files will be created in the base folder:

### `games/`
Stores snapshots of meta-games. Each file includes:
- The first line: number of rows and columns.
- The first matrix: payoffs for the **Low-cost** player.
- The second matrix: payoffs for the **High-cost** player.
- Strategy names for both players at the end of the file.

These files serve as a complete record of the evolving meta-game structure across training rounds.

### `models/`
Contains all trained agent models.  
Model files are named using the format: `<game_name>-<timestamp>.zip`

These models can be used for evaluation or to resume training.  
- Models corresponding to strategies in the meta-game are **essential** for continuing training, as new agents may need to train against them if they appear in future equilibria.  
- Models from earlier iterations that were **not added** to the game might still be useful for initializing future agents but can be removed later if needed to save space.

### `logs/`
Contains TensorBoard-compatible training logs, named to match their corresponding models.

To monitor training progress with TensorBoard, run the following command from the base directory:

```bash
tensorboard --logdir logs/
```

### 📄 Base Folder Files

#### `game_<game_name>.txt`
Text representation of the current meta-game used by the equilibrium solver.  
⚠️ **Do not delete** — This file is required for equilibrium computation.

#### `game_<game_name>.pickle`
Serialized `BimatrixGame` object that stores the full meta-game and associated strategies.  
Used to resume training without recomputation.  
⚠️ **Deleting this file will require recomputing all payoffs**, which can take hours for large games.

#### `game_<game_name>.db`
SQLite database storing:
- Trained pricing strategies (pure strategies in the meta-game)
- Detailed results of pricing games between agents
- Meta-game equilibria
- Average probabilities of strategies in these equilibria

⚠️ **Required to resume training**.  
Deleting this file will cause the experiment to start from scratch.

#### `progress_<game_name>.txt`
Logs the training progress in plain text:
- Start time of training
- Meta-game equilibria at each round
- Strategies added during each round based on equilibrium

You can monitor this file during training to check progress.

#### `error.log`
If any error occurs during training, it will be logged here.  
Useful for debugging or identifying failed runs.

---
## 🔄 Resuming Training

To resume a previously stopped experiment:

- Use the **same `game_name`** when rerunning `main_psro_rl.py`.
- Ensure all the following files are present in the base folder:

  - `game_<game_name>.txt`
  - `game_<game_name>.pickle`
  - `game_<game_name>.db`
  - Trained model files in the `models/` directory

These files are required to reload the meta-game, agent strategies, and training logs.  
⚠️ If any of these are missing, the experiment must be restarted from scratch.

---
## 👤 Author

**Sahar Jahani**  
PhD in Mathematics – London School of Economics
