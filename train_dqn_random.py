import numpy as np
import random
import time
import math
import sys
import os
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

# Tensorflow GPU Configuration
if tf.config.list_physical_devices('GPU'):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Using mixed_float16 precision for GPU.")
else:
    print("No GPU detected or configured by TensorFlow. Using float32.")

try:
    from board import ConnectFourBoard
    from dqn_agent import DQNAgent
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure board.py and dqn_agent.py are in the same directory.")
    sys.exit(1)

# Constants
PLAYER1 = 1
PLAYER2 = 2
EMPTY = 0

# Agent Perspective Constants
AGENT_ID = 1
OPPONENT_ID = -1

# Training Parameters
TOTAL_EPISODES = 100000
MAX_STEPS_PER_EPISODE = 100
SAVE_FREQ = 100                # How often to save weights (in episodes)
LOG_FREQ = 50                  # How often to print episode summary
PLOT_FREQ = 20                 # How often to update plot AND check stopping condition
WEIGHTS_FILE_PATTERN = "models/connect4-dqn-vs-random-ep{}.weights.h5"
LATEST_WEIGHTS_FILE = "models/connect4-dqn-vs-random-latest.weights.h5"
WIN_RATE_PLOT_FILE = "img/connect4-dqn-vs-random-winrate.png"

# How many agent moves trigger a learning step
REPLAY_EVERY_N_AGENT_MOVES = 1

LEARNING_RATE = 0.0005
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.9997
EPSILON_MIN = 0.01
# Smaller memory may be sufficient, prevents holding old data
MEMORY_SIZE = 50000
BATCH_SIZE = 512
# How many learning steps (replay calls) before updating target network
TARGET_UPDATE_FREQ = 800
USE_DOUBLE_DQN = True

# Win rate threshold over the PLOT_FREQ interval
STOPPING_WIN_RATE_THRESHOLD = 0.98
# How many consecutive intervals must meet the threshold
STOPPING_CONSECUTIVE_INTERVALS = 3


# Translates board from PLAYER1/PLAYER2 to AGENT_ID/OPPONENT_ID.
def translate_board_to_agent(board_state, agent_player_id):
    agent_board = np.zeros_like(board_state, dtype=np.float32)
    # Opponent is always random, but map the *other* player to OPPONENT_ID
    opponent_player_id = PLAYER2 if agent_player_id == PLAYER1 else PLAYER1
    agent_board[board_state == agent_player_id] = AGENT_ID
    agent_board[board_state == opponent_player_id] = OPPONENT_ID
    # Add channel dimension for CNN input
    return np.expand_dims(agent_board, axis=-1)


def print_game_board(board_state):
    print("\n  0   1   2   3   4   5   6 ")
    print("-----------------------------")
    for r in range(ConnectFourBoard.ROW_COUNT):
        print("|", end="")
        for c in range(ConnectFourBoard.COLUMN_COUNT):
            if board_state[r, c] == PLAYER1:
                print(" X ", end="|")
            elif board_state[r, c] == PLAYER2:
                print(" O ", end="|")
            else:
                print("   ", end="|")
        print("\n-----------------------------")
    print()


if __name__ == "__main__":
    print("--- Starting DQN vs Random Agent Training ---")
    print("Hyperparameters:")
    print(f"  LEARNING_RATE: {LEARNING_RATE}")
    print(f"  GAMMA: {GAMMA}")
    print(f"  EPSILON_DECAY: {EPSILON_DECAY}")
    print(f"  EPSILON_MIN: {EPSILON_MIN}")
    print(f"  MEMORY_SIZE: {MEMORY_SIZE}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  TARGET_UPDATE_FREQ (replay calls): {TARGET_UPDATE_FREQ}")

    # Initialize Environment and Agent
    env = ConnectFourBoard()
    state_shape = (ConnectFourBoard.ROW_COUNT,
                   ConnectFourBoard.COLUMN_COUNT, 1)  # Add channel dimension
    action_size = ConnectFourBoard.COLUMN_COUNT

    agent = DQNAgent(state_shape=state_shape,
                     action_size=action_size,
                     learning_rate=LEARNING_RATE,
                     gamma=GAMMA,
                     epsilon=EPSILON_START,
                     epsilon_decay=EPSILON_DECAY,
                     epsilon_min=EPSILON_MIN,
                     memory_size=MEMORY_SIZE,
                     batch_size=BATCH_SIZE,
                     target_update_freq=TARGET_UPDATE_FREQ,
                     use_double_dqn=USE_DOUBLE_DQN)

    # Load Previous Weights (if they exists...)
    start_episode = 0
    if os.path.exists(LATEST_WEIGHTS_FILE):
        print(f"Loading weights from {LATEST_WEIGHTS_FILE}")
        if agent.load(LATEST_WEIGHTS_FILE):
            print("Loaded previous weights. Resuming training.")
            try:
                existing_files = [f for f in os.listdir('.') if f.startswith(
                    'connect4-dqn-vs-random-ep') and f.endswith('.weights.h5')]
                if existing_files:
                    ep_numbers = [int(f.split('ep')[1].split('.')[0])
                                  for f in existing_files]
                    if hasattr(agent, 'episode') and agent.episode > 0:
                        start_episode = agent.episode
                        print(f"Resuming from agent's internal episode count: {
                              start_episode}")
                    elif ep_numbers:
                        start_episode = max(ep_numbers)
                        print(f"Resuming training from episode {
                              start_episode + 1}")
            except Exception as load_ep_err:
                print(f"Could not reliably determine start episode from filenames: {
                      load_ep_err}. Starting count from 0.")
                start_episode = 0
        else:
            print("Failed to load weights. Starting from scratch.")
            start_episode = 0
    else:
        print("No previous weights found. Starting from scratch.")
        start_episode = 0

    total_steps = 0
    episode_rewards = deque(maxlen=100)
    episode_losses = deque(maxlen=100)

    # Win rate tracking for plotting and stopping
    plot_episodes = []
    plot_win_rates = []
    wins_in_interval = 0
    losses_in_interval = 0
    draws_in_interval = 0
    games_in_interval = 0

    # Automatic stopping tracking
    consecutive_high_win_rate_intervals = 0
    training_complete = False

    # Counter for replay frequency control
    agent_moves_since_last_replay = 0

    print(f"Training for up to {
          TOTAL_EPISODES} episodes (starting from episode {start_episode + 1})...")
    print(f"Logging every {LOG_FREQ} episodes.")
    print(f"Plotting win rate and checking stopping condition every {
          PLOT_FREQ} episodes.")
    print(f"Stopping if win rate >= {STOPPING_WIN_RATE_THRESHOLD:.2f} for {
          STOPPING_CONSECUTIVE_INTERVALS} consecutive intervals.")
    print(f"Saving weights every {SAVE_FREQ} episodes.")

    for e in range(start_episode, TOTAL_EPISODES):
        episode_start_time = time.time()
        env.reset()

        player_id_agent = random.choice([PLAYER1, PLAYER2])
        player_id_opponent = PLAYER2 if player_id_agent == PLAYER1 else PLAYER1

        current_player_env = PLAYER1

        current_board_state = env.get_board()
        done = False
        step_in_episode = 0
        episode_reward_sum = 0
        episode_loss_sum = 0
        agent_learn_steps_count = 0

        # Temporary storage for the learning agent's transitions this episode
        episode_memory = []

        last_agent_state = None
        last_agent_action = None
        winner = None

        while not done and step_in_episode < MAX_STEPS_PER_EPISODE:
            valid_locations = env.get_valid_locations()
            if not valid_locations:
                done = True
                winner = None
                game_over = True
                # If agent made the last move leading to this state, record the final transition
                if last_agent_state is not None and last_agent_action is not None:
                    next_board_state = env.get_board()
                    next_state_agent_view = translate_board_to_agent(
                        next_board_state, player_id_agent)
                    episode_memory.append((last_agent_state, last_agent_action,
                                          0.0, next_state_agent_view, True))
                    episode_reward_sum += 0.0
                break

            action = -1

            is_learning_agent_turn = (current_player_env == player_id_agent)

            # Action Selection
            if is_learning_agent_turn:
                state_agent_view = translate_board_to_agent(
                    current_board_state, player_id_agent)
                last_agent_state = state_agent_view

                action = agent.act(
                    state_agent_view, valid_locations, force_exploit=False)

                # Basic safety check
                if action is None or action not in valid_locations:
                    print(
                        f"Warning: Agent returned invalid action ({action}) in ep {e+1}. Choosing random from {valid_locations}.")
                    action = random.choice(valid_locations)

                last_agent_action = action

            else:
                # Opponent's turn: Random action
                action = random.choice(valid_locations)
                last_agent_state = None
                last_agent_action = None

            row_placed = env.drop_piece(action, current_player_env)
            # Error check
            if row_placed == -1:
                print(f"CRITICAL ERROR: Ep {e+1}, Player {current_player_env} chose invalid action {
                      action} (valid: {valid_locations}). Board:\n{current_board_state}")
                done = True
                winner = None
                game_over = True
                break

            next_board_state = env.get_board()
            step_in_episode += 1
            total_steps += 1

            # Check Game Over and Determine Winner
            game_over = env.game_over
            if game_over:
                winner = env.winner

            # Reward Calculation
            if is_learning_agent_turn:
                reward = 0.0  # Default reward for non-terminal move
                is_terminal_state_after_agent = False

                if game_over:
                    is_terminal_state_after_agent = True
                    if winner == player_id_agent:
                        reward = 1.0  # Agent won
                    elif winner is None:  # Draw resulted from agent's move
                        reward = 0.0

                next_state_agent_view = translate_board_to_agent(
                    next_board_state, player_id_agent)

                # Store the transition
                if last_agent_state is not None and last_agent_action is not None:
                    episode_memory.append((last_agent_state, last_agent_action,
                                          reward, next_state_agent_view, is_terminal_state_after_agent))
                    episode_reward_sum += reward

                    agent_moves_since_last_replay += 1
                    if agent_moves_since_last_replay >= REPLAY_EVERY_N_AGENT_MOVES:
                        loss = agent.replay()  # Sample batch and train
                        if loss is not None:
                            episode_loss_sum += loss
                            agent_learn_steps_count += 1
                        agent_moves_since_last_replay = 0

            elif game_over:
                if episode_memory:
                    s, a, r_old, ns_opponent_turn, d_old = episode_memory.pop()

                    final_reward = r_old
                    if winner == player_id_opponent:  # Agent Lost
                        final_reward = -1.0
                        episode_reward_sum += final_reward
                    elif winner is None:
                        final_reward = 0.0
                        episode_reward_sum += final_reward

                    episode_memory.append(
                        (s, a, final_reward, ns_opponent_turn, True))

            current_board_state = next_board_state
            done = game_over
            if not done:
                current_player_env = PLAYER2 if current_player_env == PLAYER1 else PLAYER1

        episode_duration = time.time() - episode_start_time
        # agent.update_epsilon()  # Decay epsilon after each episode

        # Add this episode's finalized transitions to the main replay buffer
        for experience in episode_memory:
            agent.remember(*experience)

        # Update win/loss/draw counts for the current plotting interval
        games_in_interval += 1
        if winner == player_id_agent:
            wins_in_interval += 1
        elif winner == player_id_opponent:
            losses_in_interval += 1
        elif winner is None:
            draws_in_interval += 1

        episode_rewards.append(episode_reward_sum)
        avg_reward_100 = sum(episode_rewards) / \
            len(episode_rewards) if episode_rewards else 0.0

        # Calculate avg loss per learning step during the episode
        avg_loss_episode = episode_loss_sum / \
            agent_learn_steps_count if agent_learn_steps_count > 0 else 0.0
        if agent_learn_steps_count > 0:
            # Store the average for history
            episode_losses.append(avg_loss_episode)
        avg_loss_100 = sum(episode_losses) / \
            len(episode_losses) if episode_losses else 0.0

        if (e + 1) % LOG_FREQ == 0:
            print(f"Ep {e+1}/{TOTAL_EPISODES} | Steps: {step_in_episode} | Dur: {episode_duration:.2f}s | "
                  f"Eps: {agent.epsilon:.4f} | Replay: {
                      len(agent.memory)}/{agent.memory.maxlen} | "
                  f"Ep Rew: {episode_reward_sum:.1f} | Avg Rew (100): {
                avg_reward_100:.3f} | "
                f"Avg Loss (100): {avg_loss_100:.5f}")

        if (e + 1) % PLOT_FREQ == 0 and games_in_interval > 0:
            # Calculate win rate for the completed interval
            win_rate_interval = wins_in_interval / games_in_interval
            plot_episodes.append(e + 1)
            plot_win_rates.append(win_rate_interval)

            print(
                f"--- Interval Summary (Episodes {max(start_episode, e+1-games_in_interval+1)}-{e+1}) ---")
            print(f"    Games: {games_in_interval}, Wins: {wins_in_interval}, Losses: {
                  losses_in_interval}, Draws: {draws_in_interval}")
            print(f"    Win Rate: {win_rate_interval:.3f}")

            try:
                plt.figure(figsize=(12, 6))
                plt.plot(plot_episodes, plot_win_rates,
                         marker='o', linestyle='-', markersize=4)
                # threshold line
                plt.axhline(y=STOPPING_WIN_RATE_THRESHOLD, color='r', linestyle='--',
                            label=f'Stop Threshold ({STOPPING_WIN_RATE_THRESHOLD:.2f})')
                plt.title(
                    f'DQN Agent Win Rate vs Random Agent (Intervals of {PLOT_FREQ} EPs)')
                plt.xlabel('Episodes')
                plt.ylabel('Win Rate')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.ylim(-0.05, 1.05)
                plt.legend()
                if plot_episodes:
                    last_episode = plot_episodes[-1]
                    last_rate = plot_win_rates[-1]
                    plt.text(last_episode, last_rate, f' {
                             last_rate:.3f}', va='bottom' if last_rate < 0.9 else 'top')

                plt.tight_layout()
                plt.savefig(WIN_RATE_PLOT_FILE)
                plt.close()
                print(f"    Win rate plot updated and saved to {
                      WIN_RATE_PLOT_FILE}")
            except Exception as plot_err:
                print(
                    f"    Warning: Could not generate or save plot: {plot_err}")

            if win_rate_interval >= STOPPING_WIN_RATE_THRESHOLD:
                consecutive_high_win_rate_intervals += 1
                print(f"    High win rate threshold met! ({
                      consecutive_high_win_rate_intervals}/{STOPPING_CONSECUTIVE_INTERVALS} consecutive intervals)")
            else:
                if consecutive_high_win_rate_intervals > 0:
                    print(
                        f"    Win rate fell below threshold. Resetting consecutive count.")
                consecutive_high_win_rate_intervals = 0  # Reset count

            # Trigger early stopping if condition met
            if consecutive_high_win_rate_intervals >= STOPPING_CONSECUTIVE_INTERVALS:
                print(f"\n--- AUTO-STOPPING TRAINING ---")
                print(f"Achieved >= {STOPPING_WIN_RATE_THRESHOLD:.2f} win rate for {
                      STOPPING_CONSECUTIVE_INTERVALS} consecutive intervals (ending at ep {e+1}).")
                training_complete = True
                # Save final weights and plot before breaking
                final_episode_num = e + 1
                save_path = WEIGHTS_FILE_PATTERN.format(final_episode_num)
                agent.save(save_path)
                agent.save(LATEST_WEIGHTS_FILE)  # Overwrite latest as well
                print(f"Final model weights saved to {
                      save_path} and {LATEST_WEIGHTS_FILE}")
                break

            # Reset counters for the next plotting/stopping interval
            wins_in_interval = 0
            losses_in_interval = 0
            draws_in_interval = 0
            games_in_interval = 0

        # Save progress even if not stopping early
        if not training_complete and (e + 1) % SAVE_FREQ == 0:
            save_path = WEIGHTS_FILE_PATTERN.format(e+1)
            agent.save(save_path)
            agent.save(LATEST_WEIGHTS_FILE)
            print(f"Saved intermediate weights at episode {
                  e+1} to {save_path} and {LATEST_WEIGHTS_FILE}")

    # End of Training
    if not training_complete:
        # Reached total episodes without meeting stopping criteria
        print(
            f"\n--- DQN vs Random Training Finished (Reached {TOTAL_EPISODES} episodes) ---")
        final_save_path = WEIGHTS_FILE_PATTERN.format(TOTAL_EPISODES)
        agent.save(final_save_path)
        agent.save(LATEST_WEIGHTS_FILE)
        print(f"Final model weights saved to {
              final_save_path} and {LATEST_WEIGHTS_FILE}")

        # Ensure final plot is saved if training finished normally
        if plot_episodes and games_in_interval == 0:
            try:
                plt.figure(figsize=(12, 6))
                plt.plot(plot_episodes, plot_win_rates,
                         marker='o', linestyle='-', markersize=4)
                plt.axhline(y=STOPPING_WIN_RATE_THRESHOLD, color='r', linestyle='--',
                            label=f'Stop Threshold ({STOPPING_WIN_RATE_THRESHOLD:.2f})')
                plt.title(
                    f'DQN Agent Win Rate vs Random Agent (Final - {TOTAL_EPISODES} EPs)')
                plt.xlabel('Episodes')
                plt.ylabel('Win Rate')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.ylim(-0.05, 1.05)
                plt.legend()
                if plot_episodes:
                    last_episode = plot_episodes[-1]
                    last_rate = plot_win_rates[-1]
                    plt.text(last_episode, last_rate, f' {
                             last_rate:.3f}', va='bottom' if last_rate < 0.9 else 'top')
                plt.tight_layout()
                plt.savefig(WIN_RATE_PLOT_FILE)
                plt.close()
                print(f"Final win rate plot saved to {WIN_RATE_PLOT_FILE}")
            except Exception as plot_err:
                print(
                    f"Warning: Could not generate or save final plot: {plot_err}")

        elif not plot_episodes:
            print(
                "No plotting data generated (PLOT_FREQ might be > TOTAL_EPISODES or training stopped early).")

    elif training_complete and not plot_episodes:  # Handle case: stopped early before first plot
        print("Training stopped early before the first plot interval could be completed.")

    print("Training Script Completed")
