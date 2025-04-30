import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Activation
import time
import sys
import os
import random

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from board import ConnectFourBoard
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure board.py is in the same directory.")
    sys.exit(1)

# Constants
PLAYER1 = 1
PLAYER2 = 2
EMPTY = 0

# Agent Perspective Constants
AGENT_ID = 1
OPPONENT_ID = -1

# Model File
LATEST_WEIGHTS_FILE = "models/connect4-dqn-vs-random-latest.weights.h5"


def translate_board_to_agent(board_state, agent_player_id):
    # This function remains the same, it correctly translates based on player IDs
    agent_board = np.zeros_like(board_state, dtype=np.float32)
    opponent_player_id = PLAYER2 if agent_player_id == PLAYER1 else PLAYER1
    agent_board[board_state == agent_player_id] = AGENT_ID
    agent_board[board_state == opponent_player_id] = OPPONENT_ID
    return np.expand_dims(agent_board, axis=-1)


def print_game_board(board_state):
    # --- MODIFIED: Flip the board vertically for standard display ---
    # This matches the behavior of board.py's own print_board method
    flipped_board = board_state  # np.flip(board_state, 0)
    print("\n  0   1   2   3   4   5   6 ")
    print("-----------------------------")
    rows, cols = flipped_board.shape
    for r in range(rows):
        print("|", end="")
        for c in range(cols):
            if flipped_board[r, c] == PLAYER1:
                print(" X ", end="|")  # Player 1 is X
            elif flipped_board[r, c] == PLAYER2:
                print(" O ", end="|")  # Player 2 is O
            else:
                print("   ", end="|")
        print("\n-----------------------------")
    print()


def get_human_move(valid_locations):
    # This function remains the same, relies on valid_locations list
    while True:
        try:
            col_str = input(
                f"Enter column (0-{ConnectFourBoard.COLUMN_COUNT-1}): ")
            col = int(col_str)
            if col in valid_locations:
                return col
            elif 0 <= col < ConnectFourBoard.COLUMN_COUNT:
                print("Column is full. Please choose another.")
            else:
                print(f"Invalid column number. Please enter a number between 0 and {
                      ConnectFourBoard.COLUMN_COUNT-1}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except EOFError:
            print("\nInput interrupted. Exiting.")
            sys.exit(0)
        except KeyboardInterrupt:
            print("\nGame interrupted by user. Exiting.")
            sys.exit(0)


# This function defines the model architecture - Ensure it matches training
def build_agent_model(state_shape, action_size):
    """ Builds the Convolutional Neural Network model (MUST MATCH TRAINING). """
    inputs = Input(shape=state_shape)  # e.g., (6, 7, 1)

    # Layer 1 (Matches _build_conv_model from dqn_agent.py)
    conv1 = Conv2D(filters=64, kernel_size=(3, 3),
                   padding='same', kernel_initializer='he_uniform')(inputs)
    act1 = Activation('relu')(conv1)

    # Layer 2 (Matches _build_conv_model)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3),
                   padding='same', kernel_initializer='he_uniform')(act1)
    act2 = Activation('relu')(conv2)

    # Layer 3 (Matches _build_conv_model)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3),
                   padding='same', kernel_initializer='he_uniform')(act2)
    act3 = Activation('relu')(conv3)

    # Flatten and Dense Layers (Matches _build_conv_model)
    flatten = Flatten()(act3)
    dense1 = Dense(256, activation='relu',
                   kernel_initializer='he_uniform')(flatten)

    # Output layer: Linear activation for Q-values (Matches _build_conv_model)
    outputs = Dense(action_size, activation='linear',
                    kernel_initializer='he_uniform')(dense1)

    model = Model(inputs=inputs, outputs=outputs)

    print("Built model architecture matching dqn_agent.py's _build_conv_model.")
    # model.summary() # Optional: print summary to double-check
    return model


def get_agent_move(model, board_state, agent_player_id, valid_locations):
    # This function remains the same, depends on correct board_state and valid_locations
    if not valid_locations:
        return None  # Should theoretically not happen if checked before calling

    state_agent_view = translate_board_to_agent(board_state, agent_player_id)
    state_input = np.expand_dims(state_agent_view, axis=0)
    q_values = model.predict(state_input, verbose=0)[0]

    q_values_masked = np.full(ConnectFourBoard.COLUMN_COUNT, -np.inf)
    q_values_masked[valid_locations] = q_values[valid_locations]

    action = np.argmax(q_values_masked)

    # Safety check remains important
    if action not in valid_locations:
        print(f"Warning: Agent chose invalid action {action} despite masking. Q-values: {
              q_values}. Masked: {q_values_masked}. Choosing random valid move.")
        action = random.choice(valid_locations)  # Fallback

    return action


if __name__ == "__main__":
    print("--- Connect Four: Human vs DQN Agent ---")

    # Define Model Shape Parameters (Ensure these match board.py)
    state_shape = (ConnectFourBoard.ROW_COUNT,
                   ConnectFourBoard.COLUMN_COUNT, 1)
    action_size = ConnectFourBoard.COLUMN_COUNT

    # Build Model Architecture
    try:
        model = build_agent_model(state_shape, action_size)
    except Exception as e:
        print(f"Error building the model architecture: {e}")
        sys.exit(1)

    # Load Weights
    if not os.path.exists(LATEST_WEIGHTS_FILE):
        print(f"Error: Weights file not found at '{LATEST_WEIGHTS_FILE}'")
        sys.exit(1)
    try:
        print(f"Loading weights from {LATEST_WEIGHTS_FILE} into the model...")
        model.load_weights(LATEST_WEIGHTS_FILE)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)

    # Initialize Game Environment
    env = ConnectFourBoard()
    turn = random.choice([PLAYER1, PLAYER2])  # Randomly decide who starts

    # Assign Player Roles
    human_player_id = -1
    agent_player_id = -1
    while human_player_id == -1:
        choice = input(
            "Do you want to be Player 1 ('X') or Player 2 ('O')? (1/2): ").strip()
        if choice == '1':
            human_player_id = PLAYER1
            agent_player_id = PLAYER2
            print("You are Player 1 (X). The agent is Player 2 (O).")
        elif choice == '2':
            human_player_id = PLAYER2
            agent_player_id = PLAYER1
            print("You are Player 2 (O). The agent is Player 1 (X).")
        else:
            print("Invalid choice. Please enter 1 or 2.")

    print(f"Player {turn} ('{'X' if turn == PLAYER1 else 'O'}') goes first.")

    while not env.game_over:  # Use env.game_over to control the loop
        current_board = env.get_board()
        print_game_board(current_board)
        valid_locations = env.get_valid_locations()

        if not valid_locations:
            print("Board full! It's a DRAW!")
            break

        action = -1
        if turn == human_player_id:
            print(
                f"Your turn (Player {turn} - '{'X' if turn == PLAYER1 else 'O'}')")
            action = get_human_move(valid_locations)

        else:
            print(f"Agent's turn (Player {
                  turn} - '{'X' if turn == PLAYER1 else 'O'}'). Thinking...")
            start_time = time.time()
            action = get_agent_move(
                model, current_board, agent_player_id, valid_locations)
            end_time = time.time()
            print(f"Agent chose column {
                  action} (calculation time: {end_time - start_time:.3f}s)")
            time.sleep(0.5)

        if action != -1:
            row = env.drop_piece(action, turn)

            if row == -1:
                print(f"Error: Attempted to drop piece in invalid column {
                      action}. Turn {turn}")
                continue

            if env.game_over:
                final_board = env.get_board()
                print_game_board(final_board)
                if env.winner is not None:
                    if env.winner == human_player_id:
                        print("Congratulations! You WIN!")
                    else:  # Agent won
                        print("The Agent WINS! Better luck next time.")
                else:  # No winner means it must be a draw
                    print("Board full! It's a DRAW!")

            else:
                turn = PLAYER2 if turn == PLAYER1 else PLAYER1

        else:
            # Should not happen
            print("Error: No action was selected despite available moves.")
            break

    if not env.game_over:
        # Should only happen if break was hit
        print("\nGame loop exited unexpectedly.")
    print("--- Game Over ---")
