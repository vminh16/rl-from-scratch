

import pathlib
import pickle
import numpy as np
from minirl.envs.tictactoe import TicTacToe
from minirl.agents.td_value import AfterstateTD
from minirl.core import run_episode
from minirl.utils.seeding import make_rng


def render_board(board: np.ndarray) -> None:
    """
    This function is used to print the board in a readable format
    """
    symbols = {1: "X", -1: "O", 0: "."}
    print("\n   Board:       index:")
    for r in range(3):

        row_symbols = " | ".join(symbols[board[r * 3 + c]] for c in range(3))
        row_indices = " | ".join(str(r * 3 + c) for c in range(3))
        print(f"     {row_symbols}             {row_indices}")
        if r < 2:
            print("    -----------           -----------")
    print()


def get_trained_agent() -> AfterstateTD:
    """ 
    load trained agent or train if not found
    """
    model_path = pathlib.Path(__file__).parent / "results" / "agent_v_table.pkl"
    agent = AfterstateTD(alpha=0.0, epsilon=0.0)  

    if model_path.exists():
        print(f"Loading trained agent from {model_path}...")
        with open(model_path, "rb") as f:
            agent.v_table = pickle.load(f)
    else:
        print("Not found trained agent. Auto training 20,000 episodes...")
        rng = make_rng(42)
        env = TicTacToe()
        train_agent = AfterstateTD(alpha=0.1, epsilon=0.1)
        for ep in range(20_000):
            run_episode(env, train_agent, rng, train=True)
        agent.v_table = train_agent.v_table
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(agent.v_table, f)
        print("Training and save model completed!")

    return agent


def play_game():
    agent = get_trained_agent()
    env = TicTacToe()
    rng = make_rng(123)

    print("\n==========================================")
    print("      WELCOME TO TIC-TAC-TOE!           ")
    print("   AI is [ X ], You is [ O ]            ")
    print("==========================================")

    board = np.zeros(9, dtype=int)
    render_board(board)

    while True:
        # 1. Agent (AI) turn
        print("AI is thinking...")
        ai_action = agent.act(board, rng, greedy=True)
        board[ai_action] = 1
        print(f"--> AI played at: {ai_action}")
        render_board(board)

        # Check AI win
        if env.check_win(board) == 1:
            print("Agent has won!\n")
            break

        # Check draw
        if not np.any(board == 0):
            print("Draw!\n")
            break

        # 2. User turn
        while True:
            try:
                user_input = input("Your turn (input 0-8): ").strip()
                user_action = int(user_input)
                if user_action < 0 or user_action > 8:
                    print("Invalid cell! Please input from 0 to 8.")
                    continue
                if board[user_action] != 0:
                    print("Cell is occupied! Please choose another cell.")
                    continue
                break
            except ValueError:
                print("Please enter a valid integer!")

        board[user_action] = -1
        render_board(board)

        # Check user win
        if env.check_win(board) == -1:
            print("You win!\n")
            break

        # Check draw
        if not np.any(board == 0):
            print("Draw!\n")
            break


if __name__ == "__main__":
    play_game()
