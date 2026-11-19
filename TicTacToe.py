import numpy as np
from pathlib import Path
import pickle

class Board():
    def __init__(self):
        self.board = [" ", " ", " ",
                      " ", " ", " ",
                      " ", " ", " "]

    def display(self):
        print(self.board[0] + " | " + self.board[1] + " | " + self.board[2])
        print("-- --- --")
        print(self.board[3] + " | " + self.board[4] + " | " + self.board[5])
        print("-- --- --")
        print(self.board[6] + " | " + self.board[7] + " | " + self.board[8])

    def place(self, mark: str, position: int):
        if mark not in ["X", "O"]:
            raise ValueError("Mark must be X or O")
        if self.board[position] != " ":
            raise ValueError("Position is not empty")
        self.board[position] = mark

class Agent:
    def __init__(
        self,
        reward_win: int = 1,
        reward_draw: int = 0,
        reward_loss: int = -1,
        reward_move: int = 0,
        alpha: float = 0.1,
        gamma: float = 0.9,
        num_episodes: int = 5000,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.1,
        filename: str = "Q_table.pkl",
        force_train: bool = False,
        training_mode: str = "random"
    ):
        # Training mode can be: random, agent, pretrained agent, pretrained agent stacked, random X
        self.reward_win = reward_win
        self.reward_draw = reward_draw
        self.reward_loss = reward_loss
        self.reward_move = reward_move
        self.alpha = alpha
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        # Get the Q-table from a file if it exists,
        # start empty for training if not or if forced
        self.filename = filename
        Q_path = Path(self.filename)
        if Q_path.exists() and not force_train:
            with open(self.filename, 'rb') as f:
                self.Q = pickle.load(f)
        else:
            self.Q = {}
        self.force_train = force_train
        self.training_mode = training_mode
    
    def place_mark_random(self, board_state: Board) -> int:
        # For the random opponent
        empty_indices = np.where(np.array(board_state.board, dtype=str) == " ")[0]
        return np.random.choice(empty_indices)

    def place_mark(self, board_state: Board, epsilon: float) -> int:
        # Epsilon probability for random move, also if state not encountered yet
        state_str = "".join(board_state.board)
        empty_indices = np.where(np.array(board_state.board, dtype=str) == " ")[0]
        if np.random.random() < epsilon or state_str not in self.Q:
            return np.random.choice(empty_indices)
        else:
            # Make sure that returned action index is of an empty position
            placement_index = empty_indices[0]
            best_Q_value = self.Q[state_str][placement_index]
            # Need this in case of ties in Q (e.g. multiple zeros),
            # to select randomly instead of the first
            best_Q_values_tied = [best_Q_value]
            indices_tied = [placement_index]

            for index in empty_indices[1:]:
                Q_value = self.Q[state_str][index]
                if Q_value > best_Q_value:
                    placement_index = index
                    best_Q_value = Q_value
                    # Reset tied lists
                    best_Q_values_tied = [best_Q_value]
                    indices_tied = [placement_index]
                elif Q_value == best_Q_value:
                    best_Q_values_tied.append(Q_value)
                    indices_tied.append(index)

            if len(best_Q_values_tied) == 1:
                # Just return the corresponding index
                return int(placement_index)
            else:
                # Randomly select from tied indices
                return np.random.choice(indices_tied)

    def Q_update(self, state: list, action: int, reward: int, next_state: list):
        state_str = "".join(state)
        next_state_str = "".join(next_state)

        # Check if state and next_state already exist, append zeros if not
        if state_str not in self.Q:
            self.Q[state_str] = [0.0]*9
        if next_state_str not in self.Q:
            self.Q[next_state_str] = [0.0]*9

        # Actual update
        self.Q[state_str][action] = (
            self.Q[state_str][action] + self.alpha *
            (reward + self.gamma * np.max(self.Q[next_state_str]) - self.Q[state_str][action])
        )

    def save_Q_table(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.Q, f)

    def train(self):
        agent_X = Agent()  # placeholder
        if "pretrained" in self.training_mode:
            # Prepare agent X with random training
            agent_X = Agent(num_episodes=5000, filename=self.filename[:-4]+"X.pkl", training_mode="random X")
            if not agent_X.Q:
                agent_X.train()

        for episode in range(self.num_episodes):
            # Pick linear or exponential epsilon decay
            #epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (episode / self.num_episodes)
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-episode/(0.1*self.num_episodes))
            
            # Play the game, initialized to a new board each iteration
            game = Game()
            if self.training_mode == "random":
                game.play_for_training_vs_random(self, epsilon)
            elif self.training_mode == "agent":
                game.play_for_training_vs_agent(self, epsilon)
            elif self.training_mode == "pretrained agent":
                game.play_for_training_vs_pretrained_agent(self, agent_X, epsilon)
            elif self.training_mode == "pretrained agent stacked":
                for epsilon_X in [1.0, 0.75, 0.5, 0.25, 0.0]:
                    game.play_for_training_vs_pretrained_agent(self, agent_X, epsilon, epsilon_X)
                    game = Game()  # need to reset the board each loop
            elif self.training_mode == "random X":
                game.play_for_training_vs_random_X(self, epsilon)
            else:
                raise ValueError("Invalid training mode")

        # Save the filled Q-table
        self.save_Q_table()

    def run(self):
        # Check if the agent has to be trained
        if not self.Q or self.force_train:
            print("Commencing agent training")
            self.train()
            print("Training complete")

        # Play the game
        print("Now starting a game")
        game = Game()
        if "X" not in self.training_mode:
            game.play_with_agent(self)
        else:
            game.play_with_agent_X(self)
    
    def test(self, n_rounds: int = 10000):
        # Evaluate how well the agent plays the game
        # against a random opponent
        n_wins = 0
        n_draws = 0
        n_losses = 0
        for _ in range(n_rounds):
            game = Game()
            outcome = game.play_test(self)
            if outcome == 1:
                n_wins += 1
            elif outcome == 0:
                n_draws += 1
            else:
                n_losses += 1
        print(f"Results: {n_wins/n_rounds*100:.2f}% wins, {n_draws/n_rounds*100:.2f}% draws, {n_losses/n_rounds*100:.2f}% losses")

class Game:
    def __init__(self):
        self.board_state = Board()

    def ask_mark_position(self) -> int:
        while True:
            mark_index = input("Choose in which position (1-9) to place your mark: ")
            if not mark_index.isdigit():
                print("Please enter an integer.")
                continue
            
            mark_index = int(mark_index)
            if mark_index < 1 or mark_index > 9:
                print("Please enter a number between 1 and 9.")
            elif self.board_state.board[mark_index - 1] != " ":
                print("Please choose an empty position.")
            else:
                break
        return mark_index - 1

    def check_game_over(self, mark: str) -> bool:
        win_combinations = [
            (0,1,2), (3,4,5), (6,7,8),
            (0,3,6), (1,4,7), (2,5,8),
            (0,4,8), (2,4,6)
        ]
        mark_indices = np.where(np.array(self.board_state.board, dtype=str) == mark)[0]
        for combination in win_combinations:
            if combination[0] in mark_indices and combination[1] in mark_indices and combination[2] in mark_indices:
                return True
        return False

    # Play with two players in the terminal
    def play(self):
        turns = 0
        while turns < 9:
            self.board_state.display()
            if turns % 2 == 0:
                print("X's turn")
                self.board_state.place("X", self.ask_mark_position())
                if self.check_game_over("X"):
                    print()
                    print("X wins!")
                    self.board_state.display()
                    break
            else:
                print("O's turn")
                self.board_state.place("O", self.ask_mark_position())
                if self.check_game_over("O"):
                    print()
                    print("O wins!")
                    self.board_state.display()
                    break
            turns += 1
            print()

        if turns == 9:
            print("Draw!")
            self.board_state.display()

    # Play against the agent, where you are X, and the agent is O
    def play_with_agent(self, agent: Agent):
        turns = 0
        while turns < 9:
            self.board_state.display()
            if turns % 2 == 0:
                print("X's turn")
                self.board_state.place("X", self.ask_mark_position())
                if self.check_game_over("X"):
                    print()
                    print("X wins!")
                    self.board_state.display()
                    break
            else:
                print("O's turn")
                # Extract the agent's best play with epsilon=0
                placement_index = agent.place_mark(self.board_state, 0.0)
                print(f"Agent places O at position {placement_index + 1}")
                self.board_state.place("O", placement_index)
                if self.check_game_over("O"):
                    print()
                    print("O wins!")
                    self.board_state.display()
                    break
            turns += 1
            print()

        if turns == 9:
            print("Draw!")
            self.board_state.display()

    # Play against the agent, where you are O, and the agent is X
    def play_with_agent_X(self, agent: Agent):
        turns = 0
        while turns < 9:
            self.board_state.display()
            if turns % 2 == 0:
                print("X's turn")
                # Extract the agent's best play with epsilon=0
                placement_index = agent.place_mark(self.board_state, 0.0)
                print(f"Agent places X at position {placement_index + 1}")
                self.board_state.place("X", placement_index)
                if self.check_game_over("X"):
                    print()
                    print("X wins!")
                    self.board_state.display()
                    break
            else:
                print("O's turn")
                self.board_state.place("O", self.ask_mark_position())
                if self.check_game_over("O"):
                    print()
                    print("O wins!")
                    self.board_state.display()
                    break
            turns += 1
            print()

        if turns == 9:
            print("Draw!")
            self.board_state.display()

    # Play for training, with the agent vs random choices, without prints
    def play_for_training_vs_random(self, agent: Agent, epsilon: float):
        turns = 0
        # Initialize before and after states needed to update Q properly
        state_before = []
        state_after = []
        placement_index_agent_before = 0
        placement_index_agent_after = 0
        # Set reward for move, overwritten if game ends
        current_reward = agent.reward_move
        while turns < 9:
            if turns % 2 == 0:
                # X, random choice
                placement_index_random = agent.place_mark_random(self.board_state)
                self.board_state.place("X", placement_index_random)
                if self.check_game_over("X"):
                    current_reward = agent.reward_loss
                    break
            else:
                # O, agent
                # If this is the first move, meaning state_before (and state_after) is [], we set state_before;
                # else state_before stays what is was and we update state_after
                # These are always the board states after the opponent has moved
                # Something similar needs to be done for the move the agent makes, so that it matches state_before
                if not state_before:
                    state_before = self.board_state.board.copy()
                else:
                    state_after = self.board_state.board.copy()
                    placement_index_agent_before = placement_index_agent_after
                
                placement_index_agent_after = agent.place_mark(self.board_state, epsilon)
                self.board_state.place("O", placement_index_agent_after)
                
                # Update Q-table with move reward, if it is not the first move, so state_after is not [] anymore;
                # afterwards state_before can be updated to the current state_after for the next iteration
                if state_after:
                    agent.Q_update(state_before, placement_index_agent_before, current_reward, state_after)
                    state_before = state_after

                # Note it is important to put this check after the Q update,
                # to get the correct final update
                if self.check_game_over("O"):
                    current_reward = agent.reward_win
                    break
            turns += 1

        if turns == 9:
            current_reward = agent.reward_draw

        # Update Q-table with the final result to reflect a win/loss/draw;
        # in each case state_after still needs to be updated (for a draw the opponent moves last)
        # We can just use the agents move after, which is the last one done and has not been used yet
        state_after = self.board_state.board.copy()
        agent.Q_update(state_before, placement_index_agent_after, current_reward, state_after)

    # Play for training, with the agent vs random choice, where agent is X
    def play_for_training_vs_random_X(self, agent: Agent, epsilon: float):
        turns = 0
        # Initialize before and after states needed to update Q properly
        state_before = []
        state_after = []
        placement_index_agent_before = 0
        placement_index_agent_after = 0
        # Set reward for move, overwritten if game ends
        current_reward = agent.reward_move
        while turns < 9:
            if turns % 2 == 0:
                # X, agent
                # If this is the first move, meaning state_before (and state_after) is [], we set state_before;
                # else state_before stays what is was and we update state_after
                # These are always the board states after the opponent has moved
                # Something similar needs to be done for the move the agent makes, so that it matches state_before
                if not state_before:
                    state_before = self.board_state.board.copy()
                else:
                    state_after = self.board_state.board.copy()
                    placement_index_agent_before = placement_index_agent_after
                
                placement_index_agent_after = agent.place_mark(self.board_state, epsilon)
                self.board_state.place("X", placement_index_agent_after)
                
                # Update Q-table with move reward, if it is not the first move, so state_after is not [] anymore;
                # afterwards state_before can be updated to the current state_after for the next iteration
                if state_after:
                    agent.Q_update(state_before, placement_index_agent_before, current_reward, state_after)
                    state_before = state_after

                # Note it is important to put this check after the Q update,
                # to get the correct final update
                if self.check_game_over("X"):
                    current_reward = agent.reward_win
                    break
            else:
                # O, random choice
                placement_index_random = agent.place_mark_random(self.board_state)
                self.board_state.place("O", placement_index_random)
                if self.check_game_over("O"):
                    current_reward = agent.reward_loss
                    break
            turns += 1

        if turns == 9:
            current_reward = agent.reward_draw

        # Update Q-table with the final result to reflect a win/loss/draw;
        # in each case state_after still needs to be updated (for a draw the opponent moves last)
        # We can just use the agents move after, which is the last one done and has not been used yet
        state_after = self.board_state.board.copy()
        agent.Q_update(state_before, placement_index_agent_after, current_reward, state_after)
    
    # Play for training, with agent playing itself
    def play_for_training_vs_agent(self, agent: Agent, epsilon: float):
        turns = 0
        # Initialize before and after states needed to update Q properly
        state_before_O = []
        state_after_O = []
        state_before_X = []
        state_after_X = []
        placement_index_agent_before_O = 0
        placement_index_agent_after_O = 0
        placement_index_agent_before_X = 0
        placement_index_agent_after_X = 0
        # Set reward for move, overwritten if game ends
        current_reward_O = agent.reward_move
        current_reward_X = agent.reward_move
        while turns < 9:
            if turns % 2 == 0:
                # X, "opponent" agent
                # Following the same logic as O now
                if not state_before_X:
                    state_before_X = self.board_state.board.copy()
                else:
                    state_after_X = self.board_state.board.copy()
                    placement_index_agent_before_X = placement_index_agent_after_X

                placement_index_agent_after_X = agent.place_mark(self.board_state, epsilon)
                self.board_state.place("X", placement_index_agent_after_X)

                if state_after_X:
                    agent.Q_update(state_before_X, placement_index_agent_before_X, current_reward_X, state_after_X)
                    state_before_X = state_after_X
                
                if self.check_game_over("X"):
                    current_reward_O = agent.reward_loss
                    current_reward_X = agent.reward_win
                    break
            else:
                # O, agent
                # If this is the first move, meaning state_before (and state_after) is [], we set state_before;
                # else state_before stays what is was and we update state_after
                # These are always the board states after the opponent has moved
                # Something similar needs to be done for the move the agent makes, so that it matches state_before
                if not state_before_O:
                    state_before_O = self.board_state.board.copy()
                else:
                    state_after_O = self.board_state.board.copy()
                    placement_index_agent_before_O = placement_index_agent_after_O
                
                placement_index_agent_after_O = agent.place_mark(self.board_state, epsilon)
                self.board_state.place("O", placement_index_agent_after_O)
                
                # Update Q-table with move reward, if it is not the first move, so state_after is not [] anymore;
                # afterwards state_before can be updated to the current state_after for the next iteration
                if state_after_O:
                    agent.Q_update(state_before_O, placement_index_agent_before_O, current_reward_O, state_after_O)
                    state_before_O = state_after_O

                # Note it is important to put this check after the Q update,
                # to get the correct final update
                if self.check_game_over("O"):
                    current_reward_O = agent.reward_win
                    current_reward_X = agent.reward_loss
                    break
            turns += 1

        if turns == 9:
            current_reward_O = agent.reward_draw
            current_reward_X = agent.reward_draw

        # Update Q-table with the final result to reflect a win/loss/draw;
        # in each case state_after still needs to be updated (for a draw the opponent moves last)
        # We can just use the agents move after, which is the last one done and has not been used yet
        state_after_O = self.board_state.board.copy()
        agent.Q_update(state_before_O, placement_index_agent_after_O, current_reward_O, state_after_O)
        state_after_X = self.board_state.board.copy()
        agent.Q_update(state_before_X, placement_index_agent_after_X, current_reward_X, state_after_X)

    # Play for training, with the agent playing another agent trained against random choices
    def play_for_training_vs_pretrained_agent(self, agent_O: Agent, agent_X: Agent, epsilon: float, epsilon_X: float = 0.0):
        turns = 0
        # Initialize before and after states needed to update Q properly
        state_before_O = []
        state_after_O = []
        placement_index_agent_before_O = 0
        placement_index_agent_after_O = 0
        # Set reward for move, overwritten if game ends
        current_reward_O = agent_O.reward_move
        while turns < 9:
            if turns % 2 == 0:
                # X, "opponent" agent
                # Extract the best play with epsilon=0; but not all the time to keep randomness as well
                placement_index = agent_X.place_mark(self.board_state, epsilon_X)
                self.board_state.place("X", placement_index)
                
                if self.check_game_over("X"):
                    current_reward_O = agent_O.reward_loss
                    break
            else:
                # O, agent
                # If this is the first move, meaning state_before (and state_after) is [], we set state_before;
                # else state_before stays what is was and we update state_after
                # These are always the board states after the opponent has moved
                # Something similar needs to be done for the move the agent makes, so that it matches state_before
                if not state_before_O:
                    state_before_O = self.board_state.board.copy()
                else:
                    state_after_O = self.board_state.board.copy()
                    placement_index_agent_before_O = placement_index_agent_after_O
                
                placement_index_agent_after_O = agent_O.place_mark(self.board_state, epsilon)
                self.board_state.place("O", placement_index_agent_after_O)
                
                # Update Q-table with move reward, if it is not the first move, so state_after is not [] anymore;
                # afterwards state_before can be updated to the current state_after for the next iteration
                if state_after_O:
                    agent_O.Q_update(state_before_O, placement_index_agent_before_O, current_reward_O, state_after_O)
                    state_before_O = state_after_O

                # Note it is important to put this check after the Q update,
                # to get the correct final update
                if self.check_game_over("O"):
                    current_reward_O = agent_O.reward_win
                    break
            turns += 1

        if turns == 9:
            current_reward_O = agent_O.reward_draw

        # Update Q-table with the final result to reflect a win/loss/draw;
        # in each case state_after still needs to be updated (for a draw the opponent moves last)
        # We can just use the agents move after, which is the last one done and has not been used yet
        state_after_O = self.board_state.board.copy()
        agent_O.Q_update(state_before_O, placement_index_agent_after_O, current_reward_O, state_after_O)

    # Play to test how well the agent performs
    def play_test(self, agent: Agent) -> int:
        outcome = 0
        turns = 0
        while turns < 9:
            if turns % 2 == 0:
                # X, random choice
                placement_index_random = agent.place_mark_random(self.board_state)
                self.board_state.place("X", placement_index_random)
                if self.check_game_over("X"):
                    outcome = -1
                    break
            else:
                # O, agent
                placement_index = agent.place_mark(self.board_state, 0.0)
                self.board_state.place("O", placement_index)
                if self.check_game_over("O"):
                    outcome = 1
                    break
            turns += 1
        return outcome

if __name__ == "__main__":
    agent = Agent()
    agent.run()
    agent.test()