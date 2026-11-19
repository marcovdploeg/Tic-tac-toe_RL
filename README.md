# Tic-tac-toe RL

This repository contains code for an agent that can be trained with reinforcement learning (RL)
to play the game tic-tac-toe.
There is a Jupyter Notebook which explains how it was implemented and a Python script that is easy to run.

The agent is trained using a Q-table, which contains the expected future reward for each action the agent can take
given a board state.
Different training modes were tested, both against a completely random opponent and against a (partially random)
pretrained agent.

You can run the Python script to play, which automatically trains the agent with the best training mode
(random opponent) for 5000 training rounds.
This will result in a decently trained agent, but you should still be able to beat it, as it will make mistakes.
Against random opponents, it has a win/draw/lose rate of about 83/10/7%.
I found that for about 500,000 training rounds, it becomes quite difficult to still win.
At this point it has a win/draw/lose rate of about 92/7/1% against random opponents.
I included a pickle file with the Q-table for this agent with the name "Q\_table\_500k.pkl";
if this is passed as the filename you can play against it.

Applied methods in these scripts are: object-oriented programming, saving and loading objects using pickle,
reinforcement learning with a Q-table.
