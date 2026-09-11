# Stochastic-Navigator-Tabular-Reinforcement-Learning
A Python-based interactive reinforcement learning visualization of Value Iteration in a stochastic grid world that visualizes Bellman updates and greedy policy extraction in real-time.

This is assignment #3 for the machine learning course (reinforcement learning module).

Problem Statement (for Students)
In this assignment, you will program the decision-making logic of an autonomous delivery robot navigating a hazardous 10x10 warehouse to reach its charging station. The challenge is that the environment is stochastic: the warehouse floor is slippery, so the robot has an 80% chance of moving in its intended direction and a 20% chance of sliding orthogonally into an adjacent cell. Your task is to implement the Value Iteration algorithm to solve this Markov Decision Process (MDP). You must systematically calculate the expected long-term reward for every possible state in the grid so that the robot can safely avoid walls and hazards, and dynamically determine the optimal path to the goal from any starting location.

The Solution & Code Architecture
The provided Python codebase solves this environment using a tabular, bottom-up Dynamic Programming approach based on the Bellman Optimality Equation. The algorithm initializes a zero-valued state matrix and continuously sweeps across the 10x10 grid, updating each cell's value by calculating the maximum expected return for all possible actions (factoring in slip probabilities, step penalties, and the discount factor) until the matrix converges. Beyond the mathematical solver, the code utilizes seaborn and matplotlib to translate the final value matrix into a color-coded heatmap. It incorporates event listeners to make the environment interactive: when a user clicks any valid starting cell on the heatmap, the script calculates the greedy policy on the fly and animates a robot marker tracing the optimal path to the goal.
