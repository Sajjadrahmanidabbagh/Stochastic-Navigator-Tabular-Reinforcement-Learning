# By Sajjad - Spring 2026

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. Environment & Value Iteration Setup ---
GRID_HEIGHT = 10
GRID_WIDTH = 10
GOAL_STATE = (0, 9)

# ---Expanded obstacles and hazards using sets---
HAZARD_STATES = {
    (1, 8), (2, 8), (8, 2), (8, 3), (8, 4), 
    (4, 7), (5, 7), (9, 8)
}
WALL_STATES = {
    (1, 2), (2, 2), (3, 2), (4, 2), 
    (7, 7), (7, 8), (7, 9), 
    (5, 5), (6, 5), (7, 5), (3, 5)
}

ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Right, Left, Down, Up
DISCOUNT_FACTOR = 0.9
THETA = 0.001

V = np.zeros((GRID_HEIGHT, GRID_WIDTH))

def is_valid(state):
    r, c = state
    if r < 0 or r >= GRID_HEIGHT or c < 0 or c >= GRID_WIDTH or state in WALL_STATES:
        return False
    return True

def get_transitions(state, action):
    # Terminal states
    if state == GOAL_STATE or state in HAZARD_STATES:
        return [(1.0, state, 0)] 
    
    transitions = []
    # 80% chance intended direction
    intended_next = (state[0] + action[0], state[1] + action[1])
    if not is_valid(intended_next): intended_next = state
    
    reward = 1 if intended_next == GOAL_STATE else (-1 if intended_next in HAZARD_STATES else -0.04)
    transitions.append((0.8, intended_next, reward))
    
    # 20% slip dynamics (10% orthogonal)
    for slip_action in [(action[1], action[0]), (-action[1], -action[0])]:
        slip_next = (state[0] + slip_action[0], state[1] + slip_action[1])
        if not is_valid(slip_next): slip_next = state
        slip_reward = 1 if slip_next == GOAL_STATE else (-1 if slip_next in HAZARD_STATES else -0.04)
        transitions.append((0.1, slip_next, slip_reward))
        
    return transitions

def value_iteration():
    global V
    iteration = 0
    while True:
        delta = 0
        new_V = np.copy(V)
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                state = (r, c)
                if state in WALL_STATES or state == GOAL_STATE or state in HAZARD_STATES: 
                    continue
                
                action_values = []
                for action in ACTIONS:
                    val = sum([p * (rwd + DISCOUNT_FACTOR * V[ns[0], ns[1]]) 
                               for p, ns, rwd in get_transitions(state, action)])
                    action_values.append(val)
                new_V[r, c] = max(action_values)
                delta = max(delta, abs(V[r, c] - new_V[r, c]))
        V = new_V
        iteration += 1
        if delta < THETA: 
            print(f"Converged in {iteration} iterations.")
            break

# --- 2. Policy Extraction ---
def get_best_intended_path(start_state):
    """Calculates the greedy path assuming the robot never slips."""
    path = [start_state]
    current = start_state
    
    while current != GOAL_STATE and current not in HAZARD_STATES:
        best_val = -float('inf')
        best_next = current
        
        for action in ACTIONS:
            val = sum([p * (rwd + DISCOUNT_FACTOR * V[ns[0], ns[1]]) 
                       for p, ns, rwd in get_transitions(current, action)])
            
            if val > best_val:
                best_val = val
                intended = (current[0] + action[0], current[1] + action[1])
                best_next = intended if is_valid(intended) else current
                
        # Break if the policy loops (can happen if penalty/discount makes moving worse than staying)
        if best_next == current or len(path) > 50: 
            break
            
        current = best_next
        path.append(current)
        
    return path

# --- 3. Interactive Visualization ---
print("Running Value Iteration for 10x10 grid...")
value_iteration()

fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.1)
plt.title("10x10 Grid World: Click any cell to start the robot!")

# Generate abbreviated labels dynamically to fit the smaller cells
labels = np.empty((GRID_HEIGHT, GRID_WIDTH), dtype=object)
for r in range(GRID_HEIGHT):
    for c in range(GRID_WIDTH):
        val = round(V[r, c], 2)
        state = (r, c)
        if state == GOAL_STATE: labels[r, c] = f"G({val})"
        elif state in HAZARD_STATES: labels[r, c] = f"H({val})"
        elif state in WALL_STATES: labels[r, c] = "W"
        else: labels[r, c] = f"{val}"

# Draw Heatmap (smaller font size for 100 cells)
sns.heatmap(V, annot=labels, fmt="", cmap="YlGnBu", cbar=False, ax=ax, 
            linewidths=1, linecolor='black', annot_kws={"size": 8})

robot = patches.Circle((-1, -1), radius=0.3, color='red', zorder=10)
ax.add_patch(robot)

is_animating = False

def on_click(event):
    global is_animating
    if event.inaxes != ax or is_animating: return
    
    c, r = int(event.xdata), int(event.ydata)
    start_state = (r, c)
    
    if not is_valid(start_state) or start_state == GOAL_STATE or start_state in HAZARD_STATES:
        return

    is_animating = True
    path = get_best_intended_path(start_state)
    
    for state in path:
        robot.center = (state[1] + 0.5, state[0] + 0.5)
        fig.canvas.draw()
        plt.pause(0.2) # Slightly faster animation for larger grid
        
    is_animating = False

fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()
