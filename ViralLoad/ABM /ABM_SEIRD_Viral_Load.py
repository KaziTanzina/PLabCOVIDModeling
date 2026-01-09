import numpy as np
import random
import matplotlib.pyplot as plt

# ---------------- Model parameters ----------------
num_agents = 500          # Number of agents in the simulation
num_infected = 5          # Number of initially infected agents
infection_rate = 0.005    # Probability of transmission per contact (proxy for beta * S/N in ABM contact form)
latentperiod = 5
infectious_period = 14    # (kept for reference; not explicitly used in this viral-load rule set)
time_steps = 100          # Number of time steps in the simulation

# Viral-load thresholds
thresh1 = 0.05  # S -> E threshold
thresh2 = 0.9   # E -> I threshold (see note in E-state logic)
thresh3 = 0.2   # I -> R threshold

# NEW: death rate delta (per-infected-agent per time step)
# This implements: I_dot = beta * S/N * I - (gamma + delta) * I, and D_dot = sum(delta * I)
death_rate = 0.002  # delta

# ---------------- Agent definition ----------------
class Agent:
    def __init__(self, state, viralload):
        self.state = state
        self.days_in_compartment = 0
        self.viralload = viralload

    def update_state(self, neighbors):
        # Absorbing state
        if self.state == 'D':
            return

        if self.state == 'S':
            self.days_in_compartment += 1
            for neighbor in neighbors:
                if neighbor.state == 'I' and random.random() < infection_rate:
                    self.viralload += random.random() / 3
                    if self.viralload > thresh1:
                        self.state = 'E'
                        self.days_in_compartment = 0
                        break

        elif self.state == 'E':
            self.days_in_compartment += 1
            self.viralload += random.random() / 3

            # NOTE: original code had condition "if days_in_compartment < latentperiod and viralload > thresh2"
            # which triggers I early; kept as-is to preserve your rule set.
            if self.days_in_compartment < latentperiod and self.viralload > thresh2:
                self.state = 'I'
                self.days_in_compartment = 0
            elif self.days_in_compartment >= latentperiod:
                self.state = 'R'
                self.days_in_compartment = 0

        elif self.state == 'I':
            self.days_in_compartment += 1

            # NEW: death event from I with probability death_rate per step
            if random.random() < death_rate:
                self.state = 'D'
                self.days_in_compartment = 0
                self.viralload = 0.0
                return

            # If not dead, continue viral-load decay and recovery rule
            self.viralload -= random.random() / 3
            if self.viralload <= thresh3:
                self.state = 'R'
                self.days_in_compartment = 0

        elif self.state == 'R':
            self.days_in_compartment += 1
            # Reset viral load (immune)
            self.viralload -= self.viralload
            self.days_in_compartment = 0

    def get_state(self):
        return self.state


# ---------------- Simulation ----------------
def simulate():
    agents = []
    for i in range(num_agents):
        if i < num_infected:
            state = 'I'
            viralload = (thresh2 + thresh3) / 2
        else:
            state = 'S'
            viralload = 0.0
        agents.append(Agent(state, viralload))

    state_counts = []
    for t in range(time_steps):
        for agent in agents:
            neighbors = [neighbor for neighbor in agents if neighbor != agent]
            agent.update_state(neighbors)

        s_count = sum(1 for a in agents if a.get_state() == 'S')
        e_count = sum(1 for a in agents if a.get_state() == 'E')
        i_count = sum(1 for a in agents if a.get_state() == 'I')
        r_count = sum(1 for a in agents if a.get_state() == 'R')
        d_count = sum(1 for a in agents if a.get_state() == 'D')
        state_counts.append([s_count, e_count, i_count, r_count, d_count])

    return np.array(state_counts)


# ---------------- Run + plot ----------------
state_counts = simulate()

s_counts = state_counts[:, 0]
e_counts = state_counts[:, 1]
i_counts = state_counts[:, 2]
r_counts = state_counts[:, 3]
d_counts = state_counts[:, 4]

plt.figure(figsize=(10, 8))
plt.plot(s_counts, label='Susceptible')
plt.plot(e_counts, label='Exposed')
plt.plot(i_counts, label='Infected')
plt.plot(r_counts, label='Recovered')
plt.plot(d_counts, label='Dead')
plt.xlabel('Time steps')
plt.ylabel('Number of agents')
plt.title('Agent-based SEIRD simulation')
plt.legend()
plt.show()
