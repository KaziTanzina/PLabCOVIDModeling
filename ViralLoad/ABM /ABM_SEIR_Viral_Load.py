# abm_viral_load.py
# ---------------------------------------------
# Agent-based SEIRD with age-structured contacts and viral-load dynamics
# - Keeps original variable names where possible
# - Fixes: E->I transition, initial counts, partner-selection guard,
#          per-age averaging across simulations, no global races
# - Adds: SHOW_PLOTS toggle + save_and_show() helper for all plots
# ---------------------------------------------

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import time
import math
# import concurrent.futures
# from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# Model parameters (your names)
# -----------------------------
num_agents = 500             # total agents
num_exposed = 20             # initially exposed
num_infected = 20            # initially infected
num_recovered = 20           # initially recovered

latent_period = 9            # days in Exposed before Infectious (upper bound)
infectious_period = 14       # max days in Infectious before forced recovery
time_steps = 120             # simulation steps (days)

# Optional: set to >0 to use probabilistic infection instead of "viral-load push"
infection_rate = 0.0         # 0.0 keeps your original push mechanism

# Show plots interactively?
SHOW_PLOTS = True  # set False if you want to only save files

# Age groups & proportions (census-like)
age_groups = ['0-4', '5-14', '15-19', '20-39', '40-59', '60-69', '70-100']
age_probs  = [0.057, 0.122, 0.064, 0.256, 0.262, 0.120, 0.113]

# Per-step death hazard by age group (kept simple as in your code)
death_rates = [0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007]

# Immunosenescence (higher = better clearance; we keep your mapping)
immunosenescence_factors = [0.95, 0.7, 0.6, 0.3, 0.3, 0.2, 0.15]

# Viral load thresholds (individualized per agent with jitter)
thresh1 = 0.05   # S -> E if VL exceeds this
thresh2 = 0.40   # E -> I trigger (or by latent_period)
thresh3 = 0.90   # switch I from rising to falling phase
thresh4 = 0.20   # I -> R when VL drops below this

# Output directory
primary_directory = "Primary ABM Model Directory"
os.makedirs(primary_directory, exist_ok=True)

# ----------------------------------------
# Helper functions (keep things organized)
# ----------------------------------------
def clamp01(x: float) -> float:
    """Clamp viral load into [0, 1] so thresholds are interpretable."""
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

def age_group_index_for_age(age: int) -> int:
    for idx, grp in enumerate(age_groups):
        lo, hi = map(int, grp.split('-'))
        if lo <= age <= hi:
            return idx
    raise ValueError(f"Age {age} not in any group")

def pick_age_in_group(idx: int) -> int:
    lo, hi = map(int, age_groups[idx].split('-'))
    return random.randint(lo, hi)

def save_and_show(path: str):
    """Save current matplotlib figure, then show or close based on flag."""
    plt.savefig(path)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

# -----------------------
# Agent with VL dynamics
# -----------------------
class Agent:
    """
    Viral-load engine (per agent):
      S: VL pushed by contact; if VL > threshold1 -> E
      E: VL rises slowly; E -> I if (VL > threshold2) OR (days_exposed >= latent_period)
      I: VL rises until > threshold3, then "falling" (decays). Death hazard each step.
         I -> R if VL <= threshold4 or after infectious_period days
      R: VL decays to 0 (reinfection off by default)
      D: VL = 0
    VL is clamped to [0, 1].
    """
    def __init__(self, state, viralload, age):
        self.state = state
        self.days_exposed = 0
        self.days_infected = 0
        self.viralload = clamp01(viralload)
        self.immune_days = 0
        self.age = age
        self.is_dead = False

        self.age_group_index = age_group_index_for_age(age)
        self.immunosenescence_factor = immunosenescence_factors[self.age_group_index]

        # Agent-specific thresholds (adds heterogeneity)
        self.threshold1 = thresh1 + ((random.random() - 0.5) * thresh1 * 0.75)
        self.threshold2 = thresh2 + ((random.random() - 0.5) * thresh2 * 0.75)
        self.threshold3 = thresh3 + ((random.random() - 0.5) * thresh3 * 0.75)
        self.threshold4 = thresh4 + ((random.random() - 0.5) * thresh4 * 0.75)

        self.viral_load_history = []
        self.falling_viral_load = False  # switches once VL > threshold3

    def update_state(self, deaths_by_ages):
        if self.state == 'S':
            # Contact mechanics handled outside; here we only check threshold to move S->E
            if self.viralload > self.threshold1:
                self.state = 'E'
                self.days_exposed = 0

        elif self.state == 'E':
            self.days_exposed += 1
            # slow drift up while exposed
            self.viralload = clamp01(self.viralload + random.random() / 10)

            # FIXED: SEIR progression — go to I on threshold or after latent period
            if self.viralload > self.threshold2 or self.days_exposed >= latent_period:
                self.state = 'I'
                self.days_infected = 0
                if self.viralload < self.threshold2:
                    self.viralload = self.threshold2 + 1e-3

        elif self.state == 'I':
            self.days_infected += 1

            if not self.falling_viral_load:
                # rising phase
                self.viralload = clamp01(self.viralload + random.random() / 3)
                if self.viralload > self.threshold3:
                    self.falling_viral_load = True
            else:
                # falling phase (older -> slower clearance)
                self.viralload = clamp01(self.viralload - random.random() * self.immunosenescence_factor)

            # per-step death hazard (kept simple; you can make it depend on VL)
            if random.random() < death_rates[self.age_group_index]:
                self.is_dead = True

            if self.is_dead:
                self.state = 'D'
                self.viralload = 0.0
                deaths_by_ages[self.age_group_index] += 1
            else:
                # recover when VL falls enough or after max infectious duration
                if self.viralload <= self.threshold4 or self.days_infected >= infectious_period:
                    self.state = 'R'

        elif self.state == 'R':
            if self.viralload > 0:
                self.viralload = clamp01(self.viralload - (random.random() * self.immunosenescence_factor) / 3)
            # reinfection option (off)
            # if self.immune_days >= immune_period:
            #     self.state = 'S'; self.immune_days = 0
            # else:
            #     self.immune_days += 1

        elif self.state == 'D':
            self.viralload = 0.0

        # record history if positive
        if self.viralload > 0:
            self.viral_load_history.append(self.viralload)

    # helpers used elsewhere
    def get_state(self): return self.state
    def get_age(self):   return self.age
    def die(self):       self.is_dead = True

# -----------------------
# One simulation run
# -----------------------
def simulate(simulation_number: int, seed_base: int = 12345):
    random.seed(seed_base + simulation_number)
    np.random.seed(seed_base + simulation_number)

    # Allocate population across age groups
    agents_per_age_group = [math.floor(w * num_agents) for w in age_probs]
    if sum(agents_per_age_group) < num_agents:
        agents_per_age_group[-1] += num_agents - sum(agents_per_age_group)
    cumulative_agents_per_group = np.cumsum(agents_per_age_group)

    agents = []
    deaths_by_ages = [0] * len(death_rates)

    # Initialize agent states
    for i in range(num_agents):
        if i < num_recovered:
            state, viralload = 'R', 0.0
        elif i < (num_recovered + num_infected):
            state, viralload = 'I', (thresh2 + thresh3) / 2
        elif i < (num_recovered + num_infected + num_exposed):
            state, viralload = 'E', (thresh1 + thresh2) / 2
        else:
            state, viralload = 'S', 0.0

        index_of_age_group = int(np.argmax(cumulative_agents_per_group > i))
        age = pick_age_in_group(index_of_age_group)
        agents.append(Agent(state, viralload, age))

    # Bookkeeping structures
    state_counts = []
    # FIXED: include initial recovered count
    s0 = num_agents - (num_infected + num_exposed + num_recovered)
    state_counts.append([s0, num_exposed, num_infected, num_recovered, 0])

    state_dynamics_by_age = {age_group: [] for age_group in age_groups}
    viral_load_data = [[] for _ in range(num_agents)]  # per-agent time series
    viral_load_data_by_age = [[] for _ in age_groups]  # pooled by age
    viral_load_data_by_age_and_time = [[[] for _ in range(time_steps)] for _ in age_groups]
    max_viral_loads_by_age = [0.0] * len(age_groups)
    avg_viral_loads = []  # population mean (alive) over time

    # Age-structured contact matrix
    social_interaction_matrix = np.array([
        [2.5982, 0.8003, 0.3160, 0.7934, 0.3557, 0.1548, 0.0564],
        [0.6473, 4.1960, 0.6603, 0.5901, 0.4665, 0.1238, 0.0515],
        [0.1737, 1.7500, 11.1061, 0.9782, 0.7263, 0.0815, 0.0273],
        [0.5504, 0.5906, 1.2004, 1.8813, 0.9165, 0.1370, 0.0397],
        [0.3894, 0.7848, 1.3139, 1.1414, 1.3347, 0.2260, 0.0692],
        [0.3610, 0.3918, 0.3738, 0.5248, 0.5140, 0.7072, 0.1469],
        [0.1588, 0.3367, 0.3406, 0.2286, 0.3637, 0.3392, 0.3868]
    ])
    normalized_matrix = social_interaction_matrix / np.sum(social_interaction_matrix, axis=1, keepdims=True)
    row_sums = np.cumsum(normalized_matrix, axis=1)

    # Run dynamics
    for t in range(time_steps):
        # 1) Update agent internal states (VL growth/decay + transitions)
        for agent in agents:
            agent.update_state(deaths_by_ages)
            viral_load_data_by_age[agent.age_group_index].append(agent.viralload)

        # Track per-age maximum VL
        for agent in agents:
            gi = agent.age_group_index
            max_viral_loads_by_age[gi] = max(max_viral_loads_by_age[gi], agent.viralload)

        # 2) Contacts: stochastic partner by age, then S<->I interaction
        for _ in range(100):
            agent1 = random.choice(agents)
            g1 = agent1.age_group_index
            random_value = random.random()
            age_group_index2 = int(np.argmax(row_sums[g1] > random_value))

            # FIXED: guard empty target group
            group2_agents = [a for a in agents if a.age_group_index == age_group_index2]
            if not group2_agents:
                continue
            agent2 = random.choice(group2_agents)

            # Identify S and I roles
            if agent1.get_state() == 'S' and agent2.get_state() == 'I':
                susceptible_exposed_agent, infected_agent = agent1, agent2
            elif agent1.get_state() == 'I' and agent2.get_state() == 'S':
                susceptible_exposed_agent, infected_agent = agent2, agent1
            else:
                continue

            # Either probabilistic infection or "viral-load push"
            if infection_rate > 0.0:
                p = 1.0 - math.exp(-infection_rate * max(0.0, infected_agent.viralload))
                if random.random() < p:
                    susceptible_exposed_agent.viralload = max(
                        susceptible_exposed_agent.viralload, (thresh1 + thresh2) / 2
                    )
            else:
                susceptible_exposed_agent.viralload = clamp01(
                    susceptible_exposed_agent.viralload + infected_agent.viralload / 4.0
                )

        # 3) Bookkeeping (counts)
        s_count = sum(a.get_state() == 'S' for a in agents)
        e_count = sum(a.get_state() == 'E' for a in agents)
        i_count = sum(a.get_state() == 'I' for a in agents)
        r_count = sum(a.get_state() == 'R' for a in agents)
        d_count = sum(a.get_state() == 'D' for a in agents)
        state_counts.append([s_count, e_count, i_count, r_count, d_count])

        # Sanity: population conserved
        assert s_count + e_count + i_count + r_count + d_count == num_agents, "Population not conserved"

        # 4) Per-age state dynamics
        for ag in age_groups:
            gi = age_groups.index(ag)
            s_a = sum((a.age_group_index == gi) and (a.get_state() == 'S') for a in agents)
            e_a = sum((a.age_group_index == gi) and (a.get_state() == 'E') for a in agents)
            i_a = sum((a.age_group_index == gi) and (a.get_state() == 'I') for a in agents)
            r_a = sum((a.age_group_index == gi) and (a.get_state() == 'R') for a in agents)
            d_a = sum((a.age_group_index == gi) and (a.get_state() == 'D') for a in agents)
            state_dynamics_by_age[ag].append((s_a, e_a, i_a, r_a, d_a))

        # 5) Average VL among alive agents
        alive = [a for a in agents if a.get_state() != 'D']
        avg_viral_loads.append(np.mean([a.viralload for a in alive]) if alive else 0.0)

        # 6) Agent-time series
        for i, a in enumerate(agents):
            viral_load_data[i].append(a.viralload)
            viral_load_data_by_age_and_time[a.age_group_index][t].append(a.viralload)

    # Per-agent counters
    days_exposed  = [a.days_exposed  for a in agents]
    days_infected = [a.days_infected for a in agents]

    # Summary table per age (people & deaths)
    people_count = agents_per_age_group[:]  # final counts per age bin
    age_df = pd.DataFrame({'Age Group': age_groups, 'People': people_count, 'Deaths': deaths_by_ages})
    print(age_df)

    # Areas under VL curve per age (local only; not global/parallel unsafe)
    areas_this_run = [np.trapz(v) for v in viral_load_data_by_age]
    std_dev_max_viral_loads_by_age = float(np.std(max_viral_loads_by_age))
    for gi, ag in enumerate(age_groups):
        print(f"Maximum Viral Load for {ag}: {max_viral_loads_by_age[gi]}")
    print(f"Std Dev of Max Viral Loads (by age): {std_dev_max_viral_loads_by_age}")

    return (
        state_counts,                   # [time+1, 5]
        agents,                         # last snapshot of agents
        avg_viral_loads,                # [time]
        state_dynamics_by_age,          # dict: age -> [time, 5]
        # age/time VL summaries
        viral_load_data_by_age,         # pooled per age (flat list of VLs over time)
        viral_load_data,                # [N_agents, time]
        viral_load_data_by_age_and_time,# [age][time][VLs]
        days_exposed, days_infected,
        areas_this_run                  # per-age AUCs for this run
    )

# ------------------------------------
# Multiple simulations (serial & tidy)
# ------------------------------------
def run_simulations(num_simulations: int = 20):
    results = []
    for s in range(num_simulations):
        print(f"Running simulation {s+1}/{num_simulations}...")
        results.append(simulate(s))

    # Unpack stacks
    state_counts_stack = np.stack([np.array(r[0]) for r in results], axis=0)  # [sims, time+1, 5]
    avg_state_counts = state_counts_stack.mean(axis=0)

    overall_avg_loads = np.stack([np.array(r[2]) for r in results], axis=0).mean(axis=0)  # [time]

    # Build per-age per-time averages: for each age group, mean over simulations of (mean VL at each time)
    overall_avg_viral_loads_by_age = []
    for age_idx, ag in enumerate(age_groups):
        # gather per-sim per-time arrays for this age group
        per_sim_series = []
        for r in results:
            age_time_lists = r[6][age_idx]  # list over time; each entry is list of VLs for that age at that time
            # reduce: mean across agents in that age at each time
            series = [np.mean(ts) if len(ts) else 0.0 for ts in age_time_lists]
            per_sim_series.append(series)
        per_sim_series = np.array(per_sim_series)  # [sims, time]
        overall_avg_viral_loads_by_age.append(per_sim_series.mean(axis=0))
    overall_avg_viral_loads_by_age = np.array(overall_avg_viral_loads_by_age)  # [age, time]

    # Average state dynamics by age
    avg_state_dynamics_by_age = {}
    for ag in age_groups:
        stacks = np.stack([np.array(r[3][ag]) for r in results], axis=0)  # [sims, time, 5]
        avg_state_dynamics_by_age[ag] = stacks.mean(axis=0)

    # Aggregate per-age AUCs if you need them
    areas_stack = np.stack([np.array(r[9]) for r in results], axis=0)  # [sims, age]
    avg_area_by_age = areas_stack.mean(axis=0)

    # Expose some details for later printing/plots
    return dict(
        avg_state_counts=avg_state_counts,
        overall_avg_loads=overall_avg_loads,
        overall_avg_viral_loads_by_age=overall_avg_viral_loads_by_age,
        avg_state_dynamics_by_age=avg_state_dynamics_by_age,
        avg_area_by_age=avg_area_by_age,
        last_agents=results[-1][1],
        results=results
    )

# -------------
# Run + Save
# -------------
if __name__ == "__main__":
    start_time_script = time.time()
    num_simulations = 20  # start moderate; increase when satisfied
    bundle = run_simulations(num_simulations)
    total_time_script = time.time() - start_time_script
    print(f"Total time taken: {total_time_script:.2f} seconds")

    # SAVE some CSVs similar to your original
    ovr_dir = os.path.join(primary_directory, "Viral_Load_Data")
    os.makedirs(ovr_dir, exist_ok=True)

    # Overall average VL over time
    with open(os.path.join(ovr_dir, 'overall_avg_viral_load.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(bundle["overall_avg_loads"])

    # Per-age average VL over time
    for gi, ag in enumerate(age_groups):
        path = os.path.join(ovr_dir, f'avg_viral_load_time_{ag}.csv')
        np.savetxt(path, np.array(bundle["overall_avg_viral_loads_by_age"][gi]), delimiter=',', fmt='%0.5f')

    # -----------
    # Plotting
    # -----------
    plot_dir = os.path.join(primary_directory, "ABM_VL_Plotting")
    os.makedirs(plot_dir, exist_ok=True)

    avg_state_counts = bundle["avg_state_counts"]
    s_counts = avg_state_counts[:, 0]; e_counts = avg_state_counts[:, 1]
    i_counts = avg_state_counts[:, 2]; r_counts = avg_state_counts[:, 3]
    d_counts = avg_state_counts[:, 4]

    # SEIRD population
    plt.figure(figsize=(10, 8))
    plt.plot(s_counts, label='Susceptible')
    plt.plot(e_counts, label='Exposed')
    plt.plot(i_counts, label='Infected')
    plt.plot(r_counts, label='Recovered')
    plt.plot(d_counts, label='Deaths')
    plt.xlabel('Time steps'); plt.ylabel('Number of agents')
    plt.title('Agent-based SEIRD model (averaged over simulations)')
    plt.legend(); plt.grid(True)
    save_and_show(os.path.join(plot_dir, 'SEIRD_population_state_dynamics.png'))

    # Average viral load over time
    step_count = list(range(len(bundle["overall_avg_loads"])))
    plt.figure(figsize=(10, 8))
    plt.plot(step_count, bundle["overall_avg_loads"], label='Average Viral Load')
    plt.title('Average Viral Load Over Time (Averaged Across Simulations)')
    plt.xlabel('Time Steps'); plt.ylabel('Average Viral Load')
    plt.legend(); plt.grid(True)
    save_and_show(os.path.join(plot_dir, 'Average_Viral_Load_Over_Time.png'))

    # Per-age viral load
    ov_age = bundle["overall_avg_viral_loads_by_age"]  # [age, time]
    for gi, ag in enumerate(age_groups):
        plt.figure(figsize=(10, 8))
        plt.plot(ov_age[gi], label=f'Age Group {ag}')
        plt.xlabel('Time steps'); plt.ylabel('Average Viral Load')
        plt.title(f'Average Viral Load for Age Group {ag}')
        plt.legend(); plt.grid(True)
        save_and_show(os.path.join(plot_dir, f'Avg_VL_Age_{ag}.png'))

    # All ages on one chart
    plt.figure(figsize=(10, 8))
    for gi, ag in enumerate(age_groups):
        plt.plot(ov_age[gi], label=f'{ag}')
    plt.xlabel('Time steps'); plt.ylabel('Average Viral Load')
    plt.title('Average Viral Load by Age Group')
    plt.legend(); plt.grid(True)
    save_and_show(os.path.join(plot_dir, 'Average_Viral_Load_by_Age_Group.png'))

    # Print compact agent info from the last run (optional)
    print("\nAgent Information (last run):")
    print("{:<10} {:<15} {:<15} {:<5}".format("Agent ID", "Days Exposed", "Days Infected", "Age"))
    avg_inf = 0; not_inf = 0
    for i, agent in enumerate(bundle["last_agents"]):
        print("{:<10} {:<15} {:<15} {:<5}".format(i+1, agent.days_exposed, agent.days_infected, agent.age))
        avg_inf += agent.days_infected
        if agent.days_infected == 0:
            not_inf += 1
    denom = (num_agents - not_inf) if (num_agents - not_inf) > 0 else 1
    print("average days infected", avg_inf / denom)
