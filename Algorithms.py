import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random

gp.setParam('OutputFlag', 0)

def Util_many(demands, supply_list, storage, E, C, graph):
    num_agents, num_time_steps = demands.shape
    num_sources = len(supply_list)
    
    model = gp.Model("Utilitarian")
    w = model.addVars(num_agents, num_time_steps, num_sources, name="w", lb=0, ub=GRB.INFINITY)
    alpha = model.addVars(num_agents, name="alpha", lb=0, ub=1)
    
    model.setObjective(alpha.sum(), GRB.MAXIMIZE)
    for i in range(num_agents):
        for t in range(num_time_steps):
            model.addConstr(gp.quicksum(w[i, t, source_idx] for source_idx in range(num_sources)) >= alpha[i] * demands[i, t], name=f"tightness_{i}_{t}")
            for source_idx in range(num_sources):
                model.addConstr(w[i, t, source_idx] <= graph[source_idx, i] * GRB.INFINITY, name=f"graph_constraint_{i}_{t}_{source_idx}")
    
    for source_idx, supply in enumerate(supply_list):
        if storage == 0:
            for t in range(num_time_steps):
                model.addConstr(gp.quicksum(w[i, t, source_idx] for i in range(num_agents)) <= supply[t], f"supply_constraint_{source_idx}_{t}")
        elif storage == float('inf'):
            infint_cap1(model, w, supply, num_time_steps, E, source_idx)  
        else:
            finit_cap1(model, w, supply, num_time_steps, E, C, source_idx)  
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        obj_value = model.objVal
        alpha_values = {i: alpha[i].X for i in range(num_agents)}
        allocation = {(i, t, source_idx): w[i, t, source_idx].X for i in range(num_agents) for t in range(num_time_steps) for source_idx in range(num_sources)}
        return obj_value, alpha_values, allocation
    else:
        print("Optimization did not converge.")
        return None, None, None

def Util_one(demands, supply, storage, E, C):
    num_agents, num_time_steps = demands.shape
    model = gp.Model("Utilitarian")
    w = model.addVars(num_agents, num_time_steps, name="w", lb=0, ub=GRB.INFINITY)
    alpha = model.addVars(num_agents, name="alpha", lb=0, ub=1)
    model.setObjective(alpha.sum(), GRB.MAXIMIZE)
    for i in range(num_agents):
        for t in range(num_time_steps):
            model.addConstr(w[i, t] >= alpha[i] * demands[i, t], name=f"tightness_{i}_{t}")
    if storage == 0:
        for t in range(num_time_steps):
            model.addConstr(gp.quicksum(w[i, t] for i in range(num_agents)) <= supply[t], f"supply_constraint_{t}")
    elif storage == float('inf'):
        infint_cap(model, w, supply, num_time_steps, E)
    else:
        finit_cap(model, w, supply, num_time_steps, E, C)
    model.optimize()
    if model.status == GRB.OPTIMAL:
        obj_value = model.objVal
        alpha_values = {i: alpha[i].X for i in range(num_agents)}
        allocation = {i: {t: w[i, t].X for t in range(num_time_steps)} for i in range(num_agents)}
        return obj_value, alpha_values, allocation
    else:
        print("Optimization did not converge.")
        return None, None, None
    
def infint_cap(model, w, supply, num_time_steps, E):
    X = model.addVars(num_time_steps, name="X", lb=0, ub=GRB.INFINITY)
    model.addConstr(w[0,0] <= supply[0], name="supply_constraint_0")
    for t in range(1, num_time_steps):
        model.addConstr(w[0,t] <= supply[t] + X[t], name=f"supply_constraint_{t}")
        model.addConstr(X[t] <= (X[t - 1] + supply[t - 1] - w[0,t - 1]) * E[t], name=f"storage_constraint_{t}")

def finit_cap(model, w, supply, num_time_steps, E, C):
    X = model.addVars(num_time_steps, name="X", lb=0, ub=C)
    model.addConstr(w[0,0] <= supply[0], name="supply_constraint_0")
    for t in range(1, num_time_steps):
        model.addConstr(w[0,t] <= supply[t] + X[t], name=f"supply_constraint_{t}")
        model.addConstr(X[t] <= (X[t - 1] + supply[t - 1] - w[0,t - 1]) * E[t], name=f"storage_constraint_{t}")
    for t in range(num_time_steps):
        model.addConstr(X[t] <= C, name=f"upper_limit_constraint_{t}")

def infint_cap1(model, w, supply, num_time_steps, E, source_idx):
    X = model.addVars(num_time_steps, name=f"X_{source_idx}", lb=0, ub=GRB.INFINITY)
    for t in range(num_time_steps):
        for i, j, k in w.keys():
            if j == t and k == source_idx:
                model.addConstr(w[i, t, source_idx] <= supply[t] + X[t], name=f"supply_constraint_{source_idx}_{t}")
                if t > 0:
                    model.addConstr(X[t] <= (X[t - 1] + supply[t - 1] - w[i, t - 1, source_idx]) * E[t], name=f"storage_constraint_{source_idx}_{t}")

def finit_cap1(model, w, supply, num_time_steps, E, C, source_idx):
    X = model.addVars(num_time_steps, name=f"X_{source_idx}", lb=0, ub=C)
    for t in range(num_time_steps):
        for i, j, k in w.keys():
            if j == t and k == source_idx:
                model.addConstr(w[i, t, source_idx] <= supply[t] + X[t], name=f"supply_constraint_{source_idx}_{t}")
                if t > 0:
                    model.addConstr(X[t] <= (X[t - 1] + supply[t - 1] - w[i, t - 1, source_idx]) * E[t], name=f"storage_constraint_{source_idx}_{t}")
                model.addConstr(X[t] <= C, name=f"upper_limit_constraint_{source_idx}_{t}")

def greedy_allocation(demands, supplies, storage, E, C):
    num_agents, num_time_steps = demands.shape
    num_sources = len(supplies)
    total_allocation = np.zeros((num_agents, num_sources, num_time_steps))
    alphas = np.ones(num_agents)
    graph = np.zeros((num_sources, num_agents), dtype=int)
    True_demands = demands.copy()
    
    for source_idx, supply in enumerate(supplies):
        for agent_idx in range(num_agents):
            if np.random.rand() <= 0.5: 
                graph[source_idx, agent_idx] = 1
        subset_indices = np.where(graph[source_idx] == 1)[0] 
        
        if len(subset_indices) > 0:
            subset_demands = demands[subset_indices, :].copy()
            obj_value, alpha_values, allocation = Util_one(subset_demands, supply, storage, E, C)
            
            if allocation is not None:
                for subset_agent_idx, original_agent_idx in enumerate(subset_indices):
                    for t in range(num_time_steps):
                        allocation_value = allocation[subset_agent_idx][t]
                        total_allocation[original_agent_idx, source_idx, t] = allocation_value
                        demands[original_agent_idx, t] = max(0, demands[original_agent_idx, t] - allocation_value)
    
    for agent_idx in range(num_agents):
        min_alpha_for_agent = min(
            total_allocation[agent_idx, :, t].sum() / True_demands[agent_idx, t] if True_demands[agent_idx, t] > 0 else 1
            for t in range(num_time_steps)
        )
        alphas[agent_idx] = min(1, min_alpha_for_agent)
    
    return np.sum(alphas), graph

num_time_steps = 1

results = {}

for storage in [0]:
    for num_agents in range(5, 30, 5):
        for num_agents1 in range(5, 30, 5):
            approx = []
            supplies = []
            for _ in range(10):
                E = np.linspace(0.9, 1.0, 12)
                C = storage * num_agents
                demands = np.random.uniform(50, 100, size=(num_agents, num_time_steps))
                for _ in range(num_agents1):
                    supply = np.random.uniform(5, 20, size=num_time_steps)
                    supplies.append(supply)
                sum_alphas_greedy, graph = greedy_allocation(demands, supplies, storage, E, C)
                obj_util_optimal, alpha_values_optimal, allocation_optimal = Util_many(demands, supplies, storage, E, C, graph)
                if obj_util_optimal is not None:
                    approx.append((100-100*(obj_util_optimal - sum_alphas_greedy) / obj_util_optimal))
            results[(num_agents, num_agents1)] = approx
averages = {key: sum(values) / len(values) for key, values in results.items()}
