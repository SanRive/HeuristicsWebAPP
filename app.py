from flask import Flask, request, jsonify, render_template
import json
import os
import time
import numpy as np
from collections import defaultdict

app = Flask(__name__)

# In-memory submission store (not persistent)
submissions_memory = []

ALLOWED_SEEDS = [42, 123, 999]

# -------------------------------
# DE algorithm implementation
# -------------------------------
def run_de(strategy, func_name, F, CR, seed=0, D=10, NP=30, time_limit=5):
    np.random.seed(seed)

    def rastrigin(x):
        A = 10
        return A * len(x) + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)

    def ackley(x):
        a, b, c = 20, 0.2, 2*np.pi
        d = len(x)
        return -a*np.exp(-b*np.sqrt(np.sum(x**2)/d)) - np.exp(np.sum(np.cos(c*x))/d) + a + np.exp(1)

    def griewank(x):
        return np.sum(x**2)/4000 - np.prod([np.cos(xi/np.sqrt(i+1)) for i, xi in enumerate(x)]) + 1

    func_map = {"rastrigin": rastrigin, "ackley": ackley, "griewank": griewank}
    fobj = func_map[func_name]

    if func_name == "rastrigin":
        lim_inf, lim_sup = -5.12, 5.12
    elif func_name == "ackley":
        lim_inf, lim_sup = -32.768, 32.768
    elif func_name == "griewank":
        lim_inf, lim_sup = -600, 600
    else:
        raise ValueError("Unknown function name")

    poblacion = np.random.uniform(lim_inf, lim_sup, (NP, D))
    fitness = np.array([fobj(ind) for ind in poblacion])
    best = poblacion[np.argmin(fitness)]
    start_val = np.min(fitness)

    start_time = time.time()
    while time.time() - start_time < time_limit:
        for i in range(NP):
            if time.time() - start_time >= time_limit:
                break
            idxs = [idx for idx in range(NP) if idx != i]
            a, b, c, d_, e = poblacion[np.random.choice(idxs, 5, replace=False)]
            x = poblacion[i]

            if strategy == "rand/1":
                mutant = a + F * (b - c)
            elif strategy == "best/1":
                mutant = best + F * (a - b)
            elif strategy == "current-to-best/1":
                mutant = x + F * (best - x) + F * (a - b)
            elif strategy == "rand/2":
                mutant = a + F * (b - c) + F * (d_ - e)
            else:
                raise ValueError("Unknown strategy")

            j_rand = np.random.randint(D)
            trial = np.array([mutant[j] if np.random.rand() < CR or j == j_rand else x[j] for j in range(D)])
            trial = np.clip(trial, lim_inf, lim_sup)
            f_trial = fobj(trial)

            if f_trial < fitness[i]:
                poblacion[i] = trial
                fitness[i] = f_trial
                if f_trial < fobj(best):
                    best = trial

    return start_val, np.min(fitness)

# -------------------------------
# Helpers
# -------------------------------
def load_submissions():
    return submissions_memory

def save_submissions(submissions):
    global submissions_memory
    submissions_memory = submissions

def get_range(val):
    step = 0.25
    lower = round(max(0.0, val - step / 2), 2)
    upper = round(min(1.0, val + step / 2), 2)
    return f"{lower} - {upper}"

# -------------------------------
# Routes
# -------------------------------
@app.route('/submit', methods=['POST'])
def submit():
    data = request.json

    required_fields = ["username", "strategy", "F", "CR", "function", "seed"]
    if not all(field in data for field in required_fields):
        return jsonify({"status": "error", "message": "Missing required fields."}), 400

    if data["seed"] not in ALLOWED_SEEDS:
        return jsonify({"status": "error", "message": f"Seed {data['seed']} not allowed. Use one of {ALLOWED_SEEDS}."}), 400

    try:
        start_time = time.time()
        start_fitness, best_fitness = run_de(
            strategy=data["strategy"],
            func_name=data["function"],
            F=data["F"],
            CR=data["CR"],
            seed=data["seed"]
        )
        runtime = round(time.time() - start_time, 3)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Execution failed: {e}"}), 500

    result = {
        "username": data["username"],
        "strategy": data["strategy"],
        "F": data["F"],
        "CR": data["CR"],
        "F_display": get_range(data["F"]),
        "CR_display": get_range(data["CR"]),
        "fitness": round(best_fitness, 6),
        "function": data["function"],
        "seed": data["seed"],
        "runtime": runtime,
        "start_fitness": round(start_fitness, 6)
    }

    submissions = load_submissions()
    existing = next((s for s in submissions if s['username'] == result['username'] and s['function'] == result['function']), None)
    if existing:
        if result["fitness"] < existing.get("fitness", float("inf")):
            submissions = [s for s in submissions if not (s['username'] == result['username'] and s['function'] == result['function'])]
            submissions.append(result)
    else:
        submissions.append(result)

    save_submissions(submissions)
    return jsonify({"status": "ok", "fitness": result["fitness"], "runtime": result["runtime"]})

@app.route('/leaderboard')
def leaderboard():
    grouped = defaultdict(list)
    for s in load_submissions():
        grouped[s["function"]].append(s)

    for func in grouped:
        grouped[func].sort(key=lambda x: x.get("fitness", float("inf")))

    return render_template("leaderboard.html", grouped=grouped, functions=sorted(grouped.keys()))

@app.route('/leaderboard_admin')
def admin_leaderboard():
    grouped = defaultdict(list)
    for s in load_submissions():
        grouped[s["function"].lower()].append(s)

    for func in grouped:
        grouped[func].sort(key=lambda x: x.get("fitness", float("inf")))

    return render_template("admin_leaderboard.html", grouped=grouped, functions=sorted(grouped.keys()))

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
