from flask import Flask, request, jsonify, render_template
import json
import os
import time
import numpy as np

app = Flask(__name__)

# In-memory submission store (not persistent)
submissions_memory = []

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

    lim_inf, lim_sup = -5.12, 5.12
    poblacion = np.random.uniform(lim_inf, lim_sup, (NP, D))
    fitness = np.array([fobj(ind) for ind in poblacion])
    best = poblacion[np.argmin(fitness)]

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

    return np.min(fitness)

# -------------------------------
# Helpers
# -------------------------------
def load_submissions():
    return submissions_memory

def save_submissions(submissions):
    global submissions_memory
    submissions_memory = submissions

# -------------------------------
# Routes
# -------------------------------
@app.route('/submit', methods=['POST'])
def submit():
    data = request.json

    required_fields = ["team", "strategy", "F", "CR", "function", "seed"]
    if not all(field in data for field in required_fields):
        return jsonify({"status": "error", "message": "Missing required fields."}), 400

    try:
        start_time = time.time()
        fitness = run_de(
            strategy=data["strategy"],
            func_name=data["function"],
            F=data["F"],
            CR=data["CR"],
            seed=data["seed"],
            time_limit=5
        )
        runtime = round(time.time() - start_time, 3)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Execution failed: {e}"}), 500

    result = {
        "team": data["team"],
        "strategy": data["strategy"],
        "F": data["F"],
        "CR": data["CR"],
        "fitness": round(fitness, 6),
        "function": data["function"],
        "seed": data["seed"],
        "runtime": runtime
    }

    submissions = load_submissions()
    existing = next((s for s in submissions if s['team'] == result['team']), None)
    if existing:
        if result["fitness"] < existing.get("fitness", float("inf")):
            submissions = [s for s in submissions if s['team'] != result['team']]
            submissions.append(result)
    else:
        submissions.append(result)

    save_submissions(submissions)
    return jsonify({"status": "ok", "fitness": result["fitness"], "runtime": result["runtime"]})

@app.route('/leaderboard')
def leaderboard():
    submissions = load_submissions()
    submissions.sort(key=lambda x: x.get('fitness', float('inf')))
    return render_template("leaderboard.html", submissions=submissions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
