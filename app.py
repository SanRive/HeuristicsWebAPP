from flask import Flask, request, jsonify, render_template
import json
import os
import time
from core_de import run_de

app = Flask(__name__)

# Constants
SUBMISSIONS_FILE = "submissions.json"

# -------------------------------
# Helpers
# -------------------------------
def load_submissions():
    if os.path.exists(SUBMISSIONS_FILE):
        try:
            with open(SUBMISSIONS_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Warning: submissions.json is invalid, resetting.")
            return []
    return []

def save_submissions(submissions):
    with open(SUBMISSIONS_FILE, "w") as f:
        json.dump(submissions, f, indent=2)

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
            time_limit=10
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
    app.run(debug=True)
