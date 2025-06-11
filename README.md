# Heuristics Tournament Leaderboard

This is a simple Flask web app to collect and display submissions for a heuristics tournament.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   python app.py
   ```

3. Visit [http://localhost:5000/leaderboard](http://localhost:5000/leaderboard) to view the leaderboard.

## Submitting Results

Send a POST request to `/submit` with a JSON payload like this:

```
{
  "team": "Team Alpha",
  "strategy": "best/1",
  "F": 0.6,
  "CR": 0.9,
  "fitness": 0.123456,
  "function": "rastrigin",
  "seed": 42,
  "hash": "..."
}
```

Example Python code for students:

```python
import requests
response = requests.post("http://localhost:5000/submit", json=submission)
print("✅ Submitted!" if response.status_code == 200 else "❌ Error:", response.text)
```

## Notes
- Submissions are stored in `submissions.json` in the project directory.
- The leaderboard is sorted by fitness (lower is better). 