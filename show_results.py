import csv

path = r"C:\Users\SHARVIL MORE\Downloads\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\submission.csv"

with open(path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print("TOP 15 RANKED CANDIDATES")
print("-" * 100)
for r in rows[:15]:
    reasoning_short = r["reasoning"][:85]
    print(f"#{r['rank']:>3}  {r['candidate_id']}  score={r['score']}  {reasoning_short}")

print()
print(f"Total rows: {len(rows)}")
print(f"Score range: {rows[-1]['score']} - {rows[0]['score']}")
