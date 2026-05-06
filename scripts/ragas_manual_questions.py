import csv, json, random
from pathlib import Path

random_seed = 42
paper_limit = 25

questions = [
    ("main_findings", "What are the main research question and main findings of this paper?"),
    ("data_sources", "What data sources or datasets does this paper use?"),
    ("methods", "What methods or models does this paper use?"),
]

with open("sampled_papers_full.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

rows = [r for r in rows if r.get("paper_id")]
random.Random(random_seed).shuffle(rows)
rows = rows[:paper_limit]

out = Path("eval_outputs/ragas_question_sets/sampled_question_only.jsonl")
with out.open("w", encoding="utf-8") as f:
    for row in rows:
        for qid, question in questions:
            f.write(json.dumps({
                "schema_version": "1",
                "case_id": f"{row['paper_id']}__{qid}",
                "paper_id": row["paper_id"],
                "user_input": question,
                "reference": "NO_REFERENCE_USED",
                "tags": ["sampled_papers_full", "no_reference", qid],
                "metadata": {"title": row.get("title"), "filepath": row.get("filepath")}
            }) + "\n")

print(out)
print(f"papers={len(rows)} cases={len(rows) * len(questions)}")
PY
