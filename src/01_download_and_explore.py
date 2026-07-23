"""
Step 1: Download CUAD dataset and explore it
=============================================
Downloads CUAD data.zip from GitHub, extracts, parses, and explores.
"""

import os
import sys
import json
import zipfile
import requests
from collections import Counter

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Config ──────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 70)
print("  STEP 1: Download & Explore CUAD Dataset")
print("=" * 70)

# ── 1. Download data.zip from GitHub ────────────────────
ZIP_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
zip_path = os.path.join(DATA_DIR, "data.zip")
extract_dir = os.path.join(DATA_DIR, "cuad_raw")

if not os.path.exists(extract_dir):
    if not os.path.exists(zip_path):
        print(f"\n[DOWNLOAD] Fetching data.zip from GitHub...")
        print(f"   URL: {ZIP_URL}")
        resp = requests.get(ZIP_URL, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f:
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r   Downloading: {downloaded/(1024*1024):.1f} MB / {total/(1024*1024):.1f} MB ({pct:.0f}%)", end="", flush=True)
        print(f"\n   [OK] Saved to {zip_path}")
    else:
        print(f"\n[CACHED] data.zip already exists at {zip_path}")

    print(f"\n[EXTRACT] Extracting data.zip...")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    print(f"   [OK] Extracted to {extract_dir}")
    
    # List extracted files
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath) / (1024*1024)
            print(f"   - {os.path.relpath(fpath, extract_dir)} ({size:.1f} MB)")
else:
    print(f"\n[CACHED] Dataset already extracted at {extract_dir}")

# ── 2. Find and load the JSON files ────────────────────
print("\n[LOAD] Looking for JSON files...")

train_data = None
test_data = None

for root, dirs, files in os.walk(extract_dir):
    for f in files:
        fpath = os.path.join(root, f)
        if f.endswith('.json') and 'train' in f.lower():
            print(f"   Found train file: {fpath}")
            with open(fpath, 'r', encoding='utf-8') as fp:
                train_data = json.load(fp)
        elif f.endswith('.json') and 'test' in f.lower():
            print(f"   Found test file: {fpath}")
            with open(fpath, 'r', encoding='utf-8') as fp:
                test_data = json.load(fp)

# If no separate files, look for any JSON
if train_data is None:
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.endswith('.json'):
                fpath = os.path.join(root, f)
                print(f"   Found JSON: {fpath}")
                with open(fpath, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                if isinstance(data, dict) and 'data' in data:
                    # SQuAD format — split 85/15
                    all_articles = data['data']
                    split_idx = int(len(all_articles) * 0.85)
                    train_data = {'data': all_articles[:split_idx]}
                    test_data = {'data': all_articles[split_idx:]}
                    print(f"   Split: {len(all_articles)} articles -> {split_idx} train / {len(all_articles)-split_idx} test")

if train_data is None:
    print("\n[ERROR] Could not find JSON data files!")
    print("   Please check the extracted contents above.")
    sys.exit(1)

# ── 3. Parse SQuAD format ──────────────────────────────
def parse_squad(raw):
    examples = []
    for article in raw.get('data', []):
        title = article.get('title', 'Unknown')
        for para in article.get('paragraphs', []):
            context = para.get('context', '')
            for qa in para.get('qas', []):
                answers = qa.get('answers', [])
                examples.append({
                    'id': qa.get('id', ''),
                    'title': title,
                    'context': context,
                    'question': qa.get('question', ''),
                    'answers_text': [a['text'] for a in answers] if answers else [],
                    'answers_start': [a['answer_start'] for a in answers] if answers else [],
                })
    return examples

print("\n[PARSE] Converting to flat examples...")
train_ex = parse_squad(train_data)
test_ex = parse_squad(test_data) if test_data else []
print(f"   Train: {len(train_ex):,} examples")
print(f"   Test:  {len(test_ex):,} examples")
print(f"   Total: {len(train_ex) + len(test_ex):,} examples")

# ── 4. Dataset Overview ───────────────────────────────
print("\n" + "=" * 70)
print("  DATASET OVERVIEW")
print("=" * 70)

ex = train_ex[0]
print(f"\n   Fields: {list(ex.keys())}")
print(f"\n   First example:")
print(f"   ID:       {ex['id'][:80]}")
print(f"   Title:    {ex['title'][:80]}")
print(f"   Question: {ex['question'][:100]}")
print(f"   Context:  {ex['context'][:150]}...")
print(f"   Answers:  {ex['answers_text'][:2]}")

# ── 5. Clause Type Analysis ───────────────────────────
print("\n" + "=" * 70)
print("  CLAUSE TYPE ANALYSIS")
print("=" * 70)

clause_types = []
for e in train_ex:
    q = e['question']
    if '"' in q:
        parts = q.split('"')
        if len(parts) >= 2:
            clause_types.append(parts[1])

clause_counter = Counter(clause_types)
print(f"\n   Found {len(clause_counter)} unique clause types:\n")

for i, (clause, count) in enumerate(clause_counter.most_common(), 1):
    print(f"   {i:2d}. {clause:<50s} | {count:5d} examples")

# ── 6. Answer Analysis ────────────────────────────────
print("\n" + "=" * 70)
print("  ANSWER ANALYSIS")
print("=" * 70)

with_ans = 0
without_ans = 0
ans_lens = []

for e in train_ex:
    if e['answers_text'] and e['answers_text'][0]:
        with_ans += 1
        ans_lens.append(len(e['answers_text'][0]))
    else:
        without_ans += 1

print(f"\n   WITH answers:    {with_ans:,}")
print(f"   WITHOUT answers: {without_ans:,}")
if ans_lens:
    print(f"\n   Answer lengths:")
    print(f"   Min: {min(ans_lens):,} | Max: {max(ans_lens):,} | Mean: {sum(ans_lens)//len(ans_lens):,} | Median: {sorted(ans_lens)[len(ans_lens)//2]:,}")

# ── 7. Contract Types ─────────────────────────────────
print("\n" + "=" * 70)
print("  CONTRACTS")
print("=" * 70)

titles = set(e['title'] for e in train_ex)
print(f"\n   Unique contracts: {len(titles)}")
for i, t in enumerate(sorted(titles)[:10], 1):
    print(f"   {i}. {t[:80]}")

# ── 8. Sample Examples ────────────────────────────────
print("\n" + "=" * 70)
print("  SAMPLE EXAMPLES (3 with answers)")
print("=" * 70)

shown = 0
for e in train_ex:
    if e['answers_text'] and e['answers_text'][0] and shown < 3:
        shown += 1
        clause = e['question'].split('"')[1] if '"' in e['question'] else 'Unknown'
        print(f"\n--- Example {shown} ---")
        print(f"  Clause: {clause}")
        print(f"  Contract: {e['title'][:60]}")
        print(f"  Context: {e['context'][:200]}...")
        print(f"  Answer: {e['answers_text'][0][:200]}")

# ── 9. Save Summary ───────────────────────────────────
summary = {
    "dataset": "CUAD",
    "train_examples": len(train_ex),
    "test_examples": len(test_ex),
    "unique_contracts": len(titles),
    "clause_types": len(clause_counter),
    "clause_type_list": list(clause_counter.keys()),
    "with_answers": with_ans,
    "without_answers": without_ans,
    "distribution": dict(clause_counter.most_common()),
}

summary_path = os.path.join(DATA_DIR, "exploration_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\n\n[OK] Summary saved to: {summary_path}")
print("\n" + "=" * 70)
print("  DONE! Next: Run 02_format_dataset.py")
print("=" * 70)
