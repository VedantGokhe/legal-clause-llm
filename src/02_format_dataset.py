"""
Step 2: Format CUAD Dataset for Instruction Fine-Tuning
========================================================
Converts the CUAD SQuAD-format data into instruction-tuning format
that Qwen2.5-3B can learn from.

INPUT:  SQuAD format → context + question + answer span
OUTPUT: Chat format → system + user + assistant (with JSON + plain English)
"""

import os
import sys
import json
import random
from collections import defaultdict

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Config ──────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "cuad_raw")
OUTPUT_DIR = os.path.join(DATA_DIR, "formatted")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)

# ── Risk Level Mapping ─────────────────────────────────
# Map each of the 41 clause types to a risk level
# (based on how critical they are in legal review)
RISK_MAP = {
    # HIGH RISK - clauses that can cause significant legal/financial harm
    "Non-Compete": "HIGH",
    "Exclusivity": "HIGH",
    "Uncapped Liability": "HIGH",
    "Cap On Liability": "HIGH",
    "Liquidated Damages": "HIGH",
    "Ip Ownership Assignment": "HIGH",
    "Anti-Assignment": "HIGH",
    "Change Of Control": "HIGH",
    "Termination For Convenience": "HIGH",
    "Covenant Not To Sue": "HIGH",
    "Non-Disparagement": "HIGH",
    "No-Solicit Of Employees": "HIGH",
    "No-Solicit Of Customers": "HIGH",
    
    # MEDIUM RISK - important but negotiable
    "License Grant": "MEDIUM",
    "Non-Transferable License": "MEDIUM",
    "Irrevocable Or Perpetual License": "MEDIUM",
    "Affiliate License-Licensor": "MEDIUM",
    "Affiliate License-Licensee": "MEDIUM",
    "Unlimited/All-You-Can-Eat-License": "MEDIUM",
    "Revenue/Profit Sharing": "MEDIUM",
    "Price Restrictions": "MEDIUM",
    "Minimum Commitment": "MEDIUM",
    "Volume Restriction": "MEDIUM",
    "Joint Ip Ownership": "MEDIUM",
    "Source Code Escrow": "MEDIUM",
    "Post-Termination Services": "MEDIUM",
    "Insurance": "MEDIUM",
    "Warranty Duration": "MEDIUM",
    "Audit Rights": "MEDIUM",
    "Rofr/Rofo/Rofn": "MEDIUM",
    "Most Favored Nation": "MEDIUM",
    "Competitive Restriction Exception": "MEDIUM",
    "Third Party Beneficiary": "MEDIUM",
    
    # LOW RISK - standard/administrative clauses
    "Document Name": "LOW",
    "Parties": "LOW",
    "Agreement Date": "LOW",
    "Effective Date": "LOW",
    "Expiration Date": "LOW",
    "Renewal Term": "LOW",
    "Notice Period To Terminate Renewal": "LOW",
    "Governing Law": "LOW",
}

# ── Plain English Templates ────────────────────────────
# Templates that explain what each clause type means in simple words
CLAUSE_DESCRIPTIONS = {
    "Document Name": "This identifies the name/title of the contract document.",
    "Parties": "This identifies who is involved in the agreement - the companies or people signing the contract.",
    "Agreement Date": "This specifies when the contract was signed or agreed upon.",
    "Effective Date": "This specifies when the contract terms actually start taking effect.",
    "Expiration Date": "This specifies when the contract ends or expires.",
    "Renewal Term": "This describes how and when the contract can be renewed or extended.",
    "Notice Period To Terminate Renewal": "This specifies how much advance notice is needed to stop the contract from automatically renewing.",
    "Governing Law": "This specifies which state or country's laws will be used to interpret the contract if there's a dispute.",
    "Most Favored Nation": "This guarantees that one party gets terms at least as good as what the other party offers to anyone else.",
    "Non-Compete": "This restricts one party from competing with the other party, usually for a specific time period and geographic area. This can significantly limit future business activities.",
    "Exclusivity": "This gives one party exclusive rights, meaning the other party cannot work with competitors for similar services or products.",
    "No-Solicit Of Customers": "This prevents one party from approaching or taking away the other party's customers.",
    "Competitive Restriction Exception": "This defines specific exceptions or carve-outs to the competitive restrictions in the contract.",
    "No-Solicit Of Employees": "This prevents one party from hiring or recruiting the other party's employees.",
    "Non-Disparagement": "This prevents the parties from making negative or harmful public statements about each other.",
    "Termination For Convenience": "This allows one or both parties to end the contract at any time without needing a specific reason. This provides flexibility but reduces commitment certainty.",
    "Rofr/Rofo/Rofn": "Right of First Refusal/Offer/Negotiation - this gives one party the first opportunity to match or negotiate a deal before the other party can go to someone else.",
    "Change Of Control": "This specifies what happens to the contract if one of the companies is acquired, merged, or changes ownership.",
    "Anti-Assignment": "This restricts one or both parties from transferring their rights or obligations under this contract to someone else.",
    "Revenue/Profit Sharing": "This describes how revenues or profits will be split between the parties.",
    "Price Restrictions": "This sets rules about pricing, including price caps, floors, or adjustment mechanisms.",
    "Minimum Commitment": "This requires one party to purchase or deliver a minimum amount of products, services, or revenue.",
    "Volume Restriction": "This limits the maximum volume or quantity that can be purchased or sold.",
    "Ip Ownership Assignment": "This transfers intellectual property (patents, copyrights, trade secrets) from one party to another. This is often irreversible and highly impactful.",
    "Joint Ip Ownership": "This establishes that intellectual property created during the agreement is owned jointly by both parties.",
    "License Grant": "This grants one party permission to use the other party's intellectual property, technology, or products under specific conditions.",
    "Non-Transferable License": "This specifies that the license cannot be given, sold, or transferred to any third party.",
    "Affiliate License-Licensor": "This allows the licensor's affiliated companies to also benefit from or participate in the license.",
    "Affiliate License-Licensee": "This allows the licensee's affiliated companies to also use the licensed material.",
    "Unlimited/All-You-Can-Eat-License": "This grants unlimited usage rights without per-unit or volume-based restrictions.",
    "Irrevocable Or Perpetual License": "This grants a license that cannot be taken back (irrevocable) or lasts forever (perpetual), even after the contract ends.",
    "Source Code Escrow": "This requires software source code to be held by a neutral third party, to be released under certain conditions (like if the developer goes bankrupt).",
    "Post-Termination Services": "This requires certain services or support to continue even after the contract ends.",
    "Audit Rights": "This gives one party the right to inspect and verify the other party's records, books, or compliance with the contract.",
    "Uncapped Liability": "This means there is NO maximum limit on how much one party could owe in damages. This is a significant financial risk.",
    "Cap On Liability": "This sets a maximum limit on how much one party can be held liable for in damages.",
    "Liquidated Damages": "This specifies a predetermined amount of money that must be paid if one party breaches the contract.",
    "Warranty Duration": "This specifies how long the warranties and guarantees in the contract remain valid.",
    "Insurance": "This requires one or both parties to maintain specific types and amounts of insurance coverage.",
    "Covenant Not To Sue": "This is a promise by one party not to take legal action against the other party for specific claims.",
    "Third Party Beneficiary": "This identifies parties outside the contract who may have rights or benefits under the agreement.",
}

# ── System Prompt ──────────────────────────────────────
SYSTEM_PROMPT = """You are a legal contract clause analyzer. When given a contract clause, you must:

1. Identify the clause type from these categories: Document Name, Parties, Agreement Date, Effective Date, Expiration Date, Renewal Term, Notice Period To Terminate Renewal, Governing Law, Most Favored Nation, Non-Compete, Exclusivity, No-Solicit Of Customers, Competitive Restriction Exception, No-Solicit Of Employees, Non-Disparagement, Termination For Convenience, Rofr/Rofo/Rofn, Change Of Control, Anti-Assignment, Revenue/Profit Sharing, Price Restrictions, Minimum Commitment, Volume Restriction, Ip Ownership Assignment, Joint Ip Ownership, License Grant, Non-Transferable License, Affiliate License-Licensor, Affiliate License-Licensee, Unlimited/All-You-Can-Eat-License, Irrevocable Or Perpetual License, Source Code Escrow, Post-Termination Services, Audit Rights, Uncapped Liability, Cap On Liability, Liquidated Damages, Warranty Duration, Insurance, Covenant Not To Sue, Third Party Beneficiary.

2. Extract the key clause text and important terms.

3. Assess the risk level (HIGH, MEDIUM, or LOW).

4. Provide a plain English explanation that a non-lawyer can understand.

Respond in JSON format."""


print("=" * 70)
print("  STEP 2: Format CUAD Dataset for Instruction Fine-Tuning")
print("=" * 70)

# ── 1. Load Raw Data ───────────────────────────────────
print("\n[LOAD] Loading raw CUAD data...")

train_path = os.path.join(RAW_DIR, "train_separate_questions.json")
test_path = os.path.join(RAW_DIR, "test.json")

with open(train_path, 'r', encoding='utf-8') as f:
    train_raw = json.load(f)
with open(test_path, 'r', encoding='utf-8') as f:
    test_raw = json.load(f)

print(f"   Train articles: {len(train_raw['data'])}")
print(f"   Test articles:  {len(test_raw['data'])}")

# ── 2. Parse into flat examples ────────────────────────
def parse_squad(raw):
    examples = []
    for article in raw.get('data', []):
        title = article.get('title', 'Unknown')
        for para in article.get('paragraphs', []):
            context = para.get('context', '')
            for qa in para.get('qas', []):
                answers = qa.get('answers', [])
                # Extract clause type from the question
                question = qa.get('question', '')
                clause_type = None
                if '"' in question:
                    parts = question.split('"')
                    if len(parts) >= 2:
                        clause_type = parts[1]
                
                answer_texts = [a['text'] for a in answers] if answers else []
                has_answer = bool(answer_texts and answer_texts[0].strip())
                
                examples.append({
                    'id': qa.get('id', ''),
                    'title': title,
                    'context': context,
                    'clause_type': clause_type,
                    'answer_text': answer_texts[0].strip() if has_answer else None,
                    'has_answer': has_answer,
                })
    return examples

print("\n[PARSE] Parsing into flat examples...")
train_examples = parse_squad(train_raw)
test_examples = parse_squad(test_raw)
print(f"   Train: {len(train_examples):,} total examples")
print(f"   Test:  {len(test_examples):,} total examples")

# ── 3. Filter & Prepare ───────────────────────────────
# We want BOTH positive (has clause) and negative (no clause) examples
# Positive: model learns what each clause looks like
# Negative: model learns to say "this clause is not present"

positive_train = [e for e in train_examples if e['has_answer'] and e['clause_type']]
negative_train = [e for e in train_examples if not e['has_answer'] and e['clause_type']]

print(f"\n[FILTER] Filtered examples:")
print(f"   Positive (WITH clause):    {len(positive_train):,}")
print(f"   Negative (NO clause):      {len(negative_train):,}")

# Sample negatives to balance (keep ratio ~2:1 positive:negative)
max_negatives = len(positive_train) // 2
random.shuffle(negative_train)
negative_sample = negative_train[:max_negatives]
print(f"   Negative sample (2:1):     {len(negative_sample):,}")

# ── 4. Extract Key Terms ──────────────────────────────
def extract_key_terms(answer_text, clause_type):
    """Extract important terms from the clause text."""
    if not answer_text:
        return []
    
    terms = []
    text = answer_text
    
    # Look for quoted terms
    import re
    quoted = re.findall(r'"([^"]+)"', text)
    terms.extend(quoted[:3])  # Max 3 quoted terms
    
    # Look for monetary amounts
    money = re.findall(r'\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|thousand))?', text)
    terms.extend(money[:2])
    
    # Look for percentages
    pcts = re.findall(r'\d+(?:\.\d+)?%', text)
    terms.extend(pcts[:2])
    
    # Look for time periods
    times = re.findall(r'\d+\s*(?:year|month|day|week|hour)s?', text, re.IGNORECASE)
    terms.extend(times[:2])
    
    # Look for dates
    dates = re.findall(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}', text)
    terms.extend(dates[:2])
    
    # Deduplicate
    seen = set()
    unique_terms = []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique_terms.append(t)
    
    return unique_terms[:5]  # Max 5 terms

# ── 5. Create Instruction-Tuning Examples ──────────────
def create_positive_example(example):
    """Create a training example where the clause IS present."""
    clause_type = example['clause_type']
    context = example['context']
    answer = example['answer_text']
    
    # Truncate context if too long (keep it under 1500 chars for training efficiency)
    if len(context) > 1500:
        # Try to keep the answer within the truncated context
        answer_pos = context.find(answer[:50]) if answer else -1
        if answer_pos > 0:
            start = max(0, answer_pos - 500)
            end = min(len(context), answer_pos + len(answer) + 500)
            context = "..." + context[start:end] + "..."
        else:
            context = context[:1500] + "..."
    
    # Get metadata
    risk = RISK_MAP.get(clause_type, "MEDIUM")
    description = CLAUSE_DESCRIPTIONS.get(clause_type, f"This is a {clause_type} clause.")
    key_terms = extract_key_terms(answer, clause_type)
    
    # Build the user message
    user_msg = f"Analyze the following contract clause and identify any relevant legal provisions:\n\n{context}"
    
    # Build the assistant response (structured JSON)
    response = {
        "clause_type": clause_type,
        "found": True,
        "relevant_text": answer[:500] if answer else "",  # Cap at 500 chars
        "key_terms": key_terms if key_terms else [clause_type.lower()],
        "risk_level": risk,
        "plain_english": f"{description} In this contract: {answer[:200] if answer else 'N/A'}"
    }
    
    return {
        "system": SYSTEM_PROMPT,
        "user": user_msg,
        "assistant": json.dumps(response, indent=2)
    }


def create_negative_example(example):
    """Create a training example where the clause is NOT present."""
    clause_type = example['clause_type']
    context = example['context']
    
    # Truncate context
    if len(context) > 1500:
        context = context[:1500] + "..."
    
    user_msg = f"Analyze the following contract clause and identify if it contains a \"{clause_type}\" provision:\n\n{context}"
    
    response = {
        "clause_type": clause_type,
        "found": False,
        "relevant_text": "",
        "key_terms": [],
        "risk_level": "N/A",
        "plain_english": f"This contract section does not contain a {clause_type} clause."
    }
    
    return {
        "system": SYSTEM_PROMPT,
        "user": user_msg,
        "assistant": json.dumps(response, indent=2)
    }


print("\n[FORMAT] Creating instruction-tuning examples...")

# Process positive examples
formatted_positive = []
for i, ex in enumerate(positive_train):
    try:
        formatted = create_positive_example(ex)
        formatted_positive.append(formatted)
    except Exception as e:
        pass  # Skip problematic examples
    
    if (i + 1) % 2000 == 0:
        print(f"   Processed {i+1:,} / {len(positive_train):,} positive examples...")

print(f"   Positive examples created: {len(formatted_positive):,}")

# Process negative examples
formatted_negative = []
for ex in negative_sample:
    try:
        formatted = create_negative_example(ex)
        formatted_negative.append(formatted)
    except Exception:
        pass

print(f"   Negative examples created: {len(formatted_negative):,}")

# Combine and shuffle
all_formatted = formatted_positive + formatted_negative
random.shuffle(all_formatted)
print(f"   Total formatted examples:  {len(all_formatted):,}")

# ── 6. Split into Train / Validation ──────────────────
val_size = min(500, int(len(all_formatted) * 0.05))  # 5% or max 500
train_formatted = all_formatted[val_size:]
val_formatted = all_formatted[:val_size]

print(f"\n[SPLIT] Dataset split:")
print(f"   Train: {len(train_formatted):,} examples")
print(f"   Val:   {len(val_formatted):,} examples")

# ── 7. Convert to Chat Format (for Qwen2.5) ───────────
def to_chat_format(example):
    """Convert to the chat message format that Qwen2.5 expects."""
    return {
        "messages": [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]}
        ]
    }

train_chat = [to_chat_format(ex) for ex in train_formatted]
val_chat = [to_chat_format(ex) for ex in val_formatted]

# ── 8. Save to JSONL Files ────────────────────────────
def save_jsonl(data, filepath):
    """Save list of dicts as JSONL (one JSON per line)."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

train_path = os.path.join(OUTPUT_DIR, "train.jsonl")
val_path = os.path.join(OUTPUT_DIR, "val.jsonl")

save_jsonl(train_chat, train_path)
save_jsonl(val_chat, val_path)

train_size_mb = os.path.getsize(train_path) / (1024 * 1024)
val_size_mb = os.path.getsize(val_path) / (1024 * 1024)

print(f"\n[SAVE] Files saved:")
print(f"   Train: {train_path} ({train_size_mb:.1f} MB)")
print(f"   Val:   {val_path} ({val_size_mb:.1f} MB)")

# ── 9. Show Sample Examples ───────────────────────────
print("\n" + "=" * 70)
print("  SAMPLE FORMATTED EXAMPLES")
print("=" * 70)

# Show 2 positive and 1 negative example
shown_pos = 0
shown_neg = 0
for ex in all_formatted:
    parsed = json.loads(ex['assistant'])
    if parsed['found'] and shown_pos < 2:
        shown_pos += 1
        print(f"\n{'=' * 60}")
        print(f"POSITIVE Example {shown_pos}:")
        print(f"  User (first 150 chars): {ex['user'][:150]}...")
        print(f"  Assistant response:")
        print(f"    clause_type:   {parsed['clause_type']}")
        print(f"    found:         {parsed['found']}")
        print(f"    risk_level:    {parsed['risk_level']}")
        print(f"    key_terms:     {parsed['key_terms']}")
        print(f"    relevant_text: {parsed['relevant_text'][:100]}...")
        print(f"    plain_english: {parsed['plain_english'][:150]}...")
    elif not parsed['found'] and shown_neg < 1:
        shown_neg += 1
        print(f"\n{'=' * 60}")
        print(f"NEGATIVE Example:")
        print(f"  User (first 150 chars): {ex['user'][:150]}...")
        print(f"  Assistant response:")
        print(f"    clause_type:   {parsed['clause_type']}")
        print(f"    found:         {parsed['found']}")
        print(f"    plain_english: {parsed['plain_english']}")
    
    if shown_pos >= 2 and shown_neg >= 1:
        break

# ── 10. Statistics ─────────────────────────────────────
print("\n" + "=" * 70)
print("  DATASET STATISTICS")
print("=" * 70)

# Count by clause type
type_counts = defaultdict(int)
risk_counts = defaultdict(int)
for ex in all_formatted:
    parsed = json.loads(ex['assistant'])
    type_counts[parsed['clause_type']] += 1
    if parsed['found']:
        risk_counts[parsed['risk_level']] += 1

print(f"\n   By risk level:")
for risk in ['HIGH', 'MEDIUM', 'LOW']:
    print(f"   {risk:8s}: {risk_counts.get(risk, 0):,} examples")

print(f"\n   Top 10 clause types:")
for i, (ct, count) in enumerate(sorted(type_counts.items(), key=lambda x: -x[1])[:10], 1):
    print(f"   {i:2d}. {ct:<45s} | {count:5d}")

# Average text lengths
user_lens = [len(ex['user']) for ex in all_formatted]
asst_lens = [len(ex['assistant']) for ex in all_formatted]
print(f"\n   Text lengths:")
print(f"   User message:     avg {sum(user_lens)//len(user_lens):,} chars, max {max(user_lens):,}")
print(f"   Assistant output:  avg {sum(asst_lens)//len(asst_lens):,} chars, max {max(asst_lens):,}")

# Save formatting summary
format_summary = {
    "total_examples": len(all_formatted),
    "train_examples": len(train_formatted),
    "val_examples": len(val_formatted),
    "positive_examples": len(formatted_positive),
    "negative_examples": len(formatted_negative),
    "risk_distribution": dict(risk_counts),
    "format": "chat_messages (system + user + assistant)",
    "model_target": "Qwen2.5-3B-Instruct",
    "avg_user_length": sum(user_lens) // len(user_lens),
    "avg_assistant_length": sum(asst_lens) // len(asst_lens),
}

summary_path = os.path.join(OUTPUT_DIR, "format_summary.json")
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(format_summary, f, indent=2)

print(f"\n[OK] Format summary saved to: {summary_path}")
print("\n" + "=" * 70)
print("  DONE! Files ready for training:")
print(f"  Train: {train_path}")
print(f"  Val:   {val_path}")
print(f"  Next:  Upload to Google Colab and train!")
print("=" * 70)
