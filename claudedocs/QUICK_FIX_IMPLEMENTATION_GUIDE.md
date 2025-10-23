# Quick Fix Implementation Guide
## From Dionysus-2.0 to AgentFlow Integration in 6 Hours

---

## Hour 1: Minimal QA Generator

Create `dionysus-source/minimal_qa_gen.py`:

```python
#!/usr/bin/env python3
"""
Minimal QA Generator - No consciousness metaphors, just QA pairs
"""
import pandas as pd
import json
from pathlib import Path
from openai import OpenAI

client = OpenAI()

def extract_text(pdf_path: str) -> str:
    """Extract text from PDF or TXT file"""
    if pdf_path.endswith('.txt'):
        return Path(pdf_path).read_text()
    # For PDFs, use pypdf or pdfplumber
    # For now, assume .txt files
    return Path(pdf_path).read_text()

def generate_qa_batch(paper_text: str, n_questions: int = 10) -> list:
    """Generate QA pairs from paper using GPT-4"""
    prompt = f"""
    You are a neuroscience education expert. Read this research paper excerpt and generate {n_questions} question-answer pairs for training an AI agent.

    Paper excerpt:
    {paper_text[:4000]}

    Generate questions that test:
    1. Conceptual understanding (definitions, mechanisms)
    2. Application knowledge (how concepts are measured/applied)
    3. Comparative reasoning (relationships between concepts)

    Format EXACTLY as:
    Q1: [question]
    A1: [2-3 sentence answer from the paper]

    Q2: [question]
    A2: [answer]

    Continue for all {n_questions} questions.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return parse_qa_text(response.choices[0].message.content)

def parse_qa_text(text: str) -> list:
    """Parse Q1/A1 format into list of dicts"""
    qa_pairs = []
    lines = text.strip().split('\n')

    current_q = None
    for line in lines:
        line = line.strip()
        if line.startswith('Q') and ':' in line:
            current_q = line.split(':', 1)[1].strip()
        elif line.startswith('A') and ':' in line and current_q:
            current_a = line.split(':', 1)[1].strip()
            qa_pairs.append({
                'question': current_q,
                'result': current_a  # AgentFlow uses 'result', not 'answer'
            })
            current_q = None

    return qa_pairs

# Main execution
if __name__ == "__main__":
    paper_files = list(Path("consciousness_papers").glob("*.txt"))[:20]  # Start with 20

    all_qa = []
    for i, paper_file in enumerate(paper_files, 1):
        print(f"[{i}/{len(paper_files)}] Processing: {paper_file.name}")

        text = extract_text(str(paper_file))
        qa_pairs = generate_qa_batch(text, n_questions=10)

        for qa in qa_pairs:
            qa['metadata'] = json.dumps({'paper': paper_file.stem})

        all_qa.extend(qa_pairs)
        print(f"  Generated {len(qa_pairs)} QA pairs")

    # Create DataFrame
    df = pd.DataFrame(all_qa)

    # Split 80/20
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    # Save as parquet
    Path("agentflow_data").mkdir(exist_ok=True)
    train_df.to_parquet("agentflow_data/train.parquet")
    val_df.to_parquet("agentflow_data/val.parquet")

    print(f"\n✅ Done! {len(train_df)} train, {len(val_df)} val pairs")
    print(f"Saved to: agentflow_data/train.parquet")
```

**Install dependencies**:
```bash
pip install openai pandas pyarrow
```

**Run it**:
```bash
python dionysus-source/minimal_qa_gen.py
```

---

## Hour 2: Verify Output Format

```python
# verify_output.py
import pandas as pd

df = pd.read_parquet('agentflow_data/train.parquet')

print("Schema check:")
print(f"Columns: {df.columns.tolist()}")
assert 'question' in df.columns, "Missing 'question' column!"
assert 'result' in df.columns, "Missing 'result' column!"

print("\nSample data:")
print(df.head(3))

print("\nStats:")
print(f"Total QA pairs: {len(df)}")
print(f"Avg question length: {df['question'].str.len().mean():.0f} chars")
print(f"Avg answer length: {df['result'].str.len().mean():.0f} chars")

print("\n✅ Format is AgentFlow-compatible!")
```

---

## Hour 3: Custom Reward Function

Create `agentflow_integration/neuroscience_reward.py`:

```python
"""
Neuroscience-specific reward function for AgentFlow
"""
from sentence_transformers import SentenceTransformer, util
import re

class NeuroscienceReward:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def compute_score(self, prediction: str, ground_truth: str) -> float:
        """
        Reward function for neuroscience QA
        Returns: float in [0, 1]
        """
        # 1. Semantic similarity (main signal)
        pred_emb = self.model.encode(prediction, convert_to_tensor=True)
        gt_emb = self.model.encode(ground_truth, convert_to_tensor=True)
        similarity = util.cos_sim(pred_emb, gt_emb).item()

        # 2. Keyword overlap bonus
        pred_words = set(re.findall(r'\b\w+\b', prediction.lower()))
        gt_words = set(re.findall(r'\b\w+\b', ground_truth.lower()))
        keyword_overlap = len(pred_words & gt_words) / max(len(gt_words), 1)

        # 3. Length penalty (too short or too long)
        length_ratio = len(prediction) / max(len(ground_truth), 1)
        length_penalty = 1.0 if 0.5 < length_ratio < 1.5 else 0.8

        # Combine scores
        raw_score = (
            similarity * 0.7 +
            keyword_overlap * 0.2 +
            length_penalty * 0.1
        )

        # Map [0.5, 1.0] → [0.0, 1.0] for gradient signal
        return max(0.0, min(1.0, (raw_score - 0.5) * 2))

# Test it
if __name__ == "__main__":
    reward = NeuroscienceReward()

    test_cases = [
        ("Active inference is a Bayesian framework", "Active inference uses Bayesian principles", 0.9),
        ("The brain predicts sensory input", "Active inference is a Bayesian framework", 0.3),
        ("I don't know", "Active inference is a Bayesian framework", 0.0),
    ]

    for pred, gt, expected in test_cases:
        score = reward.compute_score(pred, gt)
        print(f"Score: {score:.2f} (expected ~{expected:.1f})")
        print(f"  Pred: {pred}")
        print(f"  GT:   {gt}\n")
```

**Install & test**:
```bash
pip install sentence-transformers
python agentflow_integration/neuroscience_reward.py
```

---

## Hour 4: AgentFlow Config Integration

Assuming AgentFlow is at `/path/to/AgentFlow`:

```bash
# Copy data files
cp agentflow_data/*.parquet /path/to/AgentFlow/data/train/

# Edit AgentFlow config
cd /path/to/AgentFlow
```

Edit `train/config.yaml`:

```yaml
data:
  train_files: ['data/train/train.parquet']
  val_files: ['data/train/val.parquet']

algorithm:
  adv_estimator: grpo  # Flow-GRPO
  kl_coef: 0.05

# Custom reward (optional - start with default 0/1)
# We'll add custom reward in Hour 5
```

Edit `train/rollout.py` to add custom reward (insert after imports):

```python
# ADD THIS
import sys
sys.path.append('/path/to/Dionysus-2.0/agentflow_integration')
from neuroscience_reward import NeuroscienceReward

# REPLACE the default compute_score function
neuroscience_reward = NeuroscienceReward()

def compute_score(prediction: str, ground_truth: str) -> float:
    """Custom neuroscience semantic similarity reward"""
    return neuroscience_reward.compute_score(prediction, ground_truth)
```

---

## Hour 5: Test Training Run

```bash
cd /path/to/AgentFlow

# Start rollout server (in one terminal)
bash train/serve_with_logs.sh

# Start training (in another terminal)
python -m agentflow.verl fit \
  --config train/config.yaml \
  --trainer.max_epochs=1 \
  --data.train_files=data/train/train.parquet \
  --data.val_files=data/train/val.parquet
```

**Expected Output**:
```
Epoch 1/1:
  Rollout: 100 episodes
  Avg reward: 0.65
  Policy loss: 0.12
  Value loss: 0.08

Validation:
  Accuracy: 72%
  Avg semantic similarity: 0.68
```

---

## Hour 6: Troubleshooting & Iteration

### Common Issues

**Issue 1**: "KeyError: 'result'"
```python
# Fix: Check parquet schema
df = pd.read_parquet('data/train/train.parquet')
print(df.columns)  # Must have 'result', not 'answer'
```

**Issue 2**: "Reward always 0"
```python
# Fix: Check reward function
from neuroscience_reward import NeuroscienceReward
reward = NeuroscienceReward()
print(reward.compute_score("test answer", "test answer"))  # Should be ~1.0
```

**Issue 3**: "Out of memory during training"
```yaml
# Fix: Reduce batch size in config.yaml
algorithm:
  batch_size: 4  # Try smaller batches
```

**Issue 4**: "No GPU detected"
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, run on CPU (slower but works)
# Edit config: trainer.accelerator = 'cpu'
```

---

## Validation Checklist

After 6 hours, you should have:

- [x] Generated 200+ QA pairs from neuroscience papers
- [x] Saved as `train.parquet` and `val.parquet`
- [x] Verified schema matches AgentFlow requirements
- [x] Implemented custom semantic similarity reward
- [x] Integrated reward into AgentFlow rollout
- [x] Ran 1 epoch of training successfully
- [x] Validated on held-out papers

---

## Next Steps (After Proof-of-Concept)

### Week 2: Scale Up
1. Process all 500 papers (not just 20)
2. Generate 5,000 QA pairs (10 per paper)
3. Add question diversity (use different prompts)
4. Implement quality filters

### Week 3: Advanced Rewards
1. Add domain-specific bonuses for neuroscience terms
2. Implement multi-turn dialogue rewards
3. Add citation accuracy checks
4. Weight harder questions higher

### Week 4: ActionModels.jl Integration
1. Use ActionModels.jl to generate cognitive priors
2. Simulate expert vs novice reasoning patterns
3. Use as additional training signal
4. Compare agent behavior to human models

---

## Cost Estimate

**OpenAI API Costs** (for initial 20 papers):
- 20 papers × 10 QA pairs × $0.01/generation ≈ **$2**

**Compute Costs** (for 1 epoch training):
- Local GPU: Free
- Google Colab Pro: $10/month
- Cloud GPU (A100): ~$1/hour

**Total PoC Cost**: **~$5-15**

---

## Success Metrics

After 6 hours, you'll know it works if:

1. ✅ Parquet files load without errors
2. ✅ AgentFlow accepts the data format
3. ✅ Training completes without crashes
4. ✅ Validation accuracy > 60% (random baseline is ~0%)
5. ✅ Agent generates neuroscience-relevant answers

---

## Emergency Rollback

If things go wrong, you still have Dionysus-2.0:

```bash
# Keep consciousness analysis separate
cd Dionysus-2.0
git checkout -b agentflow-integration

# Work in branch
# Main system stays untouched
```

---

## Summary

**Time Investment**: 6 hours
**Lines of Code**: ~200 (vs your current 671 files)
**Output**: Working AgentFlow integration
**Risk**: Low (doesn't break existing Dionysus)

**Key Principle**: Start with the simplest thing that works, then iterate.
