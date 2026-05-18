# code_auditor.py – LLM Code Complexity Auditor

**Triage Python functions that deserve human review.**  
A high score does **not** prove code is hallucinated. It means the function is structurally complex, statistically unusual, risky, or hard to maintain – and therefore worth a closer look.

---

## Why this tool

LLMs generate plausible‑looking code fast, but they can produce functions that are:

- **Too complex** – high cyclomatic depth, deep nesting, many arguments  
- **Statistically odd** – unusual character entropy, irregular opcode patterns  
- **Risky** – broad except, `eval`, `shell=True`, mutable defaults, dynamic attributes  

`code_auditor` quantifies these signals into a single **0–100 complexity score** so you can focus your review where it matters most.

---

## How it works

For each function/method found in `.py` files, the auditor computes:

| Metric | Source | What it measures |
|--------|--------|------------------|
| **Source entropy** | Characters | Lexical diversity of the function text |
| **Opcode pattern entropy** | Bytecode (multi‑scale) | Unpredictability of execution flow signals |
| **Fractal dimension** | Bytecode shape | Self‑similar complexity of the opcode sequence |
| **AST metrics** | Abstract Syntax Tree | LOC, cyclomatic complexity, nesting depth, calls, returns, arguments |
| **Risk patterns** | AST | Broad/silent except, `eval/exec`, `shell=True`, dynamic attribute, mutable defaults, bare raises, `assert` usage |

These are combined into a **maintainability**, **anomaly**, and **risk** sub‑score, then fused into the final **complexity score (0–100)**.

**Explainable always:** every function carries a list of *reasons* it was flagged – view them with `--explain`.

---

## Features

- **Zero external dependencies** – pure Python, single file  
- **Qualified names** for methods (`Class.method`) and nested functions  
- **Directory scanning** with smart `.venv` / `build` / test‑file exclusion  
- **Coloured terminal output**, automatically disabled for non‑TTY or file output  
- **JSON export** for further analysis  
- **CI integration** – exit code 2 if any function exceeds `--fail-above N`  
- **Sort**, filter (`--min-score`), and slice top‑scoring functions  
- **Show source** of the most suspicious functions with `--top-source N`

---

## Quick start

```bash
# Analyse a single file
python code_auditor.py my_file.py

# Analyse a whole project, show explanations, only show functions score ≥ 50
python code_auditor.py src/ --explain --min-score 50

# Export JSON
python code_auditor.py src/ --format json --output audit.json

# CI gate: fail if any function scores ≥ 85
python code_auditor.py src/ --fail-above 85
```

If no metric flag is given, **all** metrics are computed.

---

## Interpreting the score

| Score | Colour  | Meaning |
|-------|---------|---------|
| 75–100   | Red    | Likely needs review – high complexity, anomaly, or risk |
| 50–74    | Yellow | Moderate signals – worth a glance |
| 0–49     | Green  | Routine – probably fine, but still inspect if `--explain` lists something |

Remember: a low score doesn't mean “perfect” – just that the function lacks the patterns this tool measures.

---

## Advanced usage

```bash
# Only compute opcode entropy (no source entropy, no fractal)
python code_auditor.py src/ --mse

# Custom scales and embedding length
python code_auditor.py src/ --mse --scales 2 4 8 16 --m 3

# Verbose mode: extra AST/risk details, full MSE profiles
python code_auditor.py src/ --verbose

# Exclude additional directories
python code_auditor.py src/ --exclude vendor legacy

# Include test files (by default they are skipped)
python code_auditor.py src/ --include-tests

# Show source of the top 5 most complex functions
python code_auditor.py src/ --top-source 5

# Disable colour
python code_auditor.py src/ --no-color
```

---

## Limitations

- **Language‑specific** – Python bytecode only (for now).  
- **No semantic understanding** – doesn't guess *intent*, only structure and risk.  
- **Thresholds are empirical** – tuned for typical Python codebases; you may want to adjust `--min-score` per project.  

---

## License & contributing

MIT – use it, share it, improve it.  
Issues and pull requests welcome at [github/ladore/code_auditor](#).

---
