#!/usr/bin/env python3
"""
code_auditor.py – Production-grade LLM Code Review Complexity Auditor (v4.4.0)

Purpose
-------
Triage Python functions that deserve human review. The tool combines:
  • AST maintainability metrics: LOC, cyclomatic complexity, nesting, calls,
    returns, assignments, comprehensions, parameters
  • Risk/code-smell detection: broad/silent excepts, eval/exec, shell=True,
    dynamic attributes, mutable defaults, runtime asserts, globals/nonlocals
  • Source character entropy
  • Opcode pattern entropy across multiple scales
  • Experimental opcode-shape fractal signal

Important framing
-----------------
A high score does NOT prove code is hallucinated. It means the function is more
likely to deserve review because it is structurally complex, statistically
unusual, risky, or hard to maintain.

v4.4.0 fixes
------------
  • Fixed ANSI score-column truncation artifacts.
  • Fixed nested-function metric contamination: parent functions no longer absorb
    the full body complexity of nested local helpers.
  • Fixed nested-function bytecode lookup by normalizing .<locals>. qualnames and
    using a line/name fallback map.
  • Fixed silent-except detection for pass / ellipsis / constant-only bodies.
  • Rebalanced scoring toward maintainability for developer-review usefulness.
  • Restored argument validation.
  • Kept Nest in the compact table because it is a strong review signal.
  • Removed misleading "single-pass AST" claim; this uses optimized per-function
    analysis with O(1)-style bytecode lookup maps.

Compatibility
-------------
Fully backward-compatible with previous CLI flags:
  • positional paths
  • --source-entropy
  • --mse
  • --fractal
  • --scales
  • --m
  • --format table|json
  • --verbose
  • --top-source
  • --output
  • --explain
  • --min-score
  • --sort
  • --fail-above
  • --exclude
  • --include-tests
  • --no-color
  • --version

Usage
-----
    python code_auditor.py <path>
    python code_auditor.py src --explain --min-score 50
    python code_auditor.py src --format json --output audit.json
    python code_auditor.py src --fail-above 85
"""

from __future__ import annotations

import argparse
import ast
import dis
import io
import json
import math
import statistics
import sys
import types
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Tuple

# ------------------------------------------------------------------------------
# ANSI colors
# ------------------------------------------------------------------------------
RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"

# ------------------------------------------------------------------------------
# Version & constants
# ------------------------------------------------------------------------------
__version__ = "4.4.0"

DEFAULT_SCALES = [1, 2, 4, 8, 16]
DEFAULT_M = 2
MIN_BYTECODE_LENGTH = 8
MAX_OPCODE_PATTERN_ENTROPY = 5.0

# Maintainability is the most accurate primary signal for developer review.
# Anomaly remains useful, but should not outrank structural complexity.
MAINTAINABILITY_WEIGHT = 0.60
ANOMALY_WEIGHT = 0.20
RISK_WEIGHT = 0.20

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "node_modules",
    "site-packages",
    "dist",
    "build",
    "htmlcov",
}

DEFAULT_TEST_DIR_NAMES = {"test", "tests", "testing", "spec", "specs"}

SortKey = Literal[
    "complexity",
    "risk",
    "maintainability",
    "anomaly",
    "loc",
    "cyclomatic",
    "mse",
    "fractal",
    "source",
]

# ------------------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class ASTMetrics:
    loc: int = 0
    effective_loc: int = 0
    arg_count: int = 0
    decorator_count: int = 0
    cyclomatic: int = 1
    max_ast_depth: int = 0
    max_control_depth: int = 0
    branches: int = 0
    loops: int = 0
    try_blocks: int = 0
    bool_ops: int = 0
    calls: int = 0
    returns: int = 0
    assignments: int = 0
    comprehensions: int = 0
    lambdas: int = 0
    nested_functions: int = 0


@dataclass(frozen=True)
class RiskMetrics:
    broad_excepts: int = 0
    silent_excepts: int = 0
    eval_exec_calls: int = 0
    shell_true: int = 0
    dynamic_attr: int = 0
    mutable_defaults: int = 0
    assert_statements: int = 0
    global_statements: int = 0
    nonlocal_statements: int = 0
    bare_raise_outside_except: int = 0
    import_star: int = 0

    @property
    def dangerous_calls(self) -> int:
        return self.eval_exec_calls + self.shell_true

    @property
    def total_risk_events(self) -> int:
        return sum(asdict(self).values())


@dataclass
class AuditResult:
    file: str
    name: str
    lineno: int
    end_lineno: int
    src_entropy: Optional[float] = None
    opcode_entropy_mean: Optional[float] = None
    mse_area: Optional[float] = None
    mse_area_normalized: Optional[float] = None
    mse_profile: Optional[List[Tuple[int, float]]] = None
    fractal: Optional[float] = None
    ast_metrics: ASTMetrics = field(default_factory=ASTMetrics)
    risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)
    complexity: int = 0
    maintainability_score: int = 0
    anomaly_score: int = 0
    risk_score: int = 0
    reasons: List[str] = field(default_factory=list)
    source_snippet: str = ""

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------------------------
def normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def safe_mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else 0.0


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in freq.values())


def opcodes_from_code(code: types.CodeType) -> List[int]:
    return [instr.opcode for instr in dis.get_instructions(code)]


def is_probably_test_file(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    if parts & DEFAULT_TEST_DIR_NAMES:
        return True
    name = path.name.lower()
    return name.startswith("test_") or name.endswith("_test.py")


def should_use_color(no_color: bool, output_path: Optional[Path]) -> bool:
    if no_color or output_path is not None:
        return False
    return sys.stdout.isatty()


def colorize(text: str, color: str, enabled: bool) -> str:
    return f"{color}{text}{RESET}" if enabled else text


def color_score(score: int, enabled: bool) -> str:
    raw = str(score)
    if score >= 75:
        return colorize(raw, RED, enabled)
    if score >= 50:
        return colorize(raw, YELLOW, enabled)
    return colorize(raw, GREEN, enabled)


def score_cell(score: int, width: int, enabled: bool) -> str:
    """Format a colored score without passing ANSI codes through truncation."""
    raw = str(score)
    padding = " " * max(0, width - len(raw))
    return color_score(score, enabled) + padding


# ------------------------------------------------------------------------------
# Opcode pattern entropy & MSE
# ------------------------------------------------------------------------------
def _opcode_pattern_entropy(seq: List[int], m: int = DEFAULT_M) -> float:
    """
    Exact opcode n-gram novelty inspired by sample entropy.

    This is intentionally not continuous Sample Entropy with tolerance r. Python
    bytecode is symbolic/discrete, so exact opcode continuation novelty is more
    interpretable for this use case.
    """
    n = len(seq)
    if n < m + 1:
        return 0.0

    count_m: Counter[Tuple[int, ...]] = Counter()
    for i in range(n - m + 1):
        count_m[tuple(seq[i : i + m])] += 1

    count_m1: Counter[Tuple[int, ...]] = Counter()
    for i in range(n - m):
        count_m1[tuple(seq[i : i + m + 1])] += 1

    b = sum(c * (c - 1) for c in count_m.values())
    a = sum(c * (c - 1) for c in count_m1.values())

    if b == 0:
        return 0.0
    if a == 0:
        return MAX_OPCODE_PATTERN_ENTROPY
    return min(MAX_OPCODE_PATTERN_ENTROPY, -math.log(a / b))


# Backward-compatible internal alias for older references.
_sample_entropy_fast = _opcode_pattern_entropy


def _coarse_grain_mode(seq: List[int], tau: int) -> List[int]:
    if tau <= 1:
        return seq[:]

    n = len(seq) // tau
    out: List[int] = []

    for block_index in range(n):
        block = seq[block_index * tau : (block_index + 1) * tau]
        if not block:
            continue
        counts = Counter(block)
        # Stable deterministic tie-break: first occurrence in the block wins.
        mode_val = max(counts, key=lambda opcode: (counts[opcode], -block.index(opcode)))
        out.append(mode_val)

    return out


def multi_scale_entropy(
    seq: List[int],
    scales: Optional[List[int]] = None,
    m: int = DEFAULT_M,
) -> List[Tuple[int, float]]:
    if scales is None:
        scales = DEFAULT_SCALES

    profile: List[Tuple[int, float]] = []
    for tau in sorted(set(scales)):
        if tau <= 0 or tau > len(seq):
            continue
        coarse = _coarse_grain_mode(seq, tau) if tau > 1 else seq
        profile.append((tau, _opcode_pattern_entropy(coarse, m)))
    return profile


def mse_area(profile: List[Tuple[int, float]]) -> float:
    if len(profile) < 2:
        return 0.0

    area = 0.0
    for i in range(len(profile) - 1):
        x1, y1 = profile[i]
        x2, y2 = profile[i + 1]
        area += 0.5 * (x2 - x1) * (y1 + y2)
    return area


def mse_area_normalized(profile: List[Tuple[int, float]]) -> float:
    if len(profile) < 2:
        return 0.0

    area = mse_area(profile)
    max_area = sum(
        0.5 * (x2 - x1) * (MAX_OPCODE_PATTERN_ENTROPY * 2)
        for (x1, _), (x2, _) in zip(profile, profile[1:])
    )
    if max_area <= 0:
        return 0.0
    return max(0.0, min(1.0, area / max_area))


# ------------------------------------------------------------------------------
# Experimental fractal dimension
# ------------------------------------------------------------------------------
def fractal_dimension(seq: List[int]) -> float:
    """
    Experimental box-counting style metric over opcode-sequence shape.

    Retained for backward compatibility and anomaly exploration. Interpret as a
    weak secondary signal, not as formal mathematical proof of code complexity.
    """
    n = len(seq)
    if n < MIN_BYTECODE_LENGTH:
        return 0.0

    min_size = max(2, n // 32)
    max_size = max(min_size + 1, n // 2)
    if max_size <= 2:
        return 0.0

    sizes = [2**k for k in range(int(math.log2(min_size)), int(math.log2(max_size)) + 1)]
    sizes = [size for size in sizes if 2 <= size <= max_size]
    if len(sizes) < 3:
        return 0.0

    log_scales: List[float] = []
    log_counts: List[float] = []

    min_op, max_op = min(seq), max(seq)
    span = max(1, max_op - min_op)

    for size in sizes:
        boxes: set[Tuple[int, int]] = set()
        y_bins = max(1, math.ceil(n / size))

        for i, opcode in enumerate(seq):
            bx = i // size
            normalized_y = (opcode - min_op) / span
            by = min(y_bins - 1, int(normalized_y * y_bins))
            boxes.add((bx, by))

        if boxes:
            log_scales.append(math.log(size))
            log_counts.append(math.log(len(boxes)))

    if len(log_scales) < 3:
        return 0.0

    points = len(log_scales)
    sx = sum(log_scales)
    sy = sum(log_counts)
    sxx = sum(x * x for x in log_scales)
    sxy = sum(x * y for x, y in zip(log_scales, log_counts))
    denom = points * sxx - sx * sx

    if abs(denom) < 1e-12:
        return 0.0

    slope = (points * sxy - sx * sy) / denom
    return abs(slope)


# ------------------------------------------------------------------------------
# Function collection
# ------------------------------------------------------------------------------
class FunctionCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.functions: List[Dict[str, Any]] = []
        self._qual_stack: List[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._qual_stack.append(node.name)
        self.generic_visit(node)
        self._qual_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record(node)

    def _record(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = ".".join([*self._qual_stack, node.name])
        self.functions.append(
            {
                "node": node,
                "qualname": qualname,
                "lineno": node.lineno,
                "end_lineno": getattr(node, "end_lineno", node.lineno),
            }
        )

        self._qual_stack.append(node.name)
        self.generic_visit(node)
        self._qual_stack.pop()


def collect_functions(tree: ast.AST) -> List[Dict[str, Any]]:
    collector = FunctionCollector()
    collector.visit(tree)
    return collector.functions


# ------------------------------------------------------------------------------
# Per-function AST metrics and risk detection
# ------------------------------------------------------------------------------
def _is_constant_only_expr(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)


def _is_silent_stmt(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Expr):
        value = stmt.value
        if isinstance(value, ast.Constant):
            return True
        # Python may parse literal ellipsis as Constant(Ellipsis), but keep this
        # for compatibility with older AST shapes.
        if isinstance(value, ast.Ellipsis):  # pragma: no cover on newer Python
            return True
    return False


class FunctionMetricsVisitor(ast.NodeVisitor):
    """
    Metrics visitor for one root function.

    Nested local functions are counted as local abstractions but their bodies are
    intentionally not traversed. This prevents parent functions from absorbing
    the full complexity/risk profile of inner helpers, while still recording the
    existence of nested helpers as a maintainability signal.
    """

    def __init__(self) -> None:
        self.branches = 0
        self.loops = 0
        self.try_blocks = 0
        self.bool_ops = 0
        self.calls = 0
        self.returns = 0
        self.assignments = 0
        self.comprehensions = 0
        self.lambdas = 0
        self.nested_functions = 0

        self.max_ast_depth = 0
        self._ast_depth = 0
        self.max_control_depth = 0
        self._control_depth = 0

        self.broad_excepts = 0
        self.silent_excepts = 0
        self.eval_exec_calls = 0
        self.shell_true = 0
        self.dynamic_attr = 0
        self.mutable_defaults = 0
        self.assert_statements = 0
        self.global_statements = 0
        self.nonlocal_statements = 0
        self.bare_raise_outside_except = 0
        self.import_star = 0

        self._except_depth = 0
        self._root_function_seen = False

    def generic_visit(self, node: ast.AST) -> None:
        self._ast_depth += 1
        self.max_ast_depth = max(self.max_ast_depth, self._ast_depth)
        super().generic_visit(node)
        self._ast_depth -= 1

    def _visit_control(self, node: ast.AST) -> None:
        self._control_depth += 1
        self.max_control_depth = max(self.max_control_depth, self._control_depth)
        self.generic_visit(node)
        self._control_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not self._root_function_seen:
            self._root_function_seen = True
            self._check_mutable_defaults(node)
            self.generic_visit(node)
            return

        self.nested_functions += 1
        # Count a nested helper as an abstraction/call-like maintainability event,
        # but do not descend into the body.
        self.calls += 1
        self._check_mutable_defaults(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if not self._root_function_seen:
            self._root_function_seen = True
            self._check_mutable_defaults(node)
            self.generic_visit(node)
            return

        self.nested_functions += 1
        self.calls += 1
        self._check_mutable_defaults(node)

    def _check_mutable_defaults(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]
        for default in defaults:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.mutable_defaults += 1
            elif isinstance(default, ast.Call) and isinstance(default.func, ast.Name):
                if default.func.id in {"list", "dict", "set", "defaultdict"}:
                    self.mutable_defaults += 1

    def visit_If(self, node: ast.If) -> None:
        self.branches += 1
        self._visit_control(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.branches += 1
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:  # Python 3.10+
        self.branches += max(1, len(node.cases))
        self._visit_control(node)

    def visit_For(self, node: ast.For) -> None:
        self.loops += 1
        self._visit_control(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.loops += 1
        self._visit_control(node)

    def visit_While(self, node: ast.While) -> None:
        self.loops += 1
        self._visit_control(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.try_blocks += 1
        self.branches += len(node.handlers)
        self._visit_control(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.bool_ops += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.calls += 1

        if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
            self.eval_exec_calls += 1

        if isinstance(node.func, ast.Name) and node.func.id in {"getattr", "setattr", "delattr"}:
            self.dynamic_attr += 1

        for kw in node.keywords:
            if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                self.shell_true += 1

        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.returns += 1
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.assignments += 1
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.assignments += 1
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.assignments += 1
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.comprehensions += 1
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.comprehensions += 1
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.comprehensions += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.comprehensions += 1
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.lambdas += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.broad_excepts += 1
        elif isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}:
            self.broad_excepts += 1
        elif isinstance(node.type, ast.Tuple):
            if any(
                isinstance(elt, ast.Name) and elt.id in {"Exception", "BaseException"}
                for elt in node.type.elts
            ):
                self.broad_excepts += 1

        meaningful_body = [stmt for stmt in node.body if not _is_silent_stmt(stmt)]
        if not meaningful_body:
            self.silent_excepts += 1

        self._except_depth += 1
        self.generic_visit(node)
        self._except_depth -= 1

    def visit_Assert(self, node: ast.Assert) -> None:
        self.assert_statements += 1
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.global_statements += 1
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocal_statements += 1
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        if node.exc is None and self._except_depth == 0:
            self.bare_raise_outside_except += 1
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if any(alias.name == "*" for alias in node.names):
            self.import_star += 1
        self.generic_visit(node)


def _effective_loc_from_segment(segment: str) -> int:
    count = 0
    for line in segment.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


def _arg_count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    args = node.args
    total = len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs)
    if args.vararg is not None:
        total += 1
    if args.kwarg is not None:
        total += 1
    return total


def build_ast_metrics(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    visitor: FunctionMetricsVisitor,
    source_segment: str,
) -> ASTMetrics:
    loc = max(1, getattr(node, "end_lineno", node.lineno) - node.lineno + 1)
    effective_loc = _effective_loc_from_segment(source_segment) if source_segment else loc
    cyclomatic = (
        1
        + visitor.branches
        + visitor.loops
        + visitor.try_blocks
        + visitor.bool_ops
        + visitor.comprehensions
    )

    return ASTMetrics(
        loc=loc,
        effective_loc=effective_loc,
        arg_count=_arg_count(node),
        decorator_count=len(node.decorator_list),
        cyclomatic=cyclomatic,
        max_ast_depth=visitor.max_ast_depth,
        max_control_depth=visitor.max_control_depth,
        branches=visitor.branches,
        loops=visitor.loops,
        try_blocks=visitor.try_blocks,
        bool_ops=visitor.bool_ops,
        calls=visitor.calls,
        returns=visitor.returns,
        assignments=visitor.assignments,
        comprehensions=visitor.comprehensions,
        lambdas=visitor.lambdas,
        nested_functions=visitor.nested_functions,
    )


def build_risk_metrics(visitor: FunctionMetricsVisitor) -> RiskMetrics:
    return RiskMetrics(
        broad_excepts=visitor.broad_excepts,
        silent_excepts=visitor.silent_excepts,
        eval_exec_calls=visitor.eval_exec_calls,
        shell_true=visitor.shell_true,
        dynamic_attr=visitor.dynamic_attr,
        mutable_defaults=visitor.mutable_defaults,
        assert_statements=visitor.assert_statements,
        global_statements=visitor.global_statements,
        nonlocal_statements=visitor.nonlocal_statements,
        bare_raise_outside_except=visitor.bare_raise_outside_except,
        import_star=visitor.import_star,
    )


# ------------------------------------------------------------------------------
# Fast bytecode lookup
# ------------------------------------------------------------------------------
def _normalize_qualname(name: str) -> str:
    return name.replace(".<locals>.", ".")


@dataclass(frozen=True)
class CodeMaps:
    by_qualname: Dict[Tuple[int, str], types.CodeType]
    by_name: Dict[Tuple[int, str], types.CodeType]


def _build_code_maps(module_code: types.CodeType) -> CodeMaps:
    by_qualname: Dict[Tuple[int, str], types.CodeType] = {}
    by_name: Dict[Tuple[int, str], types.CodeType] = {}

    stack = [module_code]
    while stack:
        co = stack.pop()
        qualname = getattr(co, "co_qualname", co.co_name)
        normalized = _normalize_qualname(qualname)

        # Prefer first exact line/name association. Duplicate same-line lambdas or
        # generated constructs are rare; this is only a fallback aid anyway.
        by_qualname.setdefault((co.co_firstlineno, normalized), co)
        by_name.setdefault((co.co_firstlineno, co.co_name), co)

        for const in co.co_consts:
            if isinstance(const, types.CodeType):
                stack.append(const)

    return CodeMaps(by_qualname=by_qualname, by_name=by_name)


def lookup_code_object(code_maps: CodeMaps, lineno: int, qualname: str, short_name: str) -> Optional[types.CodeType]:
    code_obj = code_maps.by_qualname.get((lineno, qualname))
    if code_obj is not None:
        return code_obj
    return code_maps.by_name.get((lineno, short_name))


# ------------------------------------------------------------------------------
# Scoring and explanations
# ------------------------------------------------------------------------------
def maintainability_subscore(ast_m: ASTMetrics) -> int:
    components = [
        normalize(ast_m.cyclomatic, 4, 20) * 0.25,
        normalize(ast_m.loc, 30, 180) * 0.20,
        normalize(ast_m.max_control_depth, 2, 8) * 0.20,
        normalize(ast_m.arg_count, 4, 12) * 0.10,
        normalize(ast_m.calls, 8, 80) * 0.08,
        normalize(ast_m.returns, 2, 12) * 0.05,
        normalize(ast_m.assignments, 8, 80) * 0.05,
        normalize(ast_m.comprehensions, 1, 8) * 0.04,
        normalize(ast_m.nested_functions, 1, 5) * 0.03,
    ]
    return int(round(sum(components) * 100))


def anomaly_subscore(
    src_entropy: Optional[float],
    opcode_entropy_mean: Optional[float],
    mse_area_norm: Optional[float],
    fractal_val: Optional[float],
) -> int:
    weighted = 0.0
    weights = 0.0

    if src_entropy is not None:
        weighted += normalize(src_entropy, 3.5, 5.5) * 0.25
        weights += 0.25

    if opcode_entropy_mean is not None:
        weighted += normalize(opcode_entropy_mean, 1.0, MAX_OPCODE_PATTERN_ENTROPY) * 0.40
        weights += 0.40

    if mse_area_norm is not None:
        weighted += mse_area_norm * 0.25
        weights += 0.25

    if fractal_val is not None:
        weighted += normalize(fractal_val, 0.25, 1.25) * 0.10
        weights += 0.10

    return int(round((weighted / weights) * 100)) if weights > 0 else 0


def risk_subscore(risk_m: RiskMetrics) -> int:
    weighted = 0.0
    weighted += normalize(risk_m.broad_excepts, 0, 4) * 0.15
    weighted += normalize(risk_m.silent_excepts, 0, 3) * 0.15
    weighted += normalize(risk_m.eval_exec_calls, 0, 2) * 0.20
    weighted += normalize(risk_m.shell_true, 0, 2) * 0.20
    weighted += normalize(risk_m.dynamic_attr, 0, 8) * 0.08
    weighted += normalize(risk_m.mutable_defaults, 0, 3) * 0.08
    weighted += normalize(risk_m.assert_statements, 0, 6) * 0.04
    weighted += normalize(risk_m.global_statements + risk_m.nonlocal_statements, 0, 4) * 0.04
    weighted += normalize(risk_m.bare_raise_outside_except, 0, 1) * 0.03
    weighted += normalize(risk_m.import_star, 0, 1) * 0.03
    return int(round(weighted * 100))


def composite_complexity_score(
    maintainability: int,
    anomaly: int,
    risk: int,
    have_anomaly_metrics: bool,
) -> int:
    if have_anomaly_metrics:
        score = (
            maintainability * MAINTAINABILITY_WEIGHT
            + anomaly * ANOMALY_WEIGHT
            + risk * RISK_WEIGHT
        )
    else:
        score = maintainability * 0.75 + risk * 0.25
    return int(round(max(0.0, min(100.0, score))))


def complexity_score(
    src_entropy: Optional[float],
    mse_area_val: Optional[float],
    fractal_val: Optional[float],
) -> int:
    """Backward-compatible legacy scoring helper.

    New code should use maintainability_subscore(), anomaly_subscore(),
    risk_subscore(), and composite_complexity_score().
    """
    mse_norm = normalize(mse_area_val, 0.0, 25.0) if mse_area_val is not None else None
    return anomaly_subscore(src_entropy, None, mse_norm, fractal_val)


def explain_result(result: AuditResult) -> List[str]:
    ast_m = result.ast_metrics
    risk_m = result.risk_metrics
    reasons: List[str] = []

    if ast_m.loc >= 100:
        reasons.append(f"large function: {ast_m.loc} lines")
    elif ast_m.loc >= 50:
        reasons.append(f"medium-large function: {ast_m.loc} lines")

    if ast_m.cyclomatic >= 15:
        reasons.append(f"very high cyclomatic complexity: {ast_m.cyclomatic}")
    elif ast_m.cyclomatic >= 10:
        reasons.append(f"high cyclomatic complexity: {ast_m.cyclomatic}")

    if ast_m.max_control_depth >= 6:
        reasons.append(f"deep control nesting: {ast_m.max_control_depth}")
    elif ast_m.max_control_depth >= 4:
        reasons.append(f"moderate control nesting: {ast_m.max_control_depth}")

    if ast_m.arg_count >= 8:
        reasons.append(f"many parameters: {ast_m.arg_count}")

    if ast_m.calls >= 40:
        reasons.append(f"many function calls: {ast_m.calls}")

    if ast_m.returns >= 8:
        reasons.append(f"many return paths: {ast_m.returns}")

    if ast_m.nested_functions >= 2:
        reasons.append(f"multiple nested local helpers: {ast_m.nested_functions}")

    if result.opcode_entropy_mean is not None and result.opcode_entropy_mean >= 3.0:
        reasons.append(f"high opcode pattern entropy: {result.opcode_entropy_mean:.2f}")

    if result.src_entropy is not None and result.src_entropy >= 4.8:
        reasons.append(f"high source character entropy: {result.src_entropy:.2f}")

    if result.mse_area_normalized is not None and result.mse_area_normalized >= 0.65:
        reasons.append(f"high normalized multi-scale opcode entropy: {result.mse_area_normalized:.2f}")

    if result.fractal is not None and result.fractal >= 1.0:
        reasons.append(f"high experimental opcode-shape fractal signal: {result.fractal:.2f}")

    if risk_m.broad_excepts:
        reasons.append(f"broad exception handlers: {risk_m.broad_excepts}")

    if risk_m.silent_excepts:
        reasons.append(f"silent exception handlers: {risk_m.silent_excepts}")

    if risk_m.eval_exec_calls:
        reasons.append(f"eval/exec calls: {risk_m.eval_exec_calls}")

    if risk_m.shell_true:
        reasons.append(f"subprocess shell=True usage: {risk_m.shell_true}")

    if risk_m.dynamic_attr >= 3:
        reasons.append(f"heavy dynamic attribute usage: {risk_m.dynamic_attr}")

    if risk_m.mutable_defaults:
        reasons.append(f"mutable default arguments: {risk_m.mutable_defaults}")

    if risk_m.assert_statements >= 3:
        reasons.append(f"multiple runtime assert statements: {risk_m.assert_statements}")

    if not reasons:
        reasons.append("combined moderate signals")

    return reasons


# ------------------------------------------------------------------------------
# Core auditor
# ------------------------------------------------------------------------------
class CodeAuditor:
    def __init__(
        self,
        scales: Optional[List[int]] = None,
        m: int = DEFAULT_M,
        compute_source: bool = True,
        compute_mse: bool = True,
        compute_fractal: bool = True,
    ) -> None:
        self.scales = scales or DEFAULT_SCALES
        self.m = m
        self.compute_source = compute_source
        self.compute_mse = compute_mse
        self.compute_fractal = compute_fractal

    def analyse_file(self, path: Path) -> List[AuditResult]:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise RuntimeError(f"Cannot read {path}: {exc}") from exc

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise RuntimeError(f"Syntax error in {path}: {exc}") from exc

        funcs_info = collect_functions(tree)
        if not funcs_info:
            return []

        try:
            module_code = compile(source, str(path), "exec")
            code_maps = _build_code_maps(module_code)
        except SyntaxError:
            code_maps = CodeMaps(by_qualname={}, by_name={})

        source_lines = source.splitlines()
        results: List[AuditResult] = []

        for info in funcs_info:
            node = info["node"]
            name = info["qualname"]
            lineno = info["lineno"]
            end_lineno = info["end_lineno"]

            try:
                segment = ast.get_source_segment(source, node) or "\n".join(source_lines[lineno - 1 : end_lineno])
            except Exception:
                segment = "\n".join(source_lines[lineno - 1 : end_lineno])
            snippet = segment.strip()

            src_entropy = shannon_entropy(segment) if self.compute_source else None

            visitor = FunctionMetricsVisitor()
            visitor.visit(node)
            ast_m = build_ast_metrics(node, visitor, segment)
            risk_m = build_risk_metrics(visitor)

            opcode_entropy_mean: Optional[float] = None
            mse_area_val: Optional[float] = None
            mse_area_norm: Optional[float] = None
            fractal_val: Optional[float] = None
            mse_profile: Optional[List[Tuple[int, float]]] = None

            code_obj = lookup_code_object(code_maps, lineno, name, node.name)
            if code_obj is not None:
                bytecode = opcodes_from_code(code_obj)
                if len(bytecode) >= MIN_BYTECODE_LENGTH:
                    if self.compute_mse:
                        profile = multi_scale_entropy(bytecode, self.scales, self.m)
                        mse_profile = profile
                        mse_area_val = mse_area(profile)
                        mse_area_norm = mse_area_normalized(profile)
                        opcode_entropy_mean = safe_mean(se for _, se in profile)
                    if self.compute_fractal:
                        fractal_val = fractal_dimension(bytecode)

            maintainability = maintainability_subscore(ast_m)
            anomaly = anomaly_subscore(src_entropy, opcode_entropy_mean, mse_area_norm, fractal_val)
            risk = risk_subscore(risk_m)
            have_anomaly = any(
                value is not None
                for value in (src_entropy, opcode_entropy_mean, mse_area_norm, fractal_val)
            )
            complexity = composite_complexity_score(maintainability, anomaly, risk, have_anomaly)

            result = AuditResult(
                file=str(path),
                name=name,
                lineno=lineno,
                end_lineno=end_lineno,
                src_entropy=src_entropy,
                opcode_entropy_mean=opcode_entropy_mean,
                mse_area=mse_area_val,
                mse_area_normalized=mse_area_norm,
                mse_profile=mse_profile,
                fractal=fractal_val,
                ast_metrics=ast_m,
                risk_metrics=risk_m,
                complexity=complexity,
                maintainability_score=maintainability,
                anomaly_score=anomaly,
                risk_score=risk,
                source_snippet=snippet,
            )
            result.reasons = explain_result(result)
            results.append(result)

        return results


# ------------------------------------------------------------------------------
# CLI and file discovery
# ------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM Code Auditor v4.4.0 – maintainability-first complexity, anomaly, and risk triage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Python file(s) or directory(s)")
    parser.add_argument("--source-entropy", action="store_true", dest="source", help="Compute source Shannon entropy")
    parser.add_argument("--mse", action="store_true", help="Compute multi-scale opcode pattern entropy")
    parser.add_argument("--fractal", action="store_true", help="Compute experimental fractal dimension")
    parser.add_argument("--scales", type=int, nargs="+", default=DEFAULT_SCALES, help=f"MSE scales (default: {DEFAULT_SCALES})")
    parser.add_argument("--m", type=int, default=DEFAULT_M, help="Opcode n-gram embedding length (default: 2)")
    parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    parser.add_argument("--verbose", action="store_true", help="Full columns and extra diagnostics")
    parser.add_argument("--top-source", type=int, default=0, metavar="N", help="Show source of top N functions (0=off)")
    parser.add_argument("--output", type=Path, help="Write results to file")
    parser.add_argument("--explain", action="store_true", help="Show why each high-scoring function was flagged")
    parser.add_argument("--min-score", type=int, default=0, help="Only show functions with complexity >= N")
    parser.add_argument("--sort", choices=list(SortKey.__args__), default="complexity", help="Sort key")
    parser.add_argument("--fail-above", type=int, help="Exit code 2 if any score >= N (CI friendly)")
    parser.add_argument("--exclude", nargs="*", default=[], help="Extra exclude directory names or path fragments")
    parser.add_argument("--include-tests", action="store_true", help="Include test files")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.m <= 0:
        raise SystemExit("--m must be a positive integer")
    if any(scale <= 0 for scale in args.scales):
        raise SystemExit("--scales must contain only positive integers")
    if not (0 <= args.min_score <= 100):
        raise SystemExit("--min-score must be between 0 and 100")
    if args.fail_above is not None and not (0 <= args.fail_above <= 100):
        raise SystemExit("--fail-above must be between 0 and 100")
    if args.top_source < 0:
        raise SystemExit("--top-source must be >= 0")


def find_python_files(
    paths: Sequence[Path],
    exclude: Optional[Iterable[str]] = None,
    include_tests: bool = False,
) -> Iterator[Path]:
    exclude_set = set(DEFAULT_EXCLUDE_DIRS)
    if exclude:
        exclude_set.update(item for item in exclude if item)

    seen: set[Path] = set()

    def excluded(path: Path) -> bool:
        parts = set(path.parts)
        if parts & exclude_set:
            return True
        posix = path.as_posix()
        return any(fragment in posix for fragment in exclude_set if "/" in fragment or fragment.startswith("."))

    for p in paths:
        p = p.expanduser()
        if p.is_file() and p.suffix == ".py":
            if excluded(p) or (not include_tests and is_probably_test_file(p)):
                continue
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield p
        elif p.is_dir():
            for child in p.rglob("*.py"):
                if excluded(child) or (not include_tests and is_probably_test_file(child)):
                    continue
                resolved = child.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield child


# ------------------------------------------------------------------------------
# Output formatting
# ------------------------------------------------------------------------------
def _cell(value: str, width: int, *, truncate: bool = True) -> str:
    if truncate and len(value) > width:
        if width <= 1:
            return value[:width]
        return value[: max(0, width - 1)] + "…"
    if len(value) >= width:
        return value
    return f"{value:<{width}}"


def _path_cell(path: str, width: int) -> str:
    # Keep the compact table readable but avoid chopping the actual filename.
    # For very long paths, show an ellipsized prefix with the full basename.
    if len(path) <= width:
        return _cell(path, width)

    p = Path(path)
    basename = p.name
    if len(basename) >= width:
        return _cell(basename, width)

    remaining = width - len(basename) - 1
    if remaining <= 1:
        return _cell(basename, width)

    prefix = "…" + str(p.parent)[-remaining + 1 :]
    return _cell(f"{prefix}/{basename}", width)


def _fmt_optional(value: Optional[float], width: int, precision: int = 2) -> str:
    return _cell("N/A", width) if value is None else _cell(f"{value:.{precision}f}", width)


def print_table(results: List[AuditResult], verbose: bool, use_color: bool) -> None:
    if not results:
        print("No functions matched the selected filters.")
        return

    has_source = any(r.src_entropy is not None for r in results)
    has_mse = any(r.opcode_entropy_mean is not None or r.mse_area is not None for r in results)
    has_fractal = any(r.fractal is not None for r in results)

    cols: List[Tuple[str, int]] = [
        ("File", 28),
        ("Function", 32),
        ("Line", 6),
        ("LOC", 5),
        ("Cyclo", 6),
        ("Nest", 5),
        ("Risk", 5),
        ("Maint", 6),
        ("Anom", 5),
    ]

    if verbose:
        if has_source:
            cols.append(("SrcEnt", 8))
        if has_mse:
            cols.extend([("OpEnt", 7), ("MSEn", 6)])
        if has_fractal:
            cols.append(("FracD", 7))

    cols.append(("Complex", 8))

    header = " ".join(_cell(title, width) for title, width in cols)
    print(header)
    print("-" * len(header))

    for r in results:
        ast_m = r.ast_metrics
        row_parts = [
            _path_cell(r.file, 28),
            _cell(r.name, 32),
            _cell(str(r.lineno), 6),
            _cell(str(ast_m.loc), 5),
            _cell(str(ast_m.cyclomatic), 6),
            _cell(str(ast_m.max_control_depth), 5),
            _cell(str(r.risk_score), 5),
            _cell(str(r.maintainability_score), 6),
            _cell(str(r.anomaly_score), 5),
        ]

        if verbose:
            if has_source:
                row_parts.append(_fmt_optional(r.src_entropy, 8))
            if has_mse:
                row_parts.append(_fmt_optional(r.opcode_entropy_mean, 7))
                row_parts.append(_fmt_optional(r.mse_area_normalized, 6))
            if has_fractal:
                row_parts.append(_fmt_optional(r.fractal, 7))

        row_parts.append(score_cell(r.complexity, 8, use_color))
        print(" ".join(row_parts))

    if verbose and has_mse:
        print("\n=== Multi-scale Opcode Pattern Entropy Profiles ===")
        for r in results:
            if r.mse_profile:
                print(f"\n{r.file}:{r.name} (line {r.lineno})")
                for scale, entropy in r.mse_profile:
                    print(f"  τ={scale:2d} : {entropy:.4f}")

    if verbose:
        print("\n=== Extra AST/Risk Metrics ===")
        for r in results[: min(20, len(results))]:
            ast_m = r.ast_metrics
            risk_m = r.risk_metrics
            print(
                f"\n{r.file}:{r.name} line {r.lineno}\n"
                f"  AST: args={ast_m.arg_count}, calls={ast_m.calls}, returns={ast_m.returns}, "
                f"assignments={ast_m.assignments}, comprehensions={ast_m.comprehensions}, "
                f"nested_functions={ast_m.nested_functions}, ast_depth={ast_m.max_ast_depth}\n"
                f"  Risk: broad_except={risk_m.broad_excepts}, silent_except={risk_m.silent_excepts}, "
                f"eval_exec={risk_m.eval_exec_calls}, shell_true={risk_m.shell_true}, "
                f"dynamic_attr={risk_m.dynamic_attr}, mutable_defaults={risk_m.mutable_defaults}"
            )


def print_explanations(results: List[AuditResult], limit: int = 20) -> None:
    if not results:
        return

    print("\n=== WHY THESE FUNCTIONS WERE FLAGGED ===")
    for r in results[:limit]:
        print(f"\n{r.file}:{r.name} (line {r.lineno}) – Complexity {r.complexity}")
        for reason in r.reasons:
            print(f"  • {reason}")


def print_summary(results: List[AuditResult], all_results_count: int, top_source: int, use_color: bool) -> None:
    if not results:
        print("\nNo functions matched the selected filters.")
        return

    print("\n" + "=" * 80)
    print(colorize("SUMMARY", BOLD, use_color))

    analysed_files = len({r.file for r in results})
    avg_complexity = statistics.mean(r.complexity for r in results)
    median_complexity = statistics.median(r.complexity for r in results)
    worst = max(results, key=lambda r: r.complexity)

    print(f"• Files shown          : {analysed_files}")
    print(f"• Functions shown      : {len(results)}")
    if all_results_count != len(results):
        print(f"• Functions analysed   : {all_results_count}")
    print(f"• Average complexity   : {avg_complexity:.1f}")
    print(f"• Median complexity    : {median_complexity:.1f}")
    print(f"• Highest complexity   : {worst.file}:{worst.name} → {color_score(worst.complexity, use_color)}")

    top_n = min(top_source or 5, len(results))
    print(f"\n{colorize(f'TOP {top_n} FUNCTIONS NEEDING REVIEW', BOLD, use_color)}")
    for index, r in enumerate(results[:top_n], 1):
        print(
            f"  {index:2d}. {r.file}:{r.name} "
            f"(line {r.lineno}) → {color_score(r.complexity, use_color)} "
            f"[maint={r.maintainability_score}, anom={r.anomaly_score}, risk={r.risk_score}]"
        )

    if top_source > 0:
        print(f"\n{colorize(f'SOURCE CODE OF TOP {top_source} FUNCTIONS', BOLD, use_color)}")
        for r in results[:top_source]:
            print(
                f"\n{colorize(f'=== {r.file}:{r.name} (line {r.lineno}) – Complexity {r.complexity} ===', BOLD, use_color)}"
            )
            print(r.source_snippet)
            print("-" * 60)


def render_table_output(
    results: List[AuditResult],
    all_results_count: int,
    verbose: bool,
    top_source: int,
    explain: bool,
    use_color: bool,
) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_table(results, verbose=verbose, use_color=use_color)
        if explain:
            print_explanations(results)
        print_summary(results, all_results_count=all_results_count, top_source=top_source, use_color=use_color)
    return buf.getvalue()


def render_json_output(results: List[AuditResult]) -> str:
    return json.dumps([r.to_json_dict() for r in results], indent=2, sort_keys=False)


# ------------------------------------------------------------------------------
# Sorting and main
# ------------------------------------------------------------------------------
def _sort_value(result: AuditResult, sort_key: SortKey) -> float:
    if sort_key == "complexity":
        return result.complexity
    if sort_key == "risk":
        return result.risk_score
    if sort_key == "maintainability":
        return result.maintainability_score
    if sort_key == "anomaly":
        return result.anomaly_score
    if sort_key == "loc":
        return result.ast_metrics.loc
    if sort_key == "cyclomatic":
        return result.ast_metrics.cyclomatic
    if sort_key == "mse":
        return result.opcode_entropy_mean if result.opcode_entropy_mean is not None else -1
    if sort_key == "fractal":
        return result.fractal if result.fractal is not None else -1
    if sort_key == "source":
        return result.src_entropy if result.src_entropy is not None else -1
    return result.complexity


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    # Preserve previous behavior: if no optional statistical metric is selected,
    # compute all of them.
    if not (args.source or args.mse or args.fractal):
        args.source = args.mse = args.fractal = True

    use_color = should_use_color(args.no_color, args.output)

    auditor = CodeAuditor(
        scales=args.scales,
        m=args.m,
        compute_source=args.source,
        compute_mse=args.mse,
        compute_fractal=args.fractal,
    )

    all_results: List[AuditResult] = []
    for py_file in find_python_files(args.paths, exclude=args.exclude, include_tests=args.include_tests):
        try:
            all_results.extend(auditor.analyse_file(py_file))
        except Exception as exc:
            print(f"Skipping {py_file}: {exc}", file=sys.stderr)

    if not all_results:
        print("No Python functions found.", file=sys.stderr)
        sys.exit(1)

    all_results.sort(key=lambda r: _sort_value(r, args.sort), reverse=True)
    filtered_results = [r for r in all_results if r.complexity >= args.min_score]

    if args.format == "json":
        output_text = render_json_output(filtered_results)
    else:
        output_text = render_table_output(
            filtered_results,
            all_results_count=len(all_results),
            verbose=args.verbose,
            top_source=args.top_source,
            explain=args.explain,
            use_color=use_color,
        )

    if args.output:
        args.output.write_text(output_text, encoding="utf-8")
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output_text)

    if args.fail_above is not None:
        worst = max(r.complexity for r in all_results)
        if worst >= args.fail_above:
            print(
                f"Complexity gate failed: worst score {worst} >= --fail-above {args.fail_above}",
                file=sys.stderr,
            )
            sys.exit(2)


if __name__ == "__main__":
    main()

