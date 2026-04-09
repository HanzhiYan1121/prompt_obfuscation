"""
L4 AirSim Drone Code Evaluator
================================
Three independent sub-evaluators:
  1. FormatChecker       – structural / syntactic requirements
  2. APIAdherenceChecker – only allowed aw.* calls, no hallucinations
  3. TrajectoryEvaluator – simulate code; compare state-changes to ground truth

Usage:
    ev = DroneCodeEvaluator()
    result = ev.evaluate(code_str, expected_states)
    # result["format_score"], result["api_score"], result["trajectory_score"]
"""

import ast
import math
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any

# ── Allowed API surface ───────────────────────────────────────────────────────
ALLOWED_AW_METHODS = {
    "takeoff", "land", "go_home",
    "fly_to", "get_yaw", "set_yaw",
    "get_drone_position", "get_position",
}

ALLOWED_IMPORTS = {"math"}

FORBIDDEN_KEYWORDS = {"for", "while"}   # loops forbidden by system prompt
# 'if' is also forbidden; we detect it separately to avoid false positives
# on words like "elif" inside strings

# ─────────────────────────────────────────────────────────────────────────────
# 1. Format Checker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FormatResult:
    score: float                        # 0.0–1.0
    no_markdown_blocks: bool = True     # no ```python or ``` wrappers
    no_prose_lines: bool = True         # no lines of plain English text
    no_forbidden_loops: bool = True     # no for / while
    no_if_statements: bool = True       # no if / elif / else
    no_function_defs: bool = True       # no def statements
    only_allowed_imports: bool = True   # only import math
    parseable: bool = True              # valid Python syntax
    violations: list = field(default_factory=list)


class FormatChecker:
    """
    Rule weights (equal for simplicity, adjust as desired):
      - no_markdown_blocks    : 20%
      - no_prose_lines        : 15%
      - no_forbidden_loops    : 20%
      - no_if_statements      : 15%
      - no_function_defs      : 10%
      - only_allowed_imports  : 10%
      - parseable             : 10%
    """
    WEIGHTS = {
        "no_markdown_blocks":   0.20,
        "no_prose_lines":       0.15,
        "no_forbidden_loops":   0.20,
        "no_if_statements":     0.15,
        "no_function_defs":     0.10,
        "only_allowed_imports": 0.10,
        "parseable":            0.10,
    }

    def check(self, code: str) -> FormatResult:
        result = FormatResult(score=0.0)
        violations = []

        # ── Markdown fences ──────────────────────────────────────────────────
        if re.search(r"```", code):
            result.no_markdown_blocks = False
            violations.append("Contains markdown code fences (``` or ```python)")
            # Strip them for subsequent checks to be fair
            code = re.sub(r"```(?:python)?", "", code)

        # ── Parse AST ────────────────────────────────────────────────────────
        try:
            tree = ast.parse(textwrap.dedent(code))
        except SyntaxError as e:
            result.parseable = False
            violations.append(f"SyntaxError: {e}")
            # Can't do AST-based checks
            result.score = self._score(result)
            result.violations = violations
            return result

        # ── Prose lines (lines that are not Python statements) ────────────────
        # Heuristic: non-empty lines that start with a capital letter and contain
        # no Python punctuation (=, (, ), [, ], .) are likely prose.
        code_lines = code.splitlines()
        prose_lines = []
        for ln in code_lines:
            stripped = ln.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Likely prose: starts capital, no assignment/call markers
            if (re.match(r"^[A-Z][a-zA-Z\s,\.]+$", stripped) and
                    not any(c in stripped for c in "=()[]{}.:")):
                prose_lines.append(stripped)
        if prose_lines:
            result.no_prose_lines = False
            violations.append(f"Prose / explanation lines detected: {prose_lines[:3]}")

        # ── AST-based checks ─────────────────────────────────────────────────
        for node in ast.walk(tree):

            # Loops
            if isinstance(node, (ast.For, ast.While)):
                result.no_forbidden_loops = False
                violations.append(f"Forbidden loop ({type(node).__name__}) at line {node.lineno}")

            # if / elif / else
            if isinstance(node, ast.If):
                result.no_if_statements = False
                violations.append(f"Forbidden if-statement at line {node.lineno}")

            # Function definitions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result.no_function_defs = False
                violations.append(f"Function definition '{node.name}' at line {node.lineno}")

            # Imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in ALLOWED_IMPORTS:
                        result.only_allowed_imports = False
                        violations.append(f"Forbidden import: {alias.name}")
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod.split(".")[0] not in ALLOWED_IMPORTS:
                    result.only_allowed_imports = False
                    violations.append(f"Forbidden from-import: {mod}")

        result.score = self._score(result)
        result.violations = violations
        return result

    def _score(self, r: FormatResult) -> float:
        s = 0.0
        for attr, w in self.WEIGHTS.items():
            if getattr(r, attr, True):
                s += w
        return round(s, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 2. API Adherence Checker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class APIResult:
    score: float
    total_calls: int = 0
    valid_calls: int = 0
    hallucinated: list = field(default_factory=list)   # aw.xxx not in allowed set
    forbidden_objects: list = field(default_factory=list)  # non-aw API objects used
    violations: list = field(default_factory=list)


class APIAdherenceChecker:
    """
    Walks the AST and collects every function call.
    • aw.<method>()  → must be in ALLOWED_AW_METHODS
    • Any other module-level calls beyond math.* → flagged
    """

    def check(self, code: str) -> APIResult:
        result = APIResult(score=0.0)
        try:
            tree = ast.parse(textwrap.dedent(code))
        except SyntaxError:
            result.violations.append("Cannot parse code; API check skipped.")
            return result

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            func = node.func

            # aw.<method>() form
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                obj = func.value.id
                method = func.attr

                if obj == "aw":
                    result.total_calls += 1
                    if method in ALLOWED_AW_METHODS:
                        result.valid_calls += 1
                    else:
                        result.hallucinated.append(f"aw.{method}()")
                        result.violations.append(
                            f"API hallucination: aw.{method}() is not in the allowed list"
                        )
                elif obj == "math":
                    pass  # math.* is always allowed
                else:
                    result.forbidden_objects.append(f"{obj}.{method}()")
                    result.violations.append(
                        f"Call on non-aw object: {obj}.{method}()"
                    )

        if result.total_calls == 0:
            # No aw calls at all — probably missing takeoff etc.
            result.violations.append("No aw.* API calls detected")
            result.score = 0.0
        else:
            result.score = round(result.valid_calls / result.total_calls, 4)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Trajectory Simulator + Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class _MockAW:
    """
    Minimal AirSim mock.  Records a state_changes list: each entry is
    [Δx, Δy, Δz, Δyaw] in NED world coordinates (matches numerical_evaluator.txt).
    """
    def __init__(self):
        self.pos   = [0.0, 0.0, 0.0]   # [x, y, z] NED
        self.yaw   = 0.0               # degrees; 0 = North
        self.state_changes: list[list[float]] = []
        self.call_log: list[str] = []

    # ── API surface ──────────────────────────────────────────────────────────
    def takeoff(self):
        self.call_log.append("takeoff()")

    def land(self):
        self.call_log.append("land()")

    def go_home(self):
        self.call_log.append("go_home()")
        old = self.pos[:]
        self.pos = [0.0, 0.0, 0.0]
        self._record_move(old, self.pos, 0.0)

    def fly_to(self, xyz):
        old = self.pos[:]
        self.pos = [float(v) for v in xyz]
        self._record_move(old, self.pos, 0.0)
        self.call_log.append(f"fly_to({[round(v,3) for v in xyz]})")

    def get_yaw(self):
        return self.yaw

    def set_yaw(self, yaw):
        old_yaw = self.yaw
        self.yaw = float(yaw)
        delta_yaw = self.yaw - old_yaw
        # Normalize to (-180, 180]
        while delta_yaw >  180: delta_yaw -= 360
        while delta_yaw <= -180: delta_yaw += 360
        self.state_changes.append([0.0, 0.0, 0.0, round(delta_yaw, 2)])
        self.call_log.append(f"set_yaw({round(yaw,2)})")

    def get_drone_position(self):
        return self.pos[:]

    def get_position(self, name: str):
        return [0.0, 0.0, 0.0]   # unknown objects default to origin

    # ── internal ─────────────────────────────────────────────────────────────
    def _record_move(self, old, new, dyaw):
        dx = round(new[0] - old[0], 4)
        dy = round(new[1] - old[1], 4)
        dz = round(new[2] - old[2], 4)
        if dx != 0 or dy != 0 or dz != 0 or dyaw != 0:
            self.state_changes.append([dx, dy, dz, dyaw])


@dataclass
class TrajectoryResult:
    score: float
    executed: bool = False
    exec_error: str = ""
    simulated_states: list = field(default_factory=list)
    expected_states:  list = field(default_factory=list)
    step_scores: list = field(default_factory=list)   # per-step match (0 or 1)
    violations: list = field(default_factory=list)


class TrajectoryEvaluator:
    """
    Executes the generated code in a sandboxed mock environment and
    compares the resulting state-change sequence to the ground-truth
    using the rules from numerical_evaluator.txt:
      - Each step is [Δx, Δy, Δz, Δyaw]
      - Tolerance ≤ 1 unit for numeric values
      - Zero-change entries (no movement) are collapsed / ignored
    """
    TOLERANCE = 1.0

    def evaluate(self, code: str, expected_states: list[list[float]]) -> TrajectoryResult:
        result = TrajectoryResult(score=0.0, expected_states=expected_states)

        # ── Strip markdown fences ──────────────────────────────────────────
        clean = re.sub(r"```(?:python)?|```", "", code).strip()

        # ── Execute in mock namespace ──────────────────────────────────────
        aw = _MockAW()
        namespace: dict[str, Any] = {"aw": aw, "math": math}
        try:
            exec(compile(clean, "<drone_code>", "exec"), namespace)
            result.executed = True
        except Exception as e:
            result.exec_error = str(e)
            result.violations.append(f"Runtime error: {e}")
            return result

        sim = aw.state_changes
        result.simulated_states = sim

        if not expected_states:
            # No ground truth provided; can't score trajectory
            result.score = None
            return result

        # ── Step-by-step comparison ────────────────────────────────────────
        n_exp = len(expected_states)
        n_sim = len(sim)
        step_scores = []

        for i, exp in enumerate(expected_states):
            if i >= n_sim:
                step_scores.append(0)
                result.violations.append(f"Step {i+1}: missing (expected {exp})")
                continue
            got = sim[i]
            if self._match(exp, got):
                step_scores.append(1)
            else:
                step_scores.append(0)
                result.violations.append(
                    f"Step {i+1}: expected {exp}, got {[round(v,2) for v in got]}"
                )

        # Extra steps in sim beyond expected are penalised
        extra = n_sim - n_exp
        if extra > 0:
            result.violations.append(
                f"{extra} extra step(s) in output beyond expected trajectory"
            )

        result.step_scores = step_scores
        result.score = round(sum(step_scores) / max(n_exp, 1), 4)
        return result

    def _match(self, exp: list[float], got: list[float]) -> bool:
        """True if every element of got is within TOLERANCE of exp."""
        for e, g in zip(exp, got):
            if abs(e - g) > self.TOLERANCE:
                return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Top-level Evaluator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    format_score:     float
    api_score:        float
    trajectory_score: float | None    # None if no ground truth provided
    composite_score:  float | None    # weighted average (if trajectory available)

    format_detail:     FormatResult     = None
    api_detail:        APIResult        = None
    trajectory_detail: TrajectoryResult = None

    def summary(self) -> str:
        lines = [
            f"  Format      : {self.format_score:.3f}",
            f"  API         : {self.api_score:.3f}",
            f"  Trajectory  : {self.trajectory_score if self.trajectory_score is not None else 'N/A'}",
            f"  Composite   : {self.composite_score if self.composite_score is not None else 'N/A'}",
        ]
        return "\n".join(lines)


class DroneCodeEvaluator:
    """
    Unified entry point.

    Parameters
    ----------
    code : str
        Raw model output (may contain markdown fences, prose, etc.)
    expected_states : list[list[float]] | None
        Ground-truth [Δx, Δy, Δz, Δyaw] sequence from numerical_evaluator.txt
        rules.  If None, trajectory_score is skipped.
    weights : dict
        Weights for composite score {"format": w, "api": w, "trajectory": w}
    """

    DEFAULT_WEIGHTS = {"format": 0.30, "api": 0.35, "trajectory": 0.35}

    def __init__(self, weights: dict | None = None):
        self._fmt = FormatChecker()
        self._api = APIAdherenceChecker()
        self._trj = TrajectoryEvaluator()
        self._weights = weights or self.DEFAULT_WEIGHTS

    def evaluate(
        self,
        code: str,
        expected_states: list[list[float]] | None = None,
    ) -> EvalResult:
        fmt = self._fmt.check(code)
        api = self._api.check(code)
        trj = self._trj.evaluate(code, expected_states or [])

        # Composite
        if expected_states:
            w = self._weights
            composite = round(
                w["format"] * fmt.score
                + w["api"]  * api.score
                + w["trajectory"] * trj.score,
                4,
            )
            trj_score = trj.score
        else:
            composite = round(
                (self._weights["format"] * fmt.score + self._weights["api"] * api.score)
                / (self._weights["format"] + self._weights["api"]),
                4,
            )
            trj_score = None
            composite = None if trj_score is None else composite

        return EvalResult(
            format_score=fmt.score,
            api_score=api.score,
            trajectory_score=trj_score,
            composite_score=composite,
            format_detail=fmt,
            api_detail=api,
            trajectory_detail=trj,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Example 1: correct code ───────────────────────────────────────────────
    good_code = """
import math
aw.takeoff()
current_position = aw.get_drone_position()
aw.fly_to([current_position[0], current_position[1], current_position[2] - 10])
"""
    # Expected: takeoff (no state change recorded), then fly 10m up (Δz = -10)
    expected = [[-0, 0, -10, 0]]

    ev = DroneCodeEvaluator()
    r = ev.evaluate(good_code, expected)
    print("=== Example 1: correct code ===")
    print(r.summary())
    print("Format violations:", r.format_detail.violations)
    print("API violations:   ", r.api_detail.violations)
    print("Traj violations:  ", r.trajectory_detail.violations)
    print()

    # ── Example 2: hallucinated API + loop + markdown fence ──────────────────
    bad_code = """
```python
import math
import numpy as np
aw.takeoff()
for i in range(4):
    aw.fly_to([i*2, 0, -5])
aw.hover()   # hallucinated
```
"""
    r2 = ev.evaluate(bad_code, expected)
    print("=== Example 2: bad code ===")
    print(r2.summary())
    print("Format violations:", r2.format_detail.violations)
    print("API violations:   ", r2.api_detail.violations)
    print("Traj violations:  ", r2.trajectory_detail.violations)
    print()

    # ── Example 3: 90-degree turn + body-frame left movement ─────────────────
    # Query: "Turn 90 degrees left, fly 5m to the drone's left."
    # Expected after turn: yaw = -90 (CCW); then fly body-left in world coords
    # body_y = -5, yaw = -90 deg → theta = -90
    # world_x = 0*cos(90) + (-5)*sin(90) = -5
    # world_y = -0*sin(90) + (-5)*cos(90) = 0
    # State changes: [0,0,0,-90], [-5,0,0,0]
    turn_code = """
import math
aw.takeoff()
current_yaw = aw.get_yaw()
new_yaw = current_yaw - 90
aw.set_yaw(new_yaw)
current_position = aw.get_drone_position()
body_x = 0
body_y = -5
body_z = 0
current_yaw = aw.get_yaw()
radian_yaw = math.radians(-current_yaw)
world_x = current_position[0] + body_x * math.cos(radian_yaw) + body_y * math.sin(radian_yaw)
world_y = current_position[1] - body_x * math.sin(radian_yaw) + body_y * math.cos(radian_yaw)
world_z = current_position[2] - body_z
aw.fly_to([world_x, world_y, world_z])
"""
    expected3 = [[0, 0, 0, -90], [-5, 0, 0, 0]]
    r3 = ev.evaluate(turn_code, expected3)
    print("=== Example 3: turn + body-frame move ===")
    print(r3.summary())
    print("Simulated states:", r3.trajectory_detail.simulated_states)
    print("Traj violations: ", r3.trajectory_detail.violations)
