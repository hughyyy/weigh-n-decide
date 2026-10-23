"""Differential tests for the condition evaluator.

Seeded random expressions over a fact universe are evaluated twice: by the
implementation and by an independent reference interpreter written from the
documented semantics (module docstring + README). Both must agree on the
result truthiness and on when a ConditionError is raised. Deterministic:
the seed is fixed, so failures reproduce exactly. Stdlib only.
"""

import random
import unittest

from weigh_n_decide.conditions import ConditionError, evaluate_conditions

SEED = 0xC0FFEE

FACTS = {
    "amount": 5,
    "balance": 2.5,
    "zero": 0,
    "neg": -2.5,
    "note": "paid",
    "other": "x",
    "empty": "",
    "flag": True,
    "off": False,
    "nil": None,
    "tags": ["a", "b"],
    "numlist": [1, 2, 3],
    "nested": {"inner": 7},
    "obj": {"k": 1},
    "nestedlist": [[1], [2]],
}

NUM_ATOMS = ["amount", "balance", "neg", "zero", "numlist[0]", "numlist[-1]",
             "nested.inner", "0", "7", "1.5", "2.5"]
STR_ATOMS = ["note", "other", "empty", "'paid'", "'x'", "'alpha'", "'paidx'"]
ANY_ATOMS = NUM_ATOMS + STR_ATOMS + ["flag", "off", "nil", "tags[0]", "tags[1]",
                                    "obj", "nestedlist", "[1, 'a']", "['a', 'b']",
                                    "[]", "[1.5]", "[1, [2]]", "[true]", "[null]",
                                    "[[1]]"]
TRAP_ATOMS = ["missing", "nested.nope", "amount[0]", "numlist[9]", "numlist[-9]",
              "tags[0][0]", "nil.inner", "zero[-1]", "obj[0]", "neg.missing",
              "nestedlist[5]"]
LIST_RIGHTS = ["['a', 'b']", "[1, 2, 3]", "[1.5, 2.5]", "[1, 'x']", "[]",
               "['paid', 'x']", "[5]", "[null]", "[0]", "[1, [2]]", "tags",
               "nestedlist", "[true]"]
LIST_LEFTS = ["tags", "numlist", "nestedlist", "['a', 'b']", "[1, 2]", "[1.5]",
              "[]", "[1, [2]]"]


class RefError(Exception):
    """The reference interpreter's analogue of ConditionError."""


def _is_num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def ref_json_equal(a, b):
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(ref_json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(ref_json_equal(a[k], b[k]) for k in a)
    if _is_num(a) and _is_num(b):
        return a == b
    if type(a) is not type(b) and not (a is None and b is None):
        return False
    return a == b


def ref_resolve(facts, steps):
    current = facts
    for kind, step in steps:
        if kind == "key":
            if not isinstance(current, dict) or step not in current:
                raise RefError
            current = current[step]
        else:
            if not isinstance(current, list) or not -len(current) <= step < len(current):
                raise RefError
            current = current[step]
    return current


def ref_compare(op, left, right):
    if op in ("<", "<=", ">", ">="):
        comparable = (_is_num(left) and _is_num(right)) or (
            isinstance(left, str) and isinstance(right, str)
        )
        if not comparable:
            raise RefError
        return {"<": left < right, "<=": left <= right,
                ">": left > right, ">=": left >= right}[op]
    if op == "==":
        return ref_json_equal(left, right)
    if op == "!=":
        return not ref_json_equal(left, right)
    if op in ("in", "not in"):
        if isinstance(right, list):
            result = any(ref_json_equal(left, item) for item in right)
        elif isinstance(right, str):
            if not isinstance(left, str):
                raise RefError
            result = left in right
        else:
            raise RefError
        return result if op == "in" else not result
    if op == "contains":
        if isinstance(left, list):
            return any(ref_json_equal(right, item) for item in left)
        if isinstance(left, str):
            if not isinstance(right, str):
                raise RefError
            return right in left
        raise RefError
    raise RefError


def ref_eval(node, facts):
    kind = node["kind"]
    if kind == "lit":
        return node["value"]
    if kind == "path":
        return ref_resolve(facts, node["steps"])
    if kind == "truthy":
        return bool(ref_eval(node["operand"], facts))
    if kind == "group":
        return ref_eval(node["part"], facts)
    if kind == "not":
        return not bool(ref_eval(node["part"], facts))
    if kind in ("and", "or"):
        for part in node["parts"]:
            value = ref_eval(part, facts)
            if kind == "and" and not bool(value):
                return False
            if kind == "or" and bool(value):
                return True
        return kind == "and"
    if kind == "cmp":
        return ref_compare(
            node["op"], ref_eval(node["left"], facts), ref_eval(node["right"], facts)
        )
    raise RefError


# --------------------------------------------------------------------------
# Expression generator

def _atom(rng, trap_prob):
    if rng.random() < trap_prob:
        return rng.choice(TRAP_ATOMS)
    return rng.choice(ANY_ATOMS)


def _ord_expr(rng, mismatch_prob):
    op = rng.choice(["<", "<=", ">", ">="])
    if rng.random() < 1 - mismatch_prob:
        family = rng.choice(["num", "str"])
        if family == "num":
            left, right = rng.choice(NUM_ATOMS), rng.choice(NUM_ATOMS)
        else:
            left, right = rng.choice(STR_ATOMS), rng.choice(STR_ATOMS)
    else:
        left, right = rng.choice(NUM_ATOMS), rng.choice(STR_ATOMS)
    return f"{left} {op} {right}"


def _eq_expr(rng):
    return f"{_atom(rng, 0.05)} {rng.choice(['==', '!='])} {_atom(rng, 0.05)}"


def _in_expr(rng, mismatch_prob):
    op = rng.choice(["in", "not in"])
    roll = rng.random()
    if roll < 0.5 - mismatch_prob * 0.5:  # list right, any left
        return f"{_atom(rng, 0.05)} {op} {rng.choice(LIST_RIGHTS)}"
    if roll < 0.8 - mismatch_prob * 0.3:  # string right, string left
        return f"{rng.choice(STR_ATOMS)} {op} {rng.choice(STR_ATOMS)}"
    if rng.random() < 0.5:  # mismatch: number left over string, or any over number
        return f"{rng.choice(NUM_ATOMS)} {op} {rng.choice(STR_ATOMS)}"
    return f"{_atom(rng, 0.05)} {op} {rng.choice(NUM_ATOMS)}"


def _contains_expr(rng, mismatch_prob):
    roll = rng.random()
    if roll < 0.5 - mismatch_prob * 0.3:  # list left, any right
        return f"{rng.choice(LIST_LEFTS)} contains {_atom(rng, 0.05)}"
    if roll < 0.8 - mismatch_prob * 0.1:  # string left, string right
        return f"{rng.choice(STR_ATOMS)} contains {rng.choice(STR_ATOMS)}"
    if rng.random() < 0.5:  # mismatch: number left
        return f"{rng.choice(NUM_ATOMS)} contains {_atom(rng, 0.05)}"
    return f"{rng.choice(STR_ATOMS)} contains {rng.choice(NUM_ATOMS)}"


def build_expr(rng, depth, mismatch_prob):
    if depth <= 0 or rng.random() < 0.4:
        return _atom(rng, 0.1)
    roll = rng.random()
    if roll < 0.30:
        op = rng.choice(["and", "or"])
        return f"({build_expr(rng, depth - 1, mismatch_prob)} {op} {build_expr(rng, depth - 1, mismatch_prob)})"
    if roll < 0.40:
        return f"not ({build_expr(rng, depth - 1, mismatch_prob)})"
    if roll < 0.60:
        return _ord_expr(rng, mismatch_prob)
    if roll < 0.75:
        return _eq_expr(rng)
    if roll < 0.90:
        return _in_expr(rng, mismatch_prob)
    return _contains_expr(rng, mismatch_prob)


class DifferentialTests(unittest.TestCase):
    @staticmethod
    def check(expr):
        from weigh_n_decide.conditions import parse

        node = parse(expr)
        try:
            result = evaluate_conditions(expr, FACTS)["conditions"][0]["result"]
            impl_raised = False
        except ConditionError:
            impl_raised = True
            result = None
        try:
            ref = ref_eval(node, FACTS)
            ref_raised = False
        except RefError:
            ref_raised = True
            ref = None
        if impl_raised != ref_raised:
            return f"error disagreement: impl_raised={impl_raised} ref_raised={ref_raised}"
        if not impl_raised and bool(ref) != result:
            return f"value disagreement: impl={result} ref={bool(ref)}"
        return None

    def _sweep(self, cases, mismatch_prob):
        rng = random.Random(SEED)
        failures = []
        for _ in range(cases):
            expr = build_expr(rng, 3, mismatch_prob)
            problem = self.check(expr)
            if problem:
                failures.append(f"{expr!r}: {problem}")
        self.assertEqual(failures, [])

    def test_type_valid_expressions_agree(self):
        self._sweep(500, mismatch_prob=0.15)

    def test_mismatch_heavy_expressions_agree(self):
        self._sweep(400, mismatch_prob=0.6)

    def test_deeper_expressions_agree(self):
        rng = random.Random(SEED ^ 0xDEE9)
        failures = []
        for _ in range(150):
            expr = build_expr(rng, 4, mismatch_prob=0.2)
            problem = self.check(expr)
            if problem:
                failures.append(f"{expr!r}: {problem}")
        self.assertEqual(failures, [])

    def test_error_only_sweep(self):
        # Every deliberately broken comparison must raise in both.
        rng = random.Random(SEED ^ 0x5EED)
        for _ in range(200):
            expr = build_expr(rng, 2, mismatch_prob=1.0)
            problem = self.check(expr)
            if problem:
                self.fail(f"{expr!r}: {problem}")


class ComparisonMatrixTests(unittest.TestCase):
    """Every JSON type pair crossed with every operator agrees with the ref."""

    VALUES = {
        "boolean": (True, False),
        "integer": (7, 3),
        "float": (2.5, 3.5),
        "string": ("a", "b"),
        "null": (None, None),
        "list": ([1], ["a"]),
        "object": ({"k": 1}, {"k": 2}),
    }
    OPS = ["==", "!=", "<", "<=", ">", ">=", "in", "not in", "contains"]

    def test_every_type_pair_and_operator_agrees(self):
        import itertools

        for (lk, (left, _)), (rk, (_, right)) in itertools.product(
            self.VALUES.items(), repeat=2
        ):
            facts = {"l": left, "r": right}
            for op in self.OPS:
                expr = f"l {op} r"
                with self.subTest(left=lk, right=rk, op=op):
                    try:
                        result = evaluate_conditions(expr, facts)["conditions"][0]["result"]
                        raised = False
                    except ConditionError:
                        raised = True
                    try:
                        expected = ref_compare(op, left, right)
                        ref_raised = False
                    except RefError:
                        ref_raised = True
                    self.assertEqual(raised, ref_raised, expr)
                    if not raised:
                        self.assertEqual(result, bool(expected), expr)


class FactPathDifferentialTests(unittest.TestCase):
    """Generated paths — dotted, indexed, negative, nested — resolve correctly."""

    FACTS = {
        "a": {"b": [1, {"c": "x"}, None], "n": -2.5},
        "tags": ["a", "b"],
        "flag": True,
        "obj": {"k": None},
        "empty": [],
    }

    @staticmethod
    def collect(prefix, value, out):
        out.append((prefix, value))
        if isinstance(value, dict):
            for key, child in value.items():
                FactPathDifferentialTests.collect(f"{prefix}.{key}", child, out)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                FactPathDifferentialTests.collect(f"{prefix}[{index}]", child, out)
                FactPathDifferentialTests.collect(
                    f"{prefix}[{index - len(value)}]", child, out
                )

    def test_generated_paths_resolve_and_compare_to_literals(self):
        import json

        cases = []
        for key, value in self.FACTS.items():
            self.collect(key, value, cases)
        self.assertGreaterEqual(len(cases), 20)
        for path, value in cases:
            with self.subTest(path=path):
                truthy = evaluate_conditions(path, self.FACTS)["conditions"][0]["result"]
                self.assertIs(truthy, bool(value))
                if value is None or isinstance(value, (bool, int, float, str)):
                    literal = json.dumps(value)
                    out = evaluate_conditions(f"{path} == {literal}", self.FACTS)
                    self.assertIs(out["conditions"][0]["result"], True)


class TraceInvariantTests(unittest.TestCase):
    """Every trace fragment re-parses and evaluates to its recorded value."""

    def test_trace_fragments_are_self_consistent(self):
        import random

        rng = random.Random(SEED ^ 0x7ACE)
        checked = 0
        for _ in range(150):
            expr = build_expr(rng, 3, mismatch_prob=0.15)
            try:
                entry = evaluate_conditions(expr, FACTS)["conditions"][0]
            except ConditionError:
                continue
            self.check_trace(entry["trace"], FACTS)
            checked += 1
        self.assertGreater(checked, 50)

    def check_trace(self, trace, facts):
        fragment = trace["expr"]
        self.assertIsInstance(fragment, str)
        self.assertTrue(fragment)
        reparse = evaluate_conditions(fragment, facts)["conditions"][0]
        self.assertEqual(reparse["result"], bool(trace["value"]), fragment)
        for key in ("child", "of", "left", "right"):
            if key in trace:
                self.check_trace(trace[key], facts)
        for child in trace.get("children", []):
            self.check_trace(child, facts)


if __name__ == "__main__":
    unittest.main()
