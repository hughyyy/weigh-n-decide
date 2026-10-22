"""Condition evaluator tests: grammar, precedence, traces, fact paths, errors."""

import unittest

from weigh_n_decide.conditions import ConditionError, evaluate_conditions


def ev(expr, facts=None):
    return evaluate_conditions(expr, facts)


class GrammarTests(unittest.TestCase):
    def test_boolean_literals(self):
        self.assertIs(ev("true")["conditions"][0]["result"], True)
        self.assertIs(ev("false")["conditions"][0]["result"], False)

    def test_precedence_and_binds_tighter_than_or(self):
        self.assertIs(ev("true or false and false")["conditions"][0]["result"], True)
        self.assertIs(ev("false and false or true")["conditions"][0]["result"], True)

    def test_not_binds_tighter_than_and(self):
        self.assertIs(ev("not false and true")["conditions"][0]["result"], True)
        self.assertIs(ev("not (true and true)")["conditions"][0]["result"], False)

    def test_parens_group(self):
        self.assertIs(ev("(true or false) and false")["conditions"][0]["result"], False)

    def test_not_in_is_not_a_boolean_not(self):
        facts = {"tags": ["a", "b"]}
        self.assertIs(ev("'c' not in tags", facts)["conditions"][0]["result"], True)
        self.assertIs(ev("'a' not in tags", facts)["conditions"][0]["result"], False)

    def test_numbers_and_floats(self):
        facts = {"price": 9.99}
        self.assertIs(ev("price >= 9.5", facts)["conditions"][0]["result"], True)
        self.assertIs(ev("price > 10", facts)["conditions"][0]["result"], False)
        self.assertIs(ev("-1 < 0")["conditions"][0]["result"], True)

    def test_string_literals_and_escapes(self):
        facts = {"status": "paid"}
        self.assertIs(ev('status == "paid"', facts)["conditions"][0]["result"], True)
        self.assertIs(ev("status == 'paid'", facts)["conditions"][0]["result"], True)
        self.assertIs(
            ev('"line\\nbreak" == "line\nbreak"')["conditions"][0]["result"], True
        )
        self.assertIs(
            ev('"caf\\u00e9" == "café"')["conditions"][0]["result"], True
        )


class FactPathTests(unittest.TestCase):
    FACTS = {
        "a": {"total": 1250, "status": "overdue", "vip": False},
        "items": [{"price": 5}, {"price": 7}],
        "user": {"roles": ["admin", "billing"]},
    }

    def test_dotted_and_indexed_paths(self):
        self.assertIs(ev("a.total > 1000", self.FACTS)["conditions"][0]["result"], True)
        self.assertIs(
            ev("items[1].price == 7", self.FACTS)["conditions"][0]["result"], True
        )
        self.assertIs(
            ev("user.roles[0] == 'admin'", self.FACTS)["conditions"][0]["result"], True
        )

    def test_negative_index(self):
        self.assertIs(
            ev("user.roles[-1] == 'billing'", self.FACTS)["conditions"][0]["result"], True
        )
    def test_missing_top_level_path_is_an_error(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("missing.thing")
        self.assertIn("missing key 'missing'", str(ctx.exception))

    def test_truthiness_of_existing_values(self):
        self.assertIs(ev("a.status", self.FACTS)["conditions"][0]["result"], True)

    def test_missing_key_is_an_error(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("a.refund", self.FACTS)
        self.assertIn("missing key 'refund'", str(ctx.exception))
        self.assertIn("a.refund", str(ctx.exception))

    def test_index_out_of_range_is_an_error(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("items[5].price", self.FACTS)
        self.assertIn("out of range", str(ctx.exception))

    def test_indexing_non_list_is_an_error(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("a.total[0]", self.FACTS)
        self.assertIn("is not a list", str(ctx.exception))

    def test_key_into_non_object_is_an_error(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("items.price", self.FACTS)
        self.assertIn("is not an object", str(ctx.exception))


class ComparisonTests(unittest.TestCase):
    def test_eq_ne(self):
        self.assertIs(ev("5 == 5.0")["conditions"][0]["result"], True)
        self.assertIs(ev("'a' != 'b'")["conditions"][0]["result"], True)

    def test_bool_is_not_a_number(self):
        # JSON semantics: true is not 1.
        self.assertIs(ev("flag == 1", {"flag": True})["conditions"][0]["result"], False)
        self.assertIs(ev("flag == true", {"flag": True})["conditions"][0]["result"], True)
        self.assertIs(ev("zero == false", {"zero": 0})["conditions"][0]["result"], False)

    def test_null_handling(self):
        self.assertIs(ev("x == null", {"x": None})["conditions"][0]["result"], True)
        self.assertIs(ev("x == null", {"x": 0})["conditions"][0]["result"], False)

    def test_in_list_uses_json_equality(self):
        self.assertIs(ev("'a' in ['a', 'b']")["conditions"][0]["result"], True)
        self.assertIs(ev("true in [1, 2]")["conditions"][0]["result"], False)
        self.assertIs(ev("'x' not in ['a']")["conditions"][0]["result"], True)

    def test_in_string_is_substring(self):
        self.assertIs(ev("'aid' in 'paid'")["conditions"][0]["result"], True)
        with self.assertRaises(ConditionError):
            ev("5 in 'paid'")

    def test_contains_direction(self):
        self.assertIs(ev("['a'] contains 'a'")["conditions"][0]["result"], True)
        self.assertIs(ev("'paid' contains 'aid'")["conditions"][0]["result"], True)
        with self.assertRaises(ConditionError):
            ev("5 contains 5")

    def test_cross_type_ordering_is_an_error(self):
        for expr in ('"a" > 5', "true < 2", "null >= 1"):
            with self.assertRaises(ConditionError, msg=expr):
                ev(expr)

    def test_list_equality_is_deep(self):
        self.assertIs(ev("[1, [2]] == [1, [2]]")["conditions"][0]["result"], True)
        self.assertIs(ev("[1, 2] == [2, 1]")["conditions"][0]["result"], False)


class TraceTests(unittest.TestCase):
    def test_short_circuit_records_skipped(self):
        # 'z.missing' is never evaluated: no error, but recorded as skipped.
        out = ev("false and z.missing")
        self.assertIs(out["conditions"][0]["result"], False)
        trace = out["conditions"][0]["trace"]
        self.assertEqual(trace["skipped"], ["z.missing"])

    def test_or_short_circuit(self):
        out = ev("true or z.missing")
        self.assertIs(out["conditions"][0]["result"], True)
        self.assertEqual(out["conditions"][0]["trace"]["skipped"], ["z.missing"])

    def test_trace_has_every_node_value(self):
        facts = {"a": 5}
        trace = ev("a > 3 and a < 10", facts)["conditions"][0]["trace"]
        self.assertEqual(trace["op"], "and")
        self.assertEqual([c["expr"] for c in trace["children"]], ["a > 3", "a < 10"])
        left = trace["children"][0]
        self.assertEqual(left["left"]["expr"], "a")
        self.assertEqual(left["left"]["value"], 5)

    def test_no_trace_leak_of_unresolved_paths(self):
        out = ev("true or z.missing")
        self.assertNotIn("error", out)


class AggregationTests(unittest.TestCase):
    def test_list_all_any(self):
        out = ev(["a > 1", "b > 1"], {"a": 2, "b": 0})
        self.assertEqual([c["result"] for c in out["conditions"]], [True, False])
        self.assertIs(out["all"], False)
        self.assertIs(out["any"], True)

    def test_single_string(self):
        out = ev("1 == 1")
        self.assertIs(out["all"], True)
        self.assertIs(out["any"], True)

    def test_empty_is_an_error(self):
        with self.assertRaises(ConditionError):
            ev([])
        with self.assertRaises(ConditionError):
            ev("")

    def test_non_object_facts_is_an_error(self):
        with self.assertRaises(ConditionError):
            ev("true", facts=[1])


class SyntaxErrorTests(unittest.TestCase):
    def test_trailing_token(self):
        with self.assertRaises(ConditionError):
            ev("true false")

    def test_unbalanced_parens(self):
        with self.assertRaises(ConditionError):
            ev("(true")
        with self.assertRaises(ConditionError):
            ev("true)")

    def test_unknown_character(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("a @ b")
        self.assertIn("unexpected character '@'", str(ctx.exception))

    def test_invalid_escape(self):
        with self.assertRaises(ConditionError) as ctx:
            ev(r"'bad \q escape'")
        self.assertIn("invalid escape", str(ctx.exception))

    def test_short_u_escape(self):
        with self.assertRaises(ConditionError):
            ev("'\\u12'")

    def test_keyword_as_path_key_is_an_error(self):
        with self.assertRaises(ConditionError):
            ev("a.and == 1")

    def test_non_string_condition_is_an_error(self):
        with self.assertRaises(ConditionError):
            ev([5])


if __name__ == "__main__":
    unittest.main()