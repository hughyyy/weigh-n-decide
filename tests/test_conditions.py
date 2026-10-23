"""Condition evaluator tests: grammar, precedence, traces, fact paths, errors."""

import unittest

from weigh_n_decide.conditions import ConditionError, evaluate_conditions, parse


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
            ev(r'"line\nbreak" == v', {"v": "line\nbreak"})["conditions"][0]["result"],
            True,
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

    def test_short_circuit_records_every_skipped_part(self):
        out = ev("false and a and b", {"a": True, "b": True})
        self.assertIs(out["conditions"][0]["result"], False)
        self.assertEqual(out["conditions"][0]["trace"]["skipped"], ["a", "b"])
        out = ev("true or a or b", {"a": False, "b": False})
        self.assertEqual(out["conditions"][0]["trace"]["skipped"], ["a", "b"])


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


class EdgeCaseTests(unittest.TestCase):
    """Branches and boundary behavior the grammar tests leave untrodden."""

    def test_evaluate_condition_single_entry_api(self):
        from weigh_n_decide.conditions import evaluate_condition

        out = evaluate_condition("a == 1", {"a": 1})
        self.assertIs(out["result"], True)
        self.assertIn("trace", out)

    def test_less_or_equal(self):
        self.assertIs(ev("5 <= 5")["conditions"][0]["result"], True)
        self.assertIs(ev("'a' <= 'b'")["conditions"][0]["result"], True)
        self.assertIs(ev("'b' <= 'a'")["conditions"][0]["result"], False)

    def test_unary_minus_float(self):
        self.assertIs(ev("-1.5 < -1")["conditions"][0]["result"], True)

    def test_negative_float_literal(self):
        self.assertIs(ev("-1.5 == -1.5")["conditions"][0]["result"], True)

    def test_or_all_false_is_false(self):
        self.assertIs(ev("false or false")["conditions"][0]["result"], False)
        self.assertIs(ev("false or 0 or null or ''")["conditions"][0]["result"], False)

    def test_empty_list_literal_is_falsy(self):
        self.assertIs(ev("[]")["conditions"][0]["result"], False)
        self.assertIs(ev("[] == []")["conditions"][0]["result"], True)

    def test_list_literal_with_negative_and_float_items(self):
        self.assertIs(ev("[-1] == [-1.0]")["conditions"][0]["result"], True)
        self.assertIs(
            ev("'x' in [1, 2.0, 'x']")["conditions"][0]["result"], True
        )

    def test_null_in_list(self):
        self.assertIs(ev("null in [null]")["conditions"][0]["result"], True)
        self.assertIs(ev("0 in [null]")["conditions"][0]["result"], False)

    def test_empty_string_is_falsy(self):
        self.assertIs(ev("''")["conditions"][0]["result"], False)
        self.assertIs(ev("not ''")["conditions"][0]["result"], True)
        self.assertIs(ev("'' or 'x'")["conditions"][0]["result"], True)

    def test_dict_equality_through_facts(self):
        facts = {"x": {"a": 1, "b": [1]}, "y": {"a": 1, "b": [1]}, "z": {"a": 2, "b": [1]}}
        self.assertIs(ev("x == y", facts)["conditions"][0]["result"], True)
        self.assertIs(ev("x == z", facts)["conditions"][0]["result"], False)
        self.assertIs(ev("x != z", facts)["conditions"][0]["result"], True)

    def test_number_int_float_equality_is_numeric(self):
        self.assertIs(ev("1 == 1.0")["conditions"][0]["result"], True)
        self.assertIs(ev("[1] == [1.0]")["conditions"][0]["result"], True)
        self.assertIs(ev("1 in [1.0]")["conditions"][0]["result"], True)

    def test_double_negation(self):
        self.assertIs(ev("not not true")["conditions"][0]["result"], True)
        self.assertIs(ev("not not 1")["conditions"][0]["result"], True)

    def test_keywords_are_case_sensitive(self):
        # Bare idents are fact paths; 'True' is not the true literal.
        with self.assertRaises(ConditionError):
            ev("True")
        self.assertIs(ev("flag", {"flag": True})["conditions"][0]["result"], True)

    def test_underscore_path_keys(self):
        self.assertIs(ev("_x == 1", {"_x": 1})["conditions"][0]["result"], True)

    def test_leading_plus_is_an_error(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("+5")
        self.assertIn("unexpected character '+'", str(ctx.exception))

    def test_nested_short_circuit_preserves_skipped(self):
        out = ev("not (false and z.missing)")
        self.assertIs(out["conditions"][0]["result"], True)
        child = out["conditions"][0]["trace"]["child"]
        self.assertEqual(child["op"], "group")
        self.assertEqual(child["child"]["skipped"], ["z.missing"])

    def test_trace_shapes_for_unary_and_truthy(self):
        out = ev("not a", {"a": 0})
        trace = out["conditions"][0]["trace"]
        # `not a` parses as not(truthy(a)): the operand is wrapped in a
        # truthiness node before negation.
        self.assertEqual(trace["op"], "not")
        self.assertEqual(trace["child"]["op"], "truthy")
        self.assertEqual(trace["child"]["of"]["op"], "fact")
        self.assertEqual(trace["child"]["of"]["value"], 0)
        out2 = ev("a", {"a": 5})
        self.assertEqual(out2["conditions"][0]["trace"]["op"], "truthy")
        self.assertEqual(out2["conditions"][0]["trace"]["of"]["value"], 5)

    def test_deep_path_error_names_the_step(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("a.b.c", {"a": {"b": 1}})
        self.assertIn("'b' is not an object", str(ctx.exception))
        self.assertIn("number", str(ctx.exception))

    def test_index_into_object_reports_object_type(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("a[0]", {"a": {"b": 1}})
        self.assertIn("is not a list (got object)", str(ctx.exception))

    def test_non_json_value_in_facts_degrades_gracefully(self):
        # Facts are a JSON object by contract; non-JSON values still produce
        # a clear error, not a traceback.
        with self.assertRaises(ConditionError) as ctx:
            ev("x.y", {"x": object()})
        self.assertIn("got object", str(ctx.exception))
        # ...and a value compares equal to itself.
        self.assertIs(
            ev("x == x", {"x": (1, 2)})["conditions"][0]["result"], True
        )

    def test_negative_index_out_of_range(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("a[-2]", {"a": [1]})
        self.assertIn("index -2 out of range", str(ctx.exception))

    def test_empty_parens_are_an_error(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("()")
        self.assertIn("unexpected token ')'", str(ctx.exception))

    def test_paren_mismatch_reports_expected_token(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("(true 5)")
        self.assertIn("expected ')'", str(ctx.exception))

    def test_unary_minus_requires_a_number(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("-true")
        self.assertIn("unexpected token 'true'", str(ctx.exception))

    def test_list_literal_rejects_fact_paths(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("[a]")
        self.assertIn("literals only", str(ctx.exception))

    def test_list_literal_rejects_trailing_comma(self):
        with self.assertRaises(ConditionError):
            ev("[1,]")

    def test_fractional_index_is_an_error(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("a[1.5]", {"a": [1]})
        self.assertIn("expected an integer index", str(ctx.exception))

    def test_identifier_index_is_an_error(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("a[x]", {"a": [1]})
        self.assertIn("expected an integer index", str(ctx.exception))

    def test_in_right_operand_must_be_list_or_string(self):
        for expr in ("'x' in 5", "'x' in null", "1 in 2"):
            with self.assertRaises(ConditionError, msg=expr) as ctx:
                ev(expr)
            self.assertIn("'in' needs a list or string on the right", str(ctx.exception))

    def test_contains_over_string_needs_string_right(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("'paid' contains 5")
        self.assertIn("needs a string on the right", str(ctx.exception))

    def test_bad_unicode_escape_hex_is_an_error(self):
        with self.assertRaises(ConditionError) as ctx:
            ev(r"'\uzzzz'")
        self.assertIn("invalid \\u escape", str(ctx.exception))

    def test_whitespace_and_newlines_are_tolerated(self):
        self.assertIs(ev("  true  ")["conditions"][0]["result"], True)
        self.assertIs(ev("\ttrue\n")["conditions"][0]["result"], True)

    def test_parse_rejects_non_string_and_blank(self):
        from weigh_n_decide.conditions import parse

        for bad in (None, 5, "", "   "):
            with self.assertRaises(ConditionError, msg=repr(bad)):
                parse(bad)

    def test_conditions_must_be_string_or_list(self):
        for bad in (5, {}, None):
            with self.assertRaises(ConditionError, msg=repr(bad)):
                ev(bad)

    def test_unescape_is_defensive_about_trailing_backslash(self):
        # The tokenizer can never produce a string body ending in a lone
        # backslash (the regex pairs escapes and requires a closing quote);
        # _unescape still rejects one rather than misparse.
        from weigh_n_decide import conditions as c

        with self.assertRaises(ConditionError) as ctx:
            c._unescape("abc\\", 0)
        self.assertIn("unterminated escape", str(ctx.exception))

    def test_parse_ast_shape_for_compound_expression(self):
        from weigh_n_decide.conditions import parse

        node = parse("a.b[0] > 5 and not c")
        self.assertEqual(node["kind"], "and")
        left, right = node["parts"]
        self.assertEqual(left["kind"], "cmp")
        self.assertEqual(left["op"], ">")
        self.assertEqual(left["left"]["kind"], "path")
        self.assertEqual(
            left["left"]["steps"], [("key", "a"), ("key", "b"), ("index", 0)]
        )
        self.assertEqual(left["right"], {"kind": "lit", "value": 5, "start": 9, "text": "5"})
        self.assertEqual(right["kind"], "not")
        self.assertEqual(right["part"]["kind"], "truthy")
        self.assertEqual(right["part"]["operand"]["kind"], "path")
        self.assertEqual(right["part"]["operand"]["steps"], [("key", "c")])

    def test_evaluation_is_atomic_on_syntax_error(self):
        # A bad expression aborts the whole call: no partial results.
        with self.assertRaises(ConditionError):
            ev(["a > 1", "("])
        with self.assertRaises(ConditionError):
            ev(["true", "@"])

    def test_non_ascii_identifiers_are_rejected(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("é > 1")
        self.assertIn("unexpected character 'é'", str(ctx.exception))

    def test_leading_zeros_and_big_integers(self):
        self.assertIs(ev("007 == 7")["conditions"][0]["result"], True)
        big = "9" * 30
        self.assertIs(ev(f"{big} > 1")["conditions"][0]["result"], True)
        self.assertIs(ev(f"{big} == {big}")["conditions"][0]["result"], True)

    def test_string_escapes_carriage_return_and_quotes(self):
        facts = {"v": "a\rb"}
        self.assertIs(ev(r'"a\rb" == v', facts)["conditions"][0]["result"], True)
        quoted = {"v": 'say "hi"'}
        self.assertIs(ev(r'"say \"hi\"" == v', quoted)["conditions"][0]["result"], True)
        self.assertIs(ev(r"'it\'s' == v", {"v": "it's"})["conditions"][0]["result"], True)

    def test_not_in_over_string(self):
        self.assertIs(ev("'x' not in 'abc'")["conditions"][0]["result"], True)
        self.assertIs(ev("'a' not in 'abc'")["conditions"][0]["result"], False)

    def test_nested_list_literals(self):
        self.assertIs(ev("[['a']] contains ['a']")["conditions"][0]["result"], True)
        self.assertIs(ev("[[1]] == [[1.0]]")["conditions"][0]["result"], True)
        self.assertIs(ev("[] in [[], [1]]")["conditions"][0]["result"], True)

    def test_path_to_empty_list_is_falsy(self):
        self.assertIs(ev("items", {"items": []})["conditions"][0]["result"], False)

    def test_null_mixed_comparisons(self):
        self.assertIs(ev("null != 0")["conditions"][0]["result"], True)
        self.assertIs(ev("null != null")["conditions"][0]["result"], False)
        self.assertIs(ev("'' in ['', 'x']")["conditions"][0]["result"], True)
        self.assertIs(ev("0 in [0.0]")["conditions"][0]["result"], True)

    def test_chained_comparisons_are_rejected(self):
        # The grammar is binary-only; '1 < 2 < 3' must fail, not chain.
        with self.assertRaises(ConditionError) as ctx:
            ev("1 < 2 < 3")
        self.assertIn("unexpected token '<'", str(ctx.exception))

    def test_scientific_notation_is_supported(self):
        self.assertIs(ev("1e5 > 1")["conditions"][0]["result"], True)
        self.assertIs(ev("5e-1 > 0")["conditions"][0]["result"], True)
        self.assertIs(ev("1E+2 == 100")["conditions"][0]["result"], True)
        with self.assertRaises(ConditionError):
            ev(".5 > 0")

    def test_trailing_dot_after_number_is_rejected(self):
        with self.assertRaises(ConditionError):
            ev("1. == 1")

    def test_number_followed_by_ident_is_rejected(self):
        with self.assertRaises(ConditionError):
            ev("5abc == 5")

    def test_keywords_are_valid_top_level_path_keys(self):
        # Keywords are reserved after a dot and as operators, but a bare
        # keyword still parses as a fact path.
        self.assertIs(ev("and", {"and": True})["conditions"][0]["result"], True)
        self.assertIs(ev("or == 1", {"or": 1})["conditions"][0]["result"], True)

    def test_literal_cannot_be_indexed(self):
        for expr in ("true.foo", "null.x", "1.y"):
            with self.assertRaises(ConditionError, msg=expr):
                ev(expr)

    def test_list_membership_with_nested_literals(self):
        self.assertIs(ev("[1] in [[1], [2]]")["conditions"][0]["result"], True)
        self.assertIs(ev("[1] in [[2]]")["conditions"][0]["result"], False)
        self.assertIs(ev("[[1]] contains [1]")["conditions"][0]["result"], True)

    def test_negative_zero_index_is_the_first_element(self):
        self.assertIs(ev("a[-0] == 1", {"a": [1, 2]})["conditions"][0]["result"], True)

    def test_keyword_as_list_index_is_rejected(self):
        with self.assertRaises(ConditionError):
            ev("a[true]", {"a": [1]})

    def test_deep_mixed_path_resolves(self):
        facts = {"a": {"b": [{"c": [10, 20]}]}}
        self.assertIs(
            ev("a.b[0].c[-1] == 20", facts)["conditions"][0]["result"], True
        )

    def test_or_chain_trace_has_all_children(self):
        facts = {"a": False, "b": False, "c": True}
        out = ev("a or b or c", facts)
        trace = out["conditions"][0]["trace"]
        self.assertEqual([child["expr"] for child in trace["children"]], ["a", "b", "c"])
        # 'skipped' is only present when short-circuiting cut evaluation.
        self.assertNotIn("skipped", trace)

    def test_triple_not(self):
        self.assertIs(ev("not not not true")["conditions"][0]["result"], False)
        # 0 is falsy: not(not(not 0)) = not(not True) = True.
        self.assertIs(
            ev("not (not (not a))", {"a": 0})["conditions"][0]["result"], True
        )


class EscapeDifferentialTests(unittest.TestCase):
    """String escapes agree with the stdlib JSON decoder, case by case."""

    CASES = [
        ("plain", "plain"),
        (r"a\"b", 'a"b'),
        (r"a\\b", "a\\b"),
        (r"a\nb", "a\nb"),
        (r"a\tb\rc", "a\tb\rc"),
        (r"\u00e9", "é"),
        (r"\u0041", "A"),
        (r"\u0000", "\x00"),
        (r"\ud83d\ude00", "😀"),
    ]

    def test_escapes_match_json_decoding(self):
        import json

        from weigh_n_decide import conditions as c

        for body, expected in self.CASES:
            with self.subTest(body=body):
                self.assertEqual(c._unescape(body, 0), expected)
                self.assertEqual(json.loads('"' + body + '"'), expected)

    def test_single_quote_escape_has_no_json_counterpart(self):
        from weigh_n_decide import conditions as c

        self.assertEqual(c._unescape(r"it\'s", 0), "it's")


class NestingTests(unittest.TestCase):
    """Explicit nesting is bounded; traces quote parenthesized source verbatim."""

    def test_one_hundred_nesting_levels_is_accepted(self):
        expr = "(" * 100 + "true" + ")" * 100
        self.assertIs(ev(expr)["conditions"][0]["result"], True)

    def test_parenthesis_nesting_beyond_the_limit_is_an_error(self):
        expr = "(" * 101 + "true" + ")" * 101
        with self.assertRaises(ConditionError) as ctx:
            ev(expr)
        self.assertIn("nests deeper than 100 levels", str(ctx.exception))

    def test_unary_nesting_beyond_the_limit_is_an_error(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("not " * 101 + "true")
        self.assertIn("nests deeper than 100 levels", str(ctx.exception))

    def test_parenthesized_trace_text_is_verbatim(self):
        trace = ev("(a > 1) and b", {"a": 2, "b": True})["conditions"][0]["trace"]
        self.assertEqual(trace["expr"], "(a > 1) and b")
        self.assertEqual([child["expr"] for child in trace["children"]], ["(a > 1)", "b"])

    def test_nested_parenthesized_trace_text_is_verbatim(self):
        trace = ev("not ((a) or b)", {"a": False, "b": False})["conditions"][0]["trace"]
        self.assertEqual(trace["expr"], "not ((a) or b)")
        self.assertEqual(trace["child"]["expr"], "((a) or b)")

    def test_group_ast_quotes_the_full_span(self):
        self.assertEqual(parse("((a) or b)")["text"], "((a) or b)")
        self.assertEqual(parse("(a)")["text"], "(a)")

    def test_parentheses_group_operands_too(self):
        self.assertIs(ev("(a) == 1", {"a": 1})["conditions"][0]["result"], True)
        self.assertIs(ev("a == (1)", {"a": 1})["conditions"][0]["result"], True)
        self.assertIs(
            ev("(a.total) > 100", {"a": {"total": 250}})["conditions"][0]["result"],
            True,
        )
        self.assertIs(
            ev("(a > 1) == true", {"a": 2})["conditions"][0]["result"], True
        )


class UnicodeEscapeTests(unittest.TestCase):
    def test_surrogate_pair_escape_decodes_to_one_character(self):
        self.assertIs(
            ev(r"v == '\ud83d\ude00'", {"v": "😀"})["conditions"][0]["result"], True
        )

    def test_high_surrogate_followed_by_a_non_surrogate_stays_lone(self):
        self.assertIs(
            ev(r"v == '\ud83dA'", {"v": "\ud83dA"})["conditions"][0]["result"], True
        )

    def test_unpaired_high_surrogate_is_preserved(self):
        self.assertIs(
            ev(r"v == '\ud83d'", {"v": "\ud83d"})["conditions"][0]["result"], True
        )

    def test_unpaired_low_surrogate_is_preserved(self):
        self.assertIs(
            ev(r"v == '\ude00'", {"v": "\ude00"})["conditions"][0]["result"], True
        )

    def test_high_surrogate_before_non_surrogate_escape_decodes_both(self):
        self.assertIs(
            ev(r"v == '\ud83d\u0041'", {"v": "\ud83dA"})["conditions"][0]["result"],
            True,
        )

    def test_malformed_second_surrogate_escape_is_an_error(self):
        with self.assertRaises(ConditionError) as ctx:
            ev(r"'\ud83d\uzzzz'")
        self.assertIn("invalid \\u escape", str(ctx.exception))

    def test_short_second_surrogate_escape_is_an_error(self):
        with self.assertRaises(ConditionError) as ctx:
            ev(r"'\ud83d\u12'")
        self.assertIn("invalid \\u escape", str(ctx.exception))

    def test_surrogate_pair_range_boundaries_combine(self):
        self.assertIs(
            ev(r"v == '\ud800\udc00'", {"v": "\U00010000"})["conditions"][0]["result"],
            True,
        )
        self.assertIs(
            ev(r"v == '\udbff\udfff'", {"v": "\U0010FFFF"})["conditions"][0]["result"],
            True,
        )

    def test_pair_range_boundary_surrogates_stay_lone_alone(self):
        self.assertIs(
            ev(r"v == '\udbff'", {"v": "\udbff"})["conditions"][0]["result"], True
        )
        self.assertIs(
            ev(r"v == '\udc00'", {"v": "\udc00"})["conditions"][0]["result"], True
        )


class ComparisonEdgeTests(unittest.TestCase):
    def test_boolean_ordering_is_rejected(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("true < false")
        self.assertIn("'<' cannot compare boolean with boolean", str(ctx.exception))

    def test_in_over_object_reports_object(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("'k' in obj", {"obj": {"k": 1}})
        self.assertIn("needs a list or string on the right (got object)", str(ctx.exception))

    def test_contains_over_object_reports_object(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("obj contains 'k'", {"obj": {"k": 1}})
        self.assertIn("needs a list or string on the left (got object)", str(ctx.exception))

    def test_dict_equality_ignores_key_order_but_not_bool_identity(self):
        facts = {
            "x": {"a": True, "b": 1},
            "y": {"b": 1, "a": True},
            "z": {"a": 1, "b": 1},
        }
        self.assertIs(ev("x == y", facts)["conditions"][0]["result"], True)
        self.assertIs(ev("x == z", facts)["conditions"][0]["result"], False)
        self.assertIs(ev("x != z", facts)["conditions"][0]["result"], True)

    def test_dict_equality_requires_the_same_keys(self):
        facts = {"p": {"a": 1}, "q": {"b": 1}, "r": {"a": 1.0}}
        self.assertIs(ev("p == q", facts)["conditions"][0]["result"], False)
        self.assertIs(ev("p != q", facts)["conditions"][0]["result"], True)
        # Numeric equality applies inside nested objects too.
        self.assertIs(ev("p == r", facts)["conditions"][0]["result"], True)

    def test_greater_or_equal_accepts_equal_operands(self):
        self.assertIs(ev("5 >= 5")["conditions"][0]["result"], True)
        self.assertIs(ev("5.0 >= 5")["conditions"][0]["result"], True)
        self.assertIs(ev("'a' >= 'a'")["conditions"][0]["result"], True)
        self.assertIs(ev("5 >= 6")["conditions"][0]["result"], False)

    def test_index_equal_to_length_is_out_of_range(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("a[2]", {"a": [1, 2]})
        self.assertIn("index 2 out of range", str(ctx.exception))

    def test_empty_index_brackets_report_the_closing_bracket(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("a[]", {"a": [1]})
        self.assertIn("expected an integer index", str(ctx.exception))
        self.assertIn("got ']'", str(ctx.exception))

    def test_bare_not_after_an_operand_is_not_an_operator(self):
        # Only `not in` is an operator; `a not b` must stay a syntax error
        # rather than being read as `a not in b`.
        with self.assertRaises(ConditionError) as ctx:
            ev("a not b", {"a": True, "b": [True]})
        self.assertIn("unexpected token 'not'", str(ctx.exception))

    def test_negative_index_at_the_lower_bound(self):
        self.assertIs(ev("a[-2] == 1", {"a": [1, 2]})["conditions"][0]["result"], True)

    def test_trailing_path_syntax_is_an_error(self):
        for expr in ("a.", "a["):
            with self.assertRaises(ConditionError, msg=expr):
                ev(expr, {"a": [1]})


class AggregationEdgeTests(unittest.TestCase):
    def test_single_false_condition_aggregates_false(self):
        out = ev("a > 1", {"a": 1})
        self.assertIs(out["all"], False)
        self.assertIs(out["any"], False)

    def test_conditions_echo_their_sources_in_order(self):
        out = ev([" a > 1 ", "b"], {"a": 2, "b": False})
        self.assertEqual([c["condition"] for c in out["conditions"]], [" a > 1 ", "b"])
        self.assertIs(out["all"], False)
        self.assertIs(out["any"], True)

    def test_every_condition_gets_its_own_trace(self):
        out = ev(["a", "b"], {"a": True, "b": False})
        self.assertEqual([c["trace"]["value"] for c in out["conditions"]], [True, False])


class TruthTableTests(unittest.TestCase):
    """Exhaustive boolean valuations agree with Python's own operators."""

    EXPRESSIONS = [
        "a and b or c and d",
        "a or b and not c",
        "not (a or b) and c",
        "(a or b) and (c or d)",
        "a and not b or c and not d",
        "not a and b or not c and d",
        "a or b or c or d",
        "a and b and c and d",
        "not (a and b and c and d)",
        "a and (b or c) and (d or not a)",
    ]

    def test_every_valuation_matches_python(self):
        import itertools

        for expr in self.EXPRESSIONS:
            for values in itertools.product([True, False], repeat=4):
                facts = dict(zip("abcd", values))
                with self.subTest(expr=expr, facts=facts):
                    self.assertIs(
                        ev(expr, facts)["conditions"][0]["result"],
                        eval(expr, {"__builtins__": {}}, facts),
                    )

    def test_not_applies_to_a_whole_in_comparison(self):
        # `not x in y` is not(n (x in y)) — same as Python — never `x not in y`
        # applied to a negated operand.
        facts = {"tags": ["a"]}
        self.assertIs(ev("not 'a' in tags", facts)["conditions"][0]["result"], False)
        self.assertIs(ev("not 'b' in tags", facts)["conditions"][0]["result"], True)
        self.assertIs(ev("not not 'a' in tags", facts)["conditions"][0]["result"], True)


class ParserRobustnessFuzzTests(unittest.TestCase):
    """Arbitrary junk is always a ConditionError — never another exception."""

    TOKENS = [
        "a", "b", "c", "0", "1", "23", "1.5", "(", ")", "[", "]", "'x'",
        '"y"', "'", '"', ".", ",", "<", ">", "=", "==", "!=", "<=", ">=",
        "!", "-", "+", "*", "/", "\\", " ", "\t", "\n", "_", "and", "or",
        "not", "in", "contains", "true", "false", "null", "é",
    ]
    FACTS = {"a": 1, "b": [1], "c": "x"}

    def test_random_junk_only_raises_condition_errors(self):
        import random

        rng = random.Random(0xF022)
        unexpected = []
        for _ in range(3000):
            expr = "".join(rng.choice(self.TOKENS) for _ in range(rng.randrange(0, 12)))
            try:
                out = evaluate_conditions(expr, self.FACTS)
                self.assertIn(out["conditions"][0]["result"], (True, False))
            except ConditionError:
                continue
            except Exception as exc:  # noqa: BLE001 - the assertion is the point
                unexpected.append((expr, type(exc).__name__, str(exc)))
        self.assertEqual(unexpected, [])


class DeterminismTests(unittest.TestCase):
    def test_repeated_evaluation_is_identical(self):
        import json

        first = ev("a > 1 and (b contains 'x')", {"a": 2, "b": ["x"]})
        second = ev("a > 1 and (b contains 'x')", {"a": 2, "b": ["x"]})
        self.assertEqual(json.dumps(first, sort_keys=True),
                         json.dumps(second, sort_keys=True))


class InputLimitTests(unittest.TestCase):
    def test_token_limit(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("1 " * 5001)
        self.assertIn("too complex", str(ctx.exception))

    def test_source_length_limit(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("1" * 20_001)
        self.assertIn("too long", str(ctx.exception))

    def test_unescaped_newline_in_string_is_rejected(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("'a\nb'")
        self.assertIn("unescaped newline", str(ctx.exception))

    def test_huge_integer_literal_is_a_condition_error(self):
        with self.assertRaises(ConditionError) as ctx:
            ev("9" * 5000 + " > 1")
        self.assertIn("invalid number literal", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()