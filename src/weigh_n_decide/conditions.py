"""Literal condition evaluation: a small boolean/comparison grammar over facts.

Recursive-descent parser (no eval()) over a whitelisted operator set. Every
evaluation returns a per-node trace; a missing fact path is an error, never a
silent false.

Grammar (precedence low -> high)::

    expr    := or
    or      := and ("or" and)*
    and     := unary ("and" unary)*
    unary   := "not" unary | "(" expr ")" | comparison
    comparison := operand (("=="|!="|"<"|"<="|">"|">="|"in"|"not in"|"contains") operand)?
    operand := number | string | true | false | null | factpath
    factpath := ident ("." ident | "[" integer "]")*

A comparison with no operator is a truthiness check on its operand.
"""

from __future__ import annotations

import re

__all__ = ["ConditionError", "evaluate_condition", "evaluate_conditions", "parse"]


class ConditionError(ValueError):
    """Raised for syntax errors, unknown tokens, type mismatches, and failed fact paths."""


# Bounds on hostile or generated input. 20 KiB and 5,000 tokens are far
# beyond any readable condition while keeping parse/evaluate time bounded.
_MAX_SOURCE_LENGTH = 20_000
_MAX_TOKENS = 5_000

# Bound on explicit nesting ("(", "not", list literals). Keeps hostile or
# generated input from exhausting the interpreter stack.
_MAX_NESTING = 100


# --------------------------------------------------------------------------
# Tokenizer

_TOKEN_RE = re.compile(
    r"""
      (?P<ws>\s+)
    | (?P<number>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
    | (?P<string>'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*")
    | (?P<op>==|!=|<=|>=|<|>|\(|\)|\[|\]|,|\.|-)
    | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)

_KEYWORDS = {"and", "or", "not", "in", "contains", "true", "false", "null"}
_CMP_OPS = {"==", "!=", "<", "<=", ">", ">="}
_WORD_OPS = {"in": "in", "contains": "contains"}


class _Token:
    __slots__ = ("kind", "value", "start", "end")

    def __init__(self, kind: str, value: str, start: int, end: int):
        self.kind = kind
        self.value = value
        self.start = start
        self.end = end

    def __repr__(self):  # pragma: no cover - debug aid
        return f"_Token({self.kind}, {self.value!r}, {self.start})"


def _tokenize(src: str) -> list[_Token]:
    tokens: list[_Token] = []
    pos = 0
    while pos < len(src):
        match = _TOKEN_RE.match(src, pos)
        if not match:
            raise ConditionError(
                f"unexpected character {src[pos]!r} at position {pos}"
            )
        kind = match.lastgroup
        if kind != "ws":
            tokens.append(_Token(kind, match.group(), pos, match.end()))
            if len(tokens) > _MAX_TOKENS:
                raise ConditionError(
                    f"condition is too complex: more than {_MAX_TOKENS} tokens"
                )
        pos = match.end()
    return tokens


_ESCAPES = {
    "\\": "\\",
    "'": "'",
    '"': '"',
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "t": "\t",
    "r": "\r",
}
_HEX4_RE = re.compile(r"[0-9A-Fa-f]{4}\Z")


def _hex4(body: str, at: int, position: int) -> int:
    hexpart = body[at : at + 4]
    if not _HEX4_RE.fullmatch(hexpart):
        raise ConditionError(f"invalid \\u escape at position {position}")
    return int(hexpart, 16)


def _unescape(body: str, start: int) -> str:
    out: list[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch in "\n\r":
            raise ConditionError(
                f"unescaped newline in string literal at position {start + i}"
            )
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= len(body):
            raise ConditionError(f"unterminated escape at position {start + i}")
        esc = body[i + 1]
        if esc == "u":
            code = _hex4(body, i + 2, start + i)
            # JSON escape semantics: a high surrogate followed by a low
            # surrogate is one character, not two lone code units.
            if 0xD800 <= code <= 0xDBFF and body[i + 6 : i + 8] == "\\u":
                low = _hex4(body, i + 8, start + i + 6)
                if 0xDC00 <= low <= 0xDFFF:
                    out.append(
                        chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00))
                    )
                    i += 12
                    continue
            out.append(chr(code))
            i += 6
            continue
        if esc not in _ESCAPES:
            raise ConditionError(f"invalid escape \\{esc} at position {start + i}")
        out.append(_ESCAPES[esc])
        i += 2
    return "".join(out)


def _number_value(text: str) -> int | float:
    try:
        return float(text) if ("." in text or "e" in text or "E" in text) else int(text)
    except ValueError as exc:
        raise ConditionError(f"invalid number literal {text!r}") from exc


def _is_integer_literal(text: str) -> bool:
    return "." not in text and "e" not in text and "E" not in text


# --------------------------------------------------------------------------
# Parser

class _Parser:
    def __init__(self, src: str):
        self.src = src
        self.tokens = _tokenize(src)
        self.i = 0
        self.depth = 0

    # -- token helpers -----------------------------------------------------

    def _peek(self, ahead: int = 0) -> _Token | None:
        idx = self.i + ahead
        return self.tokens[idx] if idx < len(self.tokens) else None

    def _next(self) -> _Token:
        tok = self._peek()
        if tok is None:
            raise ConditionError(f"unexpected end of expression: {self.src!r}")
        self.i += 1
        return tok

    def _expect(self, kind: str, value: str) -> _Token:
        tok = self._next()
        if tok.kind != kind or tok.value != value:
            raise ConditionError(
                f"expected {value!r} at position {tok.start}, got {tok.value!r}"
            )
        return tok

    def _at_op(self, value: str, ahead: int = 0) -> bool:
        tok = self._peek(ahead)
        return tok is not None and tok.kind == "op" and tok.value == value

    def _at_kw(self, *words: str, ahead: int = 0) -> bool:
        tok = self._peek(ahead)
        return tok is not None and tok.kind == "ident" and tok.value in words

    def _text(self, start: int) -> str:
        return self.src[start : self.tokens[self.i - 1].end]

    def _unexpected(self, tok: _Token) -> ConditionError:
        return ConditionError(f"unexpected token {tok.value!r} at position {tok.start}")

    def _enter(self) -> None:
        self.depth += 1
        if self.depth > _MAX_NESTING:
            raise ConditionError(
                f"expression nests deeper than {_MAX_NESTING} levels"
            )

    def _leave(self) -> None:
        self.depth -= 1

    # -- grammar -----------------------------------------------------------

    def parse(self) -> dict:
        node = self._parse_or()
        tok = self._peek()
        if tok is not None:
            raise self._unexpected(tok)
        return node

    def _parse_or(self) -> dict:
        node = self._parse_and()
        if not self._at_kw("or"):
            return node
        start = node["start"]
        parts = [node]
        while self._at_kw("or"):
            self._next()
            parts.append(self._parse_and())
        return {"kind": "or", "parts": parts, "start": start, "text": self._text(start)}

    def _parse_and(self) -> dict:
        node = self._parse_unary()
        if not self._at_kw("and"):
            return node
        start = node["start"]
        parts = [node]
        while self._at_kw("and"):
            self._next()
            parts.append(self._parse_unary())
        return {"kind": "and", "parts": parts, "start": start, "text": self._text(start)}

    def _parse_unary(self) -> dict:
        if self._at_kw("not") and not self._at_kw("in", ahead=1):
            tok = self._next()
            self._enter()
            try:
                part = self._parse_unary()
            finally:
                self._leave()
            return {"kind": "not", "part": part, "start": tok.start, "text": self._text(tok.start)}
        return self._parse_comparison()

    def _parse_comparison(self) -> dict:
        left = self._parse_operand()
        tok = self._peek()
        op: str | None = None
        if tok is not None:
            if tok.kind == "op" and tok.value in _CMP_OPS:
                self._next()
                op = tok.value
            elif self._at_kw("in"):
                self._next()
                op = "in"
            elif self._at_kw("contains"):
                self._next()
                op = "contains"
            elif self._at_kw("not", ahead=0) and self._at_kw("in", ahead=1):
                self._next()
                self._next()
                op = "not in"
        if op is None:
            # A parenthesized expression already carries its own value (and
            # may be boolean); a bare operand in boolean position gets a
            # truthiness node so its trace shows the coercion.
            if left["kind"] == "group":
                return left
            return {"kind": "truthy", "operand": left, "start": left["start"], "text": self._text(left["start"])}
        right = self._parse_operand()
        return {
            "kind": "cmp",
            "op": op,
            "left": left,
            "right": right,
            "start": left["start"],
            "text": self._text(left["start"]),
        }

    def _parse_operand(self) -> dict:
        tok = self._next()
        start = tok.start
        if tok.kind == "number":
            return {"kind": "lit", "value": _number_value(tok.value), "start": start, "text": self._text(start)}
        if tok.kind == "op" and tok.value == "-":
            num = self._next()
            if num.kind != "number":
                raise self._unexpected(num)
            value = -_number_value(num.value)
            return {"kind": "lit", "value": value, "start": start, "text": self._text(start)}
        if tok.kind == "string":
            value = _unescape(tok.value[1:-1], tok.start + 1)
            return {"kind": "lit", "value": value, "start": start, "text": self._text(start)}
        if tok.kind == "ident":
            if tok.value == "true":
                return {"kind": "lit", "value": True, "start": start, "text": self._text(start)}
            if tok.value == "false":
                return {"kind": "lit", "value": False, "start": start, "text": self._text(start)}
            if tok.value == "null":
                return {"kind": "lit", "value": None, "start": start, "text": self._text(start)}
            return self._parse_path(tok)
        if tok.kind == "op" and tok.value == "[":
            return self._parse_list_literal(start)
        if tok.kind == "op" and tok.value == "(":
            self._enter()
            try:
                node = self._parse_or()
                self._expect("op", ")")
            finally:
                self._leave()
            # A lone parenthesized value (`(a)`, `(1)`) must stay a value so
            # `(a.total) > 100` compares numbers rather than booleans.
            if node["kind"] == "truthy":
                node = node["operand"]
            return {
                "kind": "group",
                "part": node,
                "start": start,
                "text": self._text(start),
            }
        raise self._unexpected(tok)

    def _parse_list_literal(self, start: int) -> dict:
        """A literal list: comma-separated literals (no fact paths — reference
        a list from facts instead)."""
        self._enter()
        try:
            items: list[object] = []
            if not self._at_op("]"):
                while True:
                    item = self._parse_operand()
                    if item["kind"] != "lit":
                        raise ConditionError(
                            f"list literal at position {item['start']} must contain "
                            f"literals only (got a fact path)"
                        )
                    items.append(item["value"])
                    if self._at_op(","):
                        self._next()
                        continue
                    break
            self._expect("op", "]")
        finally:
            self._leave()
        return {"kind": "lit", "value": items, "start": start, "text": self._text(start)}

    def _parse_path(self, first: _Token) -> dict:
        steps: list[tuple[str, object]] = [("key", first.value)]
        text_end = first.end
        while True:
            if self._at_op("."):
                self._next()
                tok = self._next()
                if tok.kind != "ident" or tok.value in _KEYWORDS:
                    raise ConditionError(
                        f"expected a name after '.' at position {tok.start}, got {tok.value!r}"
                    )
                steps.append(("key", tok.value))
                text_end = tok.end
            elif self._at_op("["):
                self._next()
                idx_tok = self._next()
                sign = 1
                if idx_tok.kind == "op" and idx_tok.value == "-":
                    sign = -1
                    idx_tok = self._next()
                if idx_tok.kind != "number" or not _is_integer_literal(idx_tok.value):
                    raise ConditionError(
                        f"expected an integer index at position {idx_tok.start}, got {idx_tok.value!r}"
                    )
                self._expect("op", "]")
                steps.append(("index", sign * int(idx_tok.value)))
                text_end = self.tokens[self.i - 1].end
            else:
                break
        return {
            "kind": "path",
            "steps": steps,
            "start": first.start,
            "text": self.src[first.start:text_end],
        }


# --------------------------------------------------------------------------
# Evaluation

def _type_name(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _truthy(value: object) -> bool:
    return bool(value)


def _json_equal(left: object, right: object) -> bool:
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _json_equal(left[k], right[k]) for k in left
        )
    if _is_number(left) and _is_number(right):
        return left == right
    if type(left) is not type(right) and not (left is None and right is None):
        return False
    return left == right


def _resolve_path(facts: dict, node: dict) -> object:
    current: object = facts
    prev_name: object = "<facts>"
    for step_kind, step_value in node["steps"]:
        if step_kind == "key":
            if not isinstance(current, dict):
                raise ConditionError(
                    f"fact path {node['text']!r}: {prev_name!r} is not an object "
                    f"(got {_type_name(current)})"
                )
            if step_value not in current:
                raise ConditionError(
                    f"fact path {node['text']!r}: missing key {step_value!r}"
                )
            current = current[step_value]
            prev_name = step_value
        else:
            if not isinstance(current, list):
                raise ConditionError(
                    f"fact path {node['text']!r}: {prev_name!r} is not a list "
                    f"(got {_type_name(current)})"
                )
            index = int(step_value)
            if not -len(current) <= index < len(current):
                raise ConditionError(
                    f"fact path {node['text']!r}: index {index} out of range "
                    f"(length {len(current)})"
                )
            current = current[index]
            prev_name = f"{prev_name}[{index}]"
    return current


def _eval(node: dict, facts: dict) -> tuple[object, dict]:
    kind = node["kind"]
    if kind == "lit":
        value = node["value"]
        return value, {"expr": node["text"], "value": value, "op": "literal"}
    if kind == "path":
        value = _resolve_path(facts, node)
        return value, {"expr": node["text"], "value": value, "op": "fact"}
    if kind == "truthy":
        value, child = _eval(node["operand"], facts)
        result = _truthy(value)
        return result, {"expr": node["text"], "value": result, "op": "truthy", "of": child}
    if kind == "group":
        value, child = _eval(node["part"], facts)
        return value, {"expr": node["text"], "value": value, "op": "group", "child": child}
    if kind == "not":
        value, child = _eval(node["part"], facts)
        result = not _truthy(value)
        return result, {"expr": node["text"], "value": result, "op": "not", "child": child}
    if kind in ("and", "or"):
        children: list[dict] = []
        skipped: list[str] = []
        result: bool | None = None
        for index, part in enumerate(node["parts"]):
            value, child = _eval(part, facts)
            children.append(child)
            truthy = _truthy(value)
            if kind == "and" and not truthy:
                result = False
                skipped = [p["text"] for p in node["parts"][index + 1 :]]
                break
            if kind == "or" and truthy:
                result = True
                skipped = [p["text"] for p in node["parts"][index + 1 :]]
                break
        if result is None:
            result = kind == "and"
        trace = {"expr": node["text"], "value": result, "op": kind, "children": children}
        if skipped:
            trace["skipped"] = skipped
        return result, trace
    if kind == "cmp":
        op = node["op"]
        left_value, left_trace = _eval(node["left"], facts)
        right_value, right_trace = _eval(node["right"], facts)
        result = _compare(op, left_value, right_value, node["text"])
        return result, {
            "expr": node["text"],
            "value": result,
            "op": op,
            "left": left_trace,
            "right": right_trace,
        }
    raise ConditionError(f"unknown node kind {kind!r}")  # pragma: no cover


def _compare(op: str, left: object, right: object, text: str) -> bool:
    if op in ("<", "<=", ">", ">="):
        comparable = (_is_number(left) and _is_number(right)) or (
            isinstance(left, str) and isinstance(right, str)
        )
        if not comparable:
            raise ConditionError(
                f"{text!r}: {op!r} cannot compare {_type_name(left)} with {_type_name(right)}"
            )
        if op == "<":
            return left < right  # type: ignore[operator]
        if op == "<=":
            return left <= right  # type: ignore[operator]
        if op == ">":
            return left > right  # type: ignore[operator]
        return left >= right  # type: ignore[operator]
    if op == "==":
        return _json_equal(left, right)
    if op == "!=":
        return not _json_equal(left, right)
    if op in ("in", "not in"):
        if isinstance(right, list):
            result = any(_json_equal(left, item) for item in right)
        elif isinstance(right, str):
            if not isinstance(left, str):
                raise ConditionError(
                    f"{text!r}: 'in' over a string needs a string on the left "
                    f"(got {_type_name(left)})"
                )
            result = left in right
        else:
            raise ConditionError(
                f"{text!r}: 'in' needs a list or string on the right "
                f"(got {_type_name(right)})"
            )
        return result if op == "in" else not result
    if op == "contains":
        if isinstance(left, list):
            return any(_json_equal(right, item) for item in left)
        if isinstance(left, str):
            if not isinstance(right, str):
                raise ConditionError(
                    f"{text!r}: 'contains' over a string needs a string on the right "
                    f"(got {_type_name(right)})"
                )
            return right in left
        raise ConditionError(
            f"{text!r}: 'contains' needs a list or string on the left "
            f"(got {_type_name(left)})"
        )
    raise ConditionError(f"unknown operator {op!r}")  # pragma: no cover


# --------------------------------------------------------------------------
# Public API

def parse(expr: str) -> dict:
    """Parse one condition expression into an AST (exposed for tests/tools)."""
    if not isinstance(expr, str) or not expr.strip():
        raise ConditionError("condition must be a non-empty string")
    if len(expr) > _MAX_SOURCE_LENGTH:
        raise ConditionError(
            f"condition is too long: {len(expr)} characters (max {_MAX_SOURCE_LENGTH})"
        )
    return _Parser(expr).parse()


def evaluate_condition(expr: str, facts: dict | None = None) -> dict:
    """Evaluate one condition; returns {"result": bool, "trace": node}."""
    return evaluate_conditions([expr], facts)["conditions"][0]


def evaluate_conditions(conditions: str | list[str], facts: dict | None = None) -> dict:
    """Evaluate one or more conditions against facts.

    Returns {"conditions": [{"condition", "result", "trace"}, ...],
    "all": bool, "any": bool}. Raises ConditionError on syntax errors, type
    mismatches, and failed fact paths.
    """
    if isinstance(conditions, str):
        conditions = [conditions]
    if not isinstance(conditions, list) or not conditions:
        raise ConditionError(
            "conditions must be a non-empty expression or list of expressions"
        )
    if facts is None:
        facts = {}
    if not isinstance(facts, dict):
        raise ConditionError("facts must be a JSON object")
    results: list[bool] = []
    entries: list[dict] = []
    for expr in conditions:
        if not isinstance(expr, str):
            raise ConditionError("each condition must be a string")
        node = parse(expr)
        value, trace = _eval(node, facts)
        result = _truthy(value)
        results.append(result)
        entries.append({"condition": expr, "result": result, "trace": trace})
    return {
        "conditions": entries,
        "all": all(results),
        "any": any(results),
    }