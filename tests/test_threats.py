"""Модель угроз и тесты не расходятся: у каждой серьёзной угрозы есть существующий тест."""

import ast
import json
from pathlib import Path

THREATS = json.loads(Path("threats.json").read_text(encoding="utf-8"))


def _test_exists(reference: str) -> bool:
    path, name = reference.split("::")
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return any(isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name for node in tree.body)


def test_serious_threats_have_tests():
    missing = [t["id"] for t in THREATS if t["impact"] in ("critical", "high") and not t["test"]]
    assert missing == []


def test_referenced_tests_exist():
    broken = [ref for t in THREATS for ref in t["test"] if not _test_exists(ref)]
    assert broken == []
