"""AIME answer extraction, pass@1, and majority voting.

The boxed-answer path follows the same decision order as PM-KVQ: prefer the
last ``\\boxed{}`` span, then a light normalization before string equality.
"""
from __future__ import annotations

import re
from collections import Counter


def last_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed")
    if start < 0:
        start = text.rfind("\\fbox")
    if start < 0:
        return None
    opening = text.find("{", start)
    if opening < 0:
        return None
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]
    return None


def normalize_answer(text: str | None) -> str | None:
    if text is None:
        return None
    value = str(text).strip()
    value = value.replace(",", "")
    value = value.replace("\\!", "").replace("\\ ", "").replace(" ", "")
    value = value.replace("tfrac", "frac").replace("dfrac", "frac")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("$", "")
    value = value.rstrip(".")
    return value


def extract_answer(text: str) -> str | None:
    boxed = last_boxed(text)
    if boxed is not None:
        return normalize_answer(boxed)
    return normalize_answer(text.strip().splitlines()[-1] if text.strip() else None)


def answers_equal(prediction: str | None, gold: str | None) -> bool:
    left = normalize_answer(prediction)
    right = normalize_answer(gold)
    if left is None or right is None:
        return False
    return left == right


def pass_at_1(judgements: list[bool]) -> float:
    if not judgements:
        return float("nan")
    return float(sum(bool(item) for item in judgements) / len(judgements))


def majority_vote(answers: list[str | None]) -> str | None:
    counts = Counter(answer for answer in answers if answer is not None)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def summarize_generations(records: list[dict]) -> dict:
    """``records`` contain problem id, gold, prediction, and response index."""
    judgements = [answers_equal(row["prediction"], row["gold"]) for row in records]
    by_problem: dict[str, list[dict]] = {}
    for row in records:
        by_problem.setdefault(str(row["id"]), []).append(row)
    votes = []
    for rows in by_problem.values():
        chosen = majority_vote([row["prediction"] for row in rows])
        votes.append(answers_equal(chosen, rows[0]["gold"]))
    return {
        "pass_at_1": pass_at_1(judgements),
        "voting_accuracy": pass_at_1(votes),
        "n_responses": len(records),
        "n_problems": len(by_problem),
    }
