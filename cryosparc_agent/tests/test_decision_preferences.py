import ast
import os
from pathlib import Path
from selection_safety import SELECTION_SAFETY_GUIDANCE

from visual_decision_contract import (
    EXTRACT_BOX_PREFERENCE_GUIDANCE,
    REFINEMENT_PREFERENCE_GUIDANCE,
    build_stage_decision_messages,
)


def test_extract_backend_neutral_prompt_includes_diameter_preference():
    messages = build_stage_decision_messages("extract_box_size", {})
    text = messages[1]["content"][0]["text"]
    assert EXTRACT_BOX_PREFERENCE_GUIDANCE in text
    assert "not a hard constraint" in text


def test_executor_static_prompt_shares_preferences_without_importing_runtime():
    # Evaluate just the static prompt: no MCP sessions or model/API initialization.
    source = Path(__file__).resolve().parents[1] / "autonomous_mcp_closed_loop.py"
    tree = ast.parse(source.read_text())
    assignment = next(
        node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "STATIC_DECISION_INSTRUCTIONS"
                for t in node.targets)
    )
    prompt = eval(compile(ast.Expression(assignment.value), str(source), "eval"), {
        "CRYOSIFT_SCORE_GUIDANCE": "", "CRYOSIFT_EMPTY_GUIDANCE": "",
        "SELECTION_SAFETY_GUIDANCE": SELECTION_SAFETY_GUIDANCE,
        "REFINEMENT_PREFERENCE_GUIDANCE": REFINEMENT_PREFERENCE_GUIDANCE,
        "EXTRACT_BOX_PREFERENCE_GUIDANCE": EXTRACT_BOX_PREFERENCE_GUIDANCE,
        "os": os,
    })["instruction"]
    assert REFINEMENT_PREFERENCE_GUIDANCE in prompt
    assert EXTRACT_BOX_PREFERENCE_GUIDANCE in prompt
    assert SELECTION_SAFETY_GUIDANCE in prompt
    assert "prefer reclassifying that branch" not in prompt
    assert "recommended_box_size_pix as the next extraction candidate" not in prompt
