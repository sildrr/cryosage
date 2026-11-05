"""Scoped, opt-in experiment facts; never modifies candidates or permissions."""
import json
import os
from pathlib import Path


def planner_context(observation):
    path=os.getenv('CRYOAGENT_COVERAGE_CONTEXT')
    if not path:
        return None
    value=json.loads(Path(path).read_text())
    if (value.get('project_uid'),value.get('workspace_uid')) != (
            observation.get('project_uid'),observation.get('workspace_uid')):
        return None
    return value
