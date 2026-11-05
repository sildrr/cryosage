"""Durable human interventions consumed only at safe decision boundaries."""
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


async def decision_boundary(directory, current_node, validate_target):
    if not directory:
        return {"current_node": current_node, "guidance": None, "stop": False}
    root = Path(directory)
    state_file = root / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"paused": False, "guidance": None}
    (root / "processing").mkdir(parents=True, exist_ok=True)
    while True:
        # Revisit interrupted claims on checkpoint resume; receipt files prevent replay.
        files = sorted((root / "processing").glob("*.json")) + sorted((root / "inbox").glob("*.json"))
        for file in files:
            command = json.loads(file.read_text())
            receipt_file = root / "receipts" / file.name
            if receipt_file.exists():
                file.unlink(missing_ok=True)
                continue
            claimed = root / "processing" / file.name
            if file.parent.name == "inbox":
                file.replace(claimed)
            receipt = {"id": command["id"], "action": command["action"],
                       "status": "applied", "applied_at": datetime.now(timezone.utc).isoformat()}
            try:
                action = command["action"]
                if action == "guide":
                    target = command.get("target_node")
                    if target:
                        await validate_target(target)
                        current_node = target
                    state["guidance"] = {"id": command["id"], "text": command["message"], "target_node": target}
                    state["current_node"] = current_node
                elif action == "pause":
                    state["paused"] = True
                elif action == "resume":
                    if command.get("target_node"):
                        await validate_target(command["target_node"])
                        current_node = command["target_node"]
                    state["paused"] = False
                    state["stop"] = False
                    if command.get("message"):
                        state["guidance"] = {"id": command["id"], "text": command["message"],
                                             "target_node": command.get("target_node")}
                elif action == "stop":
                    state["stop"] = True
                    state["paused"] = False
                else:
                    raise ValueError("Unknown intervention action")
            except Exception as exc:
                receipt.update(status="rejected", message=str(exc))
            save(state_file, state)
            save(receipt_file, receipt)
            claimed.unlink(missing_ok=True)
        if state.get("stop") or not state.get("paused"):
            return {"current_node": current_node, "guidance": state.get("guidance"), "stop": bool(state.get("stop"))}
        await asyncio.sleep(2)
