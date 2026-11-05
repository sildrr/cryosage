"""Plan three additional fixed-subset collection rounds without submitting jobs."""

import copy
from pathlib import Path

from visual_rl_collection import read_json, validate_collection_config, write_json


ROOT = Path("/hdd1/linweifan/cryosparc/cryoagent_visual_rl")
BASELINE = "visual-rl-100-collection-20261009"
EXPERIMENT = "visual-rl-100-exploration-20261009"


def prepare():
    source = ROOT / "runs/preparation" / BASELINE
    target = ROOT / "runs/preparation" / EXPERIMENT
    config = copy.deepcopy(read_json(source / "collection_config.json"))
    baseline = read_json(source / "batch_manifest.json")
    rows = baseline["trials"]
    if len(rows) != 100 or len({row["dataset_id"] for row in rows}) != 100:
        raise ValueError("Expected 100 unique prepared datasets")
    config["experiment_id"] = EXPERIMENT
    config["limits"].update(max_trials_per_dataset=3, max_trials_total=300)
    config["exploration"] = {
        "feedback_experiment_ids": [BASELINE],
        "baseline_experiment_id": BASELINE,
        "additional_rounds": 3,
        "guidance": (
            "This is fixed-subset parameter exploration, not weight optimization. "
            "Compare same_dataset_prior_feedback and previous_stage_parameters. "
            "Prefer a meaningful, evidence-supported alternative inside search_space "
            "to an already tried parameter setting. Explore diameter ranges, score "
            "thresholds, and extraction box sizes while preserving complete particles. "
            "Do not maximize retained particle count alone or change values arbitrarily. "
            "Repeated settings are allowed if the evidence supports them; identical "
            "settings are replicates, not distinct parameter strategies. Only compare "
            "quality metrics within the same dataset and fixed micrograph subset."
        ),
    }
    config = validate_collection_config(config)
    config_path = target / "collection_config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Existing exploration config differs; use a new experiment ID")
    write_json(config_path, config)
    for round_index in range(1, 4):
        manifest = {
            "batch_id": f"{EXPERIMENT}-round-{round_index}",
            "baseline_batch_id": baseline["batch_id"],
            "exploration_round": round_index,
            "trials": [
                {key: value for key, value in row.items() if key != "trial_id"}
                for row in rows
            ],
        }
        path = target / f"round_{round_index}_manifest.json"
        if path.exists() and read_json(path) != manifest:
            raise ValueError(f"Existing manifest differs: {path}")
        write_json(path, manifest)
    print(f"Prepared 3 x 100 additional trials at {target}", flush=True)


if __name__ == "__main__":
    prepare()
