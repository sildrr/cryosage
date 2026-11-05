# CryoSage

Cryo-EM Scientific Agent for Guided Exploration. This repository packages the
production API-backed Agent and private web console as of **2026-11-05**.
It connects planning, visual observation, knowledge retrieval and execution to
CryoSPARC, including pilot exploration, full-data processing, refinement and
an explicitly authorized CryoNet model-building handoff.

This is software for an existing cryo-EM computing deployment, not a bundled
CryoSPARC installation or a public multi-tenant SaaS. A frontend-only preview
can run without CryoSPARC; real processing needs the prerequisites below.

## Quick Install

Use Linux, Python 3.11 or 3.12, and a writable checkout on shared storage visible
to the login and controller nodes. Do not install into another user's environment.

```bash
git clone git@github.com:sildrr/cryosage.git
cd cryosage
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c requirements-tested.txt -e '.[test]'
cryosage-web init-config
```

Edit `~/.config/cryosage/deployment.env`, filling in your own values from the
template. The command creates it with mode 600. Keep credentials out of the
repository. At minimum configure:

- Model API base URL, key and exact planner/executor model IDs.
- CryoSPARC hostname, base port, account email/password and writable project UID.
- Absolute shared HDD data/log directories and narrowly scoped input roots.
- CPU Slurm partition for controllers, and existing CryoSPARC GPU lane names.
- CryoSift interpreter, runner and weights for the full production pilot gate.

```bash
cryosage-web doctor
cryosage-web doctor --workflow --online
cryosage-web start
cryosage-web status
```

`doctor` never submits a Job or calls a paid model. `--online` only checks the
configured CryoSPARC project and registered lane names. It does not prove that
the service account has sufficient disk quota, that queues can start Jobs, or
that a model provider supports vision/tools. Validate one bounded real workflow
after provisioning, with explicit authorization for API/image transmission.

The UI binds only to `127.0.0.1:8777`. Open it through an SSH tunnel:

```bash
ssh -N -L 8777:127.0.0.1:8777 YOUR_USER@YOUR_LOGIN_HOST
```

Visit `http://127.0.0.1:8777`. The first start creates the browser access token
in `~/.config/cryosage/web.token`; read that file locally and enter it in the UI.
This is not your model API key or CryoSPARC password.

To preview only the web interface, use `cryosage-web serve` after configuration;
this does not require a scheduler until a real task is submitted. Do not submit
tasks until `doctor --workflow --online` succeeds and your provider is verified.

## Real Workflow Prerequisites

1. Licensed, administrator-maintained CryoSPARC service and GPU workers with
   registered Slurm lanes. This release was developed against cryosparc-tools 5.0.3.
2. `sbatch`, `squeue`, `sinfo`, `scontrol` and `scancel` available to the submitting
   user. The controller Python environment and repository must be compute-visible.
3. Original movies/micrographs, or MRCS stacks plus STAR metadata, readable by both
   the controller user and CryoSPARC execution account. The latter also needs
   writable project storage and adequate HDD/SSD quotas.
4. An OpenAI-compatible provider supporting image messages and tool calls. Supply
   exact provider-specific model names; some gateways require an `openai/` prefix.
5. External CryoSift code/environment/weights for pilot class-quality assessment.
   crYOLO size initialization and ProSHADE symmetry evidence are separately
   configured tools, not bundled dependencies. See [deployment](docs/DEPLOYMENT.md).

## Execution And Resource Policy

The CPU-only Slurm controller calls the model API and MCP tools. CryoSPARC queues
its own GPU Jobs separately; controller QoS does not configure native GPU lane QoS.
For supported native Jobs (Motion Correction, Patch CTF, Extract, Class 2D), the
Agent queries Slurm just before enqueueing and requests 1..8 free GPUs on a single
usable node. Set `CRYOAGENT_WEB_MAX_GPUS` to lower the cap. Blob/Template Picker,
Ab-initio and Homo/NU-refine remain native single-GPU operations. CPU/interactive
jobs keep their existing queue behavior. Slurm remains authoritative: measured
free capacity is not a reservation and does not guarantee immediate execution.

New web workflows use the production pilot-to-full strategy, including visual
Inspect/Extract review, objective class-quality checks, class/view recovery,
template comparisons and bounded downstream planning. User controls work at
safe decision boundaries. Each task has a 100-created-Job default budget,
including pilot, failed and optimization jobs; imported particles bypass picking.
Leaving symmetry blank means unknown, not a supplied known C1 point group.

```bash
cryosage-web resume TASK_ID
cryosage-web stop
```

Stopping the web server does not stop submitted Slurm controllers or GPU Jobs.
Resume refuses a duplicate active controller. Inspect ambiguous creation errors
before retrying: a remote Job may already exist. Never erase a budget/creation
marker without reconciling actual CryoSPARC Job IDs.

## Included / Excluded

Included: current frontend/static showcase, API and MCP runner, planning and
observation logic, pilot/full strategies, optional-tool adapters, deployment CLI,
Slurm controller template and source tests. The homepage contains historical
public-dataset examples, not promised performance on new data.

Not included: credentials, live tasks/logs, experiment data, maps, model weights,
third-party environments/source distributions, licensed CryoSPARC components or
the team's knowledge-base database. Historical ablation/RL collection modules
may be present as support code; they are not production deployment commands.

`SOURCE_SNAPSHOT.json` records the exact production source snapshot hashes.
`docs/PORTABILITY.md` describes release-only changes; the scientific decision
prompts and processing strategies are not redesigned for distribution.

## Tests

```bash
python -m pytest release_tests -q
PYTHONPATH="$PWD:$PWD/cryosparc_agent" python -m pytest webapp/tests -q
```

Release verification and limitations are recorded in `VALIDATION.md`. Offline
packaging checks are not an end-to-end reconstruction benchmark.

## Security And Distribution

Use only with trusted collaborators behind SSH or a protected reverse proxy.
The prototype has shared-account authorization, no tenant isolation, no public
upload quota and no hardened multi-user deployment boundary. Model evidence is
sent to the configured provider only after the workflow consent step. CryoNet
uploads additionally require explicit authorization for map/sequence transfer.
Keep logs private: they can contain scientific images, reasoning and parameters.

No open-source license has been selected for this private source release.
Third-party tools and weights retain their own licenses; obtain them separately.
Contact: lisongya21@mails.tsinghua.edu.cn.
