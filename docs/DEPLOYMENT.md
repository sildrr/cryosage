# Deployment Guide

## Storage, Accounts And Slurm

The web process submits controller jobs **as its own Linux user**. CryoSPARC
submits native processing Jobs as the account configured by its cluster lane.
These may be different Linux users. Do not assume `chmod` on the web directory
grants quota or read access to all original movies and metadata. Confirm shared
mounts, project write access and account quotas with the administrator.

Point `CRYOAGENT_WEB_DATA_ROOT` and `CRYOAGENT_WEB_LOG_ROOT` at shared HDD storage.
They contain uploads, task state, prompts, scientific evidence and exported
maps. Keep the Python environment and installed source on a filesystem readable
from every controller node. Restrict `CRYOAGENT_WEB_INPUT_ROOTS` to approved
colon-separated absolute input roots. Nested symlink escapes remain rejected.

The default `webapp/controller.sbatch` contains only generic resources (four
CPU cores, 64 GB, one node, seven days). `CRYOAGENT_WEB_SLURM_PARTITION`,
`CRYOAGENT_WEB_SLURM_NODELIST` and `CRYOAGENT_WEB_SLURM_QOS` become validated
`sbatch` flags. You may supply `CRYOAGENT_WEB_SLURM_SCRIPT` pointing at your own
script. It must preserve exported environment, use the installed interpreter,
and execute `python -u -m webapp.supervisor "$@"` from the installed runtime root.
The CLI exports `CRYOAGENT_REPO_ROOT` and `CRYOAGENT_WEB_PYTHON` for this purpose.

Set `CRYOAGENT_WEB_GPU_LANES` to existing CryoSPARC lane names, comma-separated.
Their templates must use `{{ num_gpu }}` in their GPU resource request rather
than a hard-coded GPU count. Native-job memory/CPU/QoS settings are owned by
those lane templates. This distribution does not edit the CryoSPARC database,
Slurm configuration, lane templates, drivers or GPU reservations.

## Model API And Proxy

Set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` and
`CRYOAGENT_FIXED_API_MODEL` in the private config. Model names and transport
compatibility are provider-specific. The runtime preserves the existing chat,
Responses-adapter, image-budget and retry implementations. A provider passing
a text health probe has not necessarily passed vision or tool-call validation.

Direct access requires no proxy configuration. If using a remote HTTP proxy,
set the usual `HTTP_PROXY`/`HTTPS_PROXY` to its compute-reachable address. If a
proxy is loopback-only on the login node, use the authenticated relay:

```text
HTTP_PROXY=http://127.0.0.1:18888
HTTPS_PROXY=http://127.0.0.1:18888
CRYOAGENT_WEB_PROXY_UPSTREAM=http://127.0.0.1:18888
CRYOAGENT_WEB_PROXY_HOST=your-login-node-internal-hostname
CRYOAGENT_WEB_PROXY_PORT=18889
CRYOAGENT_WEB_PROXY_ALLOWED_HOSTS=your-api-host.example
```

The relay starts with the managed web service when a loopback proxy is present.
It accepts only authenticated, allowlisted targets. Bind/firewall it to the
trusted compute network. Its token defaults to
`~/.config/cryosage/web-proxy.token`. Do not expose it as an open Internet proxy.
HTTP proxy credentials must remain in private config, never logs or Git.
CryoSPARC hostnames are bypassed separately; do not globally disable proxies.

## External Scientific Tools

Weights and external code are **not** shipped or downloaded by the installer.
Maintain separate environments to avoid TensorFlow/PyTorch dependency conflicts.

| Tool | Required Configuration | Behavior If Missing |
| --- | --- | --- |
| CryoSift | `CRYOAGENT_CRYOSIFT_PYTHON`, `CRYOAGENT_CRYOSIFT_RUNNER`, `CRYOAGENT_CRYOSIFT_WEIGHTS` | Full pilot quality gate cannot be claimed functional; workflow doctor fails |
| crYOLO | `CRYOAGENT_CRYOLO_PYTHON`, `CRYOAGENT_CRYOLO_SOURCE`, `CRYOAGENT_CRYOLO_WEIGHTS` | Records missing-tool evidence and falls back to visual scale estimation |
| ProSHADE | `CRYOAGENT_PROSHADE_BINARY` | Symmetry evidence unavailable; does not establish C1 as correct |
| KB | `CRYOAGENT_KB_ROOT`, or the compatible SQLite snapshot interface | Disable KB in the task until configured; never substitute invented retrieval |

CryoSift's runner contract is:

```bash
/envs/cryosift/bin/python /tools/CryoSift/cryosift_runner.py \
  --input /path/to/classes.mrc --output /path/to/evaluation --weights /path/to/weights.pth
```

It must emit the score format understood by `cryosparc_agent/cryosift_adapter.py`:
one class index and score per row in `score.txt`/supported score files. The
default runner convention is one-based, converted to CryoSPARC's zero-based
class IDs. Lower scores are better. Successful empty recommendations differ
from a failed tool invocation. Preserve source Job IDs when using scores.

The existing full KB bridge loads `mcp/tools/kb_tools.py` from
`CRYOAGENT_KB_ROOT`; its own database/assets must be configured in that tool.
The SQLite snapshot interface uses `CRYOAGENT_BENCHMARK_KB_SNAPSHOT` and the
bundled read-only `benchmarks/agent_vs_codex_20261017/kb_server.py`. It expects
the team's snapshot schema, not an arbitrary SQLite file. No original images
or maps are retrieved through that snapshot interface.

## CryoNet

The UI handoff submits only after explicit authorization to transmit the map
and FASTA. Install `requests` (included) and use standard CA verification.
`CRYOAGENT_CRYONET_CA_BUNDLE` can specify an administrator-approved custom CA
bundle. TLS verification is never disabled. The button records submission
status/job ID; it does not deploy local ModelAngelo or guarantee remote
completion/download. Other local building tools are not bundled here.

## Recovery And Operations

Use `cryosage-web resume TASK_ID` to restart a stopped frontend controller in
the same workspace. Existing native Jobs are adopted, not blindly duplicated.
Pilot states and downstream checkpoints are distinct artifacts. Preserve both.
Storage-failed native Jobs require inspecting/clearing incomplete outputs and
requeuing the affected Job after the administrator resolves the write/quota
problem; a controller restart alone cannot repair corrupt particle stacks.

Review Slurm/controller logs and CryoSPARC status separately. `df` free space
does not establish per-user quota headroom. This release does not automatically
delete caches, cancel native jobs, escalate privileges or change scientific
budgets when recovering. A claimed target FSC is not structural validation.
