# Release Validation

Version: 2026.11.5. Date: 2026-11-05 (Asia/Shanghai).

## Completed

- Installed the project and declared dependencies into a **new isolated Python
  3.11 virtual environment**, without inheriting production site-packages.
- `pip check`: no broken requirements.
- Frontend and release tests passed: **241 tests and four subtests** (including
  seven release-specific tests). The suite uses mocks for native Job submission.
- Started a real loopback web server on an isolated temporary port, data root
  and token file. Verified HTML, health, four showcase cases, rejection of an
  unauthenticated task-list request, and an authenticated empty task list.
- Built a wheel and installed it into a separate target. From outside the source
  checkout, verified installed package resolution, deployment templates, UI
  assets, MCP imports and read-only dependency doctor.
- Verified `autonomous_mcp_closed_loop.py --help` and MCP server import without
  requiring local model weights or production credentials.
- Scanned the allowlisted source snapshot for token/private-key patterns; Git
  publication receives a separate final staged-file scan.

The tested dependency set is pinned in `requirements-tested.txt`. In particular,
the isolated environment resolved mcp 1.30.0, cryosparc-tools 5.0.3, pydantic
2.14.2, numpy 2.4.6 and Pillow 13.0.0. These packaging-test dependencies differ
from some production environment versions; a real workflow on the destination
cluster remains required. Python 3.12 is allowed by metadata but was not tested.

## Not Claimed

No real CryoSPARC Job was submitted, no paid model request was sent and no map
or sequence was uploaded during release validation. External CryoSift/crYOLO
weights, KB content, ProSHADE and licensed CryoSPARC are not included or installed
by pip. A successful offline doctor is not end-to-end reconstruction validation.

The old production tests that expected a developer-specific controller path or
the `.config/cryoagent` development profile were adapted to this release's
portable template and isolated `.config/cryosage` profile. Three queue mocks now
explicitly provide the native `status` attribute. Showcase provenance metadata
was included so the homepage consistency tests can validate exported assets.
No scientific expected values were changed to make these deployment tests pass.

No live service, existing controller, runtime profile, credentials, scientific
data or GPU task was changed while preparing this release.
