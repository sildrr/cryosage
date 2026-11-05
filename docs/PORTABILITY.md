# Release-Only Adaptations

The production checkout and running controllers were not edited for packaging.
The copied source hashes are recorded in `SOURCE_SNAPSHOT.json`. Release files
differ only to support independent installation/configuration:

- Python packaging and `cryosage-web` command; private environment-file loader,
  configuration template and read-only deployment doctor.
- Storage/interpreter defaults use the installed environment and current user's
  directories instead of the original developers' paths. Production HDD paths
  must be supplied by the deploying user.
- Generic CPU Slurm controller script; configurable registered GPU lane names.
  The current dynamic 1..8-GPU enqueue policy is retained.
- Separate `~/.config/cryosage/` UI/relay tokens and runtime profile, avoiding
  collision with an existing development deployment on the same account.
- Managed start creates a proxy relay only when the configured upstream proxy
  is loopback-only. Additional provider hosts can be explicitly allowlisted.
- External crYOLO/CryoSift/KB defaults no longer point at another user's tool
  environment. The tools, weights and KB are separately provisioned.
- CryoNet uses the requests/system CA bundle by default rather than a developer's
  local certificate filename; a custom CA path remains configurable.

Scientific prompts, planning options, visual evidence extraction, score
direction, pilot gates, parameter preferences, replay budgets and completion
criteria are copied from the production version. Historical unsupported or
research-only modules remain support source, not new public CLI commands.

No end-to-end processing run, paid model call, real reconstruction or downstream
upload is performed by packaging tests. Provisioned installations still need
a bounded real workflow validation before scientific use.
