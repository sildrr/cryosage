# Multiple particle STAR inputs

Change date: 2026-11-12.

Some datasets have multiple particle STAR files referring to the same set of
MRCS stacks. The web form accepts multiple absolute STAR paths separated by
commas, or multiple uploaded STAR files. Each path and stack reference is
validated before submission. The worker creates a task-local combined STAR for
CryoSPARC's particle import; the source files are left intact.

Multiple files must have identical optics tables and particle columns, and use
the supported `data_optics`/`data_particles` loop layout. Incompatible files are
rejected instead of silently changing acquisition metadata. Particle rows are
concatenated as supplied, so overlapping inputs can contain duplicate particles.
Single-STAR workflows continue to use the original file directly.
