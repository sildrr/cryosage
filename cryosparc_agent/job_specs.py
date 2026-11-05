# Defines lightweight job metadata used by the generic CryoSPARC job executor.
from copy import deepcopy
from typing import Any
import os


HIGH_GPU_APPROVAL_THRESHOLD = 4
MAX_EXTRACT_BOX_SIZE_PIX = 720
DEFAULT_GPU_LANE = os.getenv('CRYOAGENT_WEB_GPU_LANES', 'gpu_slurm').split(',')[0].strip()
EXTRACT_SAMPLING_GUIDANCE = (
    "bin_size_pix is the absolute OUTPUT EDGE LENGTH in pixels after Fourier cropping, "
    "NOT a downsampling factor. With box_size_pix=384, bin_size_pix=2 produces 2x2 "
    "particle images, not two-fold downsampling; two-fold downsampling would output 192x192. "
    "For full-resolution reconstruction and optimization, prefer no Fourier cropping: "
    "leave this optional parameter unset/blank, or use the same output edge length as "
    "box_size_pix when an explicit size is required. Never use 1 or 2 as a no-binning flag. "
    "A smaller output edge increases pixel size and worsens the Nyquist limit, potentially "
    "discarding information needed for the target resolution. Avoid downsampling during "
    "optimization unless explicitly justified; reducing GPU cost alone is not evidence "
    "that the final resolution will be preserved."
)


def annotate_extraction_parameters(job_type, template):
    """Add physical meaning without replacing live Registry constraints/defaults."""
    if job_type in {'extract_micrographs_multi', 'extract_micrographs_cpu_parallel'}:
        if 'bin_size_pix' in template:
            template['bin_size_pix']['title'] = 'Fourier-crop to output box size (pixels)'
            template['bin_size_pix']['description'] = EXTRACT_SAMPLING_GUIDANCE
    return template


# Parameter rules are intentionally small: only expose knobs the model may edit.
JOB_SPECS: dict[str, dict[str, Any]] = {
    'agent_reference_union': {
        'category': 'curation', 'requires_gpu': False, 'requires_approval': False,
        'interactive': False, 'parameter_template': {},
    },
    'agent_scale_cleanup': {
        'category': 'curation', 'requires_gpu': False, 'requires_approval': False,
        'interactive': False,
        'parameter_template': {'quantile': {'type': 'number', 'const': .05, 'default': .05}},
    },
    'volume_tools': {
        'category': 'volume_processing', 'requires_gpu': False, 'requires_approval': False,
        'interactive': False,
        'parameter_template': {'vol_lpf_A': {'type': 'number'}, 'vol_resample_N': {'type': 'integer'}},
    },
    'reference_select_2D': {
        'category': 'curation', 'requires_gpu': True, 'requires_approval': False,
        'interactive': False, 'default_lane': DEFAULT_GPU_LANE,
        'parameter_template': {'select_mode': {'type': 'string'}, 'res_filter': {'type': 'number'},
            'max_res': {'type': 'number'}, 'lp_filter': {'type': 'boolean'}},
    },
    "import_movies": {
        "category": "import",
        "requires_gpu": False,
        "requires_approval": False,
        "interactive": False,
        "parameter_template": {
            "blob_paths": {"type": "string", "required": True},
            "gainref_path": {"type": "string"},
            "psize_A": {"type": "number", "minimum": 0},
            "accel_kv": {"type": "number", "minimum": 0},
            "cs_mm": {"type": "number", "minimum": 0},
            "total_dose_e_per_A2": {"type": "number", "minimum": 0},
        },
    },
    "import_micrographs": {
        "category": "import",
        "requires_gpu": False,
        "requires_approval": False,
        "interactive": False,
        "parameter_template": {
            "blob_paths": {"type": "string", "required": True},
            "psize_A": {"type": "number", "minimum": 0},
            "accel_kv": {"type": "number", "minimum": 0},
            "cs_mm": {"type": "number", "minimum": 0},
            "total_dose_e_per_A2": {"type": "number", "minimum": 0},
        },
    },
    "patch_motion_correction_multi": {
        "category": "motion_correction",
        "requires_gpu": True,
        "requires_approval": False,
        "interactive": False,
        "default_lane": DEFAULT_GPU_LANE,
        "parameter_template": {
            "compute_num_gpus": {"type": "integer", "minimum": 1, "maximum": 8},
        },
    },
    "patch_ctf_estimation_multi": {
        "category": "ctf",
        "requires_gpu": True,
        "requires_approval": False,
        "interactive": False,
        "default_lane": DEFAULT_GPU_LANE,
        "parameter_template": {
            "compute_num_gpus": {"type": "integer", "minimum": 1, "maximum": 8},
            "df_search_min": {"type": "number", "minimum": 0},
            "df_search_max": {"type": "number", "minimum": 0},
        },
    },
    "curate_exposures_v2": {
        "category": "curation",
        "requires_gpu": False,
        "requires_approval": True,
        "interactive": True,
        "parameter_template": {},
    },
    "blob_picker_gpu": {
        "category": "picking",
        "requires_gpu": True,
        "requires_approval": False,
        "interactive": False,
        "default_lane": DEFAULT_GPU_LANE,
        "parameter_template": {
            "diameter": {"type": "number", "minimum": 0},
            "diameter_max": {"type": "number", "minimum": 0},
            "num_process": {"type": "integer", "minimum": 1},
        },
    },
    "template_picker_gpu": {
        "category": "picking",
        "requires_gpu": True,
        "requires_approval": False,
        "interactive": False,
        "default_lane": DEFAULT_GPU_LANE,
        "parameter_template": {
            "diameter": {"type": "number", "minimum": 0},
            # CryoSPARC multiplies this value by ``diameter`` to obtain the
            # physical exclusion distance. It is not an Angstrom value.
            "min_distance": {
                "type": "number",
                "minimum": 0,
                "description": "Minimum separation as a multiplier of particle diameter, not Angstroms.",
            },
            "use_ctf": {"type": "boolean"},
        },
    },
    "create_templates": {
        "category": "template_generation",
        "requires_gpu": False,
        "requires_approval": False,
        "interactive": False,
        "parameter_template": {
            "n_templates": {"type": "integer", "minimum": 1},
            "angular_step": {"type": "number", "minimum": 0},
            "max_tilt": {"type": "number", "minimum": 0},
        },
    },
    "inspect_picks_v2": {
        "category": "inspection",
        "requires_gpu": False,
        "requires_approval": False,
        "interactive": False,
        # CryoSPARC labels this Job interactive, but explicit score thresholds
        # make it autonomously executable through queue() without a lane.
        "queue_without_lane": True,
        "parameter_template": {
            "n_mic_to_plot": {"type": "integer", "minimum": 1},
            "checkpoint_freq": {"type": "integer", "minimum": 1},
            "calibrate_ncc": {"type": "boolean"},
            "calibrate_pow": {"type": "boolean"},
            "ncc_score_thresh": {"type": "number"},
            "lpower_thresh_min": {"type": "number"},
            "lpower_thresh_max": {"type": "number"},
            "curv_thresh": {"type": "number"},
            "sinu_thresh": {"type": "number"},
            "do_auto_cluster": {"type": "boolean"},
            "lpower_target": {"type": "number"},
            "dilation_bins": {"type": "integer", "minimum": 0},
            "keep_threshold": {"type": "number"},
        },
    },
    "junk_detector_v1": {
        "category": "exposure_curation",
        "requires_gpu": True,
        "requires_approval": False,
        "interactive": False,
        "default_lane": DEFAULT_GPU_LANE,
        "parameter_template": {
            "min_dist": {"type": "number", "minimum": 0},
            "enable_cat1": {"type": "boolean"},
            "min_dist_cat1": {"type": "number", "minimum": 0},
            "enable_cat2": {"type": "boolean"},
            "min_dist_cat2": {"type": "number", "minimum": 0},
            "enable_cat3": {"type": "boolean"},
            "min_dist_cat3": {"type": "number", "minimum": 0},
            "min_label_area": {"type": "number", "minimum": 0},
        },
    },
    "extract_micrographs_multi": {
        "category": "extraction",
        "requires_gpu": True,
        "requires_approval": False,
        "interactive": False,
        "default_lane": DEFAULT_GPU_LANE,
        "parameter_template": {
            "compute_num_gpus": {"type": "integer", "minimum": 1, "maximum": 8},
            "box_size_pix": {"type": "integer", "minimum": 32, "maximum": MAX_EXTRACT_BOX_SIZE_PIX},
            "bin_size_pix": {"type": "integer", "minimum": 2},
            "num_extract": {"type": "integer", "minimum": 1},
        },
    },
    "extract_micrographs_cpu_parallel": {
        "category": "extraction",
        "requires_gpu": False,
        "requires_approval": False,
        "interactive": False,
        "default_lane": DEFAULT_GPU_LANE,
        "parameter_template": {
            "box_size_pix": {"type": "integer", "minimum": 32, "maximum": MAX_EXTRACT_BOX_SIZE_PIX},
            "bin_size_pix": {"type": "integer", "minimum": 2},
            "num_extract": {"type": "integer", "minimum": 1},
            "compute_num_cores": {"type": "integer", "minimum": 1},
        },
    },
    "class_2D_new": {
        "category": "classification",
        "requires_gpu": True,
        "requires_approval": False,
        "interactive": False,
        "default_lane": DEFAULT_GPU_LANE,
        "parameter_template": {
            "compute_num_gpus": {"type": "integer", "minimum": 1, "maximum": 8},
            "class2D_K": {"type": "integer", "minimum": 2},
        },
    },
    "select_2D": {
        "category": "selection",
        "requires_gpu": False,
        # CryoSPARC exposes Select 2D as interactive, but the agent supplies
        # an explicit class list from visual evidence. Submit it without a
        # compute lane, just as the autonomous Inspect Picks path does.
        "requires_approval": False,
        "interactive": False,
        "queue_without_lane": True,
        "parameter_template": {
            # CryoSPARC expects a comma-separated class-index string.
            "selected_templates": {
                "type": "string",
                "required": True,
                "description": "Comma-separated Class 2D IDs selected from visual and CryoSift evidence.",
            },
            "resolution_better_than": {"type": "number", "minimum": 0},
            "particle_count_above": {"type": "integer", "minimum": 0},
        },
    },
    "homo_abinit": {
        "category": "reconstruction",
        "requires_gpu": True,
        "requires_approval": False,
        "interactive": False,
        "default_lane": DEFAULT_GPU_LANE,
        "parameter_template": {
            "abinit_symmetry": {"type": "string"},
        },
    },
    "homo_refine_new": {
        "category": "refinement",
        "requires_gpu": True,
        "requires_approval": False,
        "interactive": False,
        "default_lane": DEFAULT_GPU_LANE,
        "parameter_template": {
            "compute_use_ssd": {"type": "boolean"},
            "refine_symmetry": {"type": "string"},
            "refine_defocus_refine": {"type": "boolean"},
            "refine_ctf_global_refine": {"type": "boolean"},
        },
    },
    "nonuniform_refine_new": {
        "category": "refinement",
        "requires_gpu": True,
        "requires_approval": False,
        "interactive": False,
        "default_lane": DEFAULT_GPU_LANE,
        "parameter_template": {
            "compute_use_ssd": {"type": "boolean"},
            "refine_symmetry": {"type": "string"},
            "refine_defocus_refine": {"type": "boolean"},
            "refine_ctf_global_refine": {"type": "boolean"},
        },
    },
}


def get_job_spec(job_type: str) -> dict[str, Any]:
    """Return metadata for a job type, using safe defaults for unknown jobs."""
    spec = deepcopy(JOB_SPECS.get(job_type, {}))
    spec.setdefault("category", "unknown")
    spec.setdefault("requires_gpu", False)
    spec.setdefault("requires_approval", True)
    spec.setdefault("interactive", False)
    spec.setdefault("queue_without_lane", False)
    # CryoSPARC requires a queue lane even when this Job itself requests no GPU.
    spec.setdefault("default_lane", DEFAULT_GPU_LANE)
    spec.setdefault("max_auto_gpus", HIGH_GPU_APPROVAL_THRESHOLD)
    spec.setdefault("parameter_template", {})
    if job_type in {'homo_refine_new', 'nonuniform_refine_new'}:
        for name in ('refine_nu_enable', 'refine_do_marg', 'refine_gs_resplit',
                     'crg_do_tilt', 'crg_do_trefoil', 'crg_do_spherical',
                     'crg_do_tetrafoil', 'crg_do_anisomag'):
            spec['parameter_template'].setdefault(name, {'type': 'boolean'})
        for name in ('refine_res_init', 'crl_df_range', 'crl_min_res_A', 'crg_min_res_A'):
            spec['parameter_template'].setdefault(name, {'type': 'number', 'minimum': 0})
        spec['parameter_template'].setdefault('refine_num_final_iterations', {'type': 'integer', 'minimum': 0})
    return spec


def get_parameter_template(job_type: str) -> dict[str, dict[str, Any]]:
    """Return the editable parameter template for a job type."""
    template = deepcopy(get_job_spec(job_type)["parameter_template"])
    if job_type == 'inspect_picks_v2':
        from inspect_score_contract import raw_parameter_template
        template = raw_parameter_template(template)
    return annotate_extraction_parameters(job_type, template)


def list_supported_job_types() -> list[str]:
    """Return job types with explicit local metadata."""
    return sorted(JOB_SPECS)
