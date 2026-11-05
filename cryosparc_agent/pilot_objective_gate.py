"""Reference-free, deterministic pilot promotion gate; no model usability vote."""
from dataclasses import asdict, dataclass
import math

import numpy as np
from PIL import Image

VERSION = 'pilot-objective-v2'


@dataclass(frozen=True)
class GatePolicy:
    cryosift_threshold: float = 3.0
    min_particles: int = 2000
    min_groups: int = 3
    min_group_fraction: float = .05
    duplicate_ncc: float = .85
    image_size: int = 64
    rotation_step_degrees: int = 15
    max_shift_fraction: float = .10
    compare_mirrors: bool = True

    def __post_init__(self):
        for name in ('cryosift_threshold', 'min_group_fraction', 'duplicate_ncc', 'max_shift_fraction'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError('Non-finite gate policy: ' + name)
        if self.cryosift_threshold <= 0 or not 0 < self.duplicate_ncc <= 1:
            raise ValueError('Invalid quality or similarity threshold')
        if not 0 < self.min_group_fraction <= 1 or not 0 <= self.max_shift_fraction <= .25:
            raise ValueError('Invalid group weight or translation range')
        if type(self.min_particles) is not int or self.min_particles <= 0:
            raise ValueError('Positive integer particle requirement needed')
        if type(self.min_groups) is not int or self.min_groups < 2:
            raise ValueError('At least two supported groups required')
        if self.min_groups*self.min_group_fraction > 1+1e-12:
            raise ValueError('Group support requirements are mathematically impossible')
        if type(self.image_size) is not int or not 32 <= self.image_size <= 128 or self.image_size % 2:
            raise ValueError('Even image size in 32..128 needed')
        if type(self.rotation_step_degrees) is not int or not 1 <= self.rotation_step_degrees <= 90 or 360 % self.rotation_step_degrees:
            raise ValueError('Rotation step must divide 360')
        if type(self.compare_mirrors) is not bool:
            raise ValueError('Mirror option must be boolean')


def _unit(image):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError('Non-finite or non-2D class image')
    y, x = np.indices(image.shape)
    radius = np.sqrt((x-(image.shape[1]-1)/2)**2 + (y-(image.shape[0]-1)/2)**2)
    scale = min(image.shape)
    mask = np.clip((.48*scale-radius)/(.06*scale), 0, 1)
    values = (image - np.sum(image*mask)/mask.sum()) * mask
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError('Flat class image has no shape evidence')
    return values/norm


def pairwise_shape_ncc(images, policy=GatePolicy()):
    """Maximum padded FFT NCC over in-plane rotation, small shifts and mirrors.

    It intentionally groups contrast-inverted/mirrored duplicates conservatively.
    This measures 2D shape differences, not actual 3D orientations.
    """
    prepared = []
    for image in images:
        array = np.asarray(image, dtype=np.float32)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError('Invalid class image')
        small = Image.fromarray(array).resize((policy.image_size, policy.image_size), Image.Resampling.BILINEAR)
        prepared.append(_unit(small))
    size = policy.image_size
    matrix = np.eye(len(prepared))
    if len(prepared) < 2:
        return matrix
    transforms = np.fft.fft2(np.stack(prepared), s=(2*size, 2*size), axes=(-2, -1))
    shift = round(policy.max_shift_fraction*size)
    offsets = np.arange(-shift, shift+1) % (2*size)
    for j in range(1, len(prepared)):
        best = np.zeros(j)
        variants = [prepared[j]]
        if policy.compare_mirrors:
            variants.append(np.fliplr(prepared[j]))
        for variant in variants:
            pil = Image.fromarray(variant.astype(np.float32))
            for angle in range(0, 360, policy.rotation_step_degrees):
                rotated = _unit(pil.rotate(angle, resample=Image.Resampling.BILINEAR))
                transform = np.fft.fft2(rotated, s=(2*size, 2*size))
                correlations = np.fft.ifft2(transforms[:j]*transform.conj(), axes=(-2, -1)).real
                window = correlations[:, offsets[:, None], offsets]
                best = np.maximum(best, np.abs(window).max(axis=(1, 2)))
        matrix[:j, j] = matrix[j, :j] = np.clip(best, 0, 1)
    return matrix


def shape_groups(matrix, threshold):
    """Complete-link grouping avoids chaining unrelated shapes through a bridge."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError('Finite square similarity matrix required')
    if not np.allclose(matrix, matrix.T) or not np.allclose(np.diag(matrix), 1):
        raise ValueError('Similarity matrix must be symmetric with unit diagonal')
    if np.any(matrix < 0) or np.any(matrix > 1+1e-8):
        raise ValueError('NCC similarities must be in 0..1')
    groups = [[i] for i in range(len(matrix))]
    while len(groups) > 1:
        choices = [(float(matrix[np.ix_(a, b)].min()), i, j)
                   for i, a in enumerate(groups) for j, b in enumerate(groups) if i < j]
        value, i, j = max(choices, key=lambda v: (v[0], -v[1], -v[2]))
        if value < threshold:
            break
        groups[i] = sorted(groups[i] + groups[j])
        groups.pop(j)
    return groups


def gate_from_matrix(ids, counts, scores, matrix, policy=GatePolicy()):
    """All supplied IDs must already pass the strict CryoSift quality filter."""
    if len(set(ids)) != len(ids) or np.shape(matrix) != (len(ids), len(ids)):
        raise ValueError('Class ID/matrix mismatch')
    if any(type(counts.get(i)) is not int or counts[i] <= 0 for i in ids):
        raise ValueError('Positive native particle counts required')
    if any(i not in scores or not math.isfinite(scores[i]) or scores[i] >= policy.cryosift_threshold for i in ids):
        raise ValueError('Only CryoSift-qualified classes enter diversity calculation')
    total = sum(counts[i] for i in ids)
    groups = []
    for members in shape_groups(matrix, policy.duplicate_ncc):
        group_ids = [ids[i] for i in members]
        number = sum(counts[i] for i in group_ids)
        fraction = number/total if total else 0.0
        groups.append(dict(class_ids=group_ids, particles=number, particle_fraction=fraction,
                           supported=fraction >= policy.min_group_fraction))
    supported = sum(g['supported'] for g in groups)
    probabilities = [g['particle_fraction'] for g in groups]
    effective = math.exp(-sum(p*math.log(p) for p in probabilities if p)) if probabilities else 0.0
    failed = []
    if total < policy.min_particles:
        failed.append('insufficient_cryosift_qualified_particles')
    if supported < policy.min_groups:
        failed.append('insufficient_supported_2d_shape_groups')
    return dict(metric_version=VERSION, policy=asdict(policy), available=True, passed=not failed,
        failed_conditions=failed, promotion_class_ids=list(ids), qualified_particles=total,
        particle_weighted_score=sum(scores[i]*counts[i] for i in ids)/total if total else None,
        supported_group_count=supported, effective_group_count=effective, groups=groups,
        similarity_class_ids=list(ids), pairwise_shape_ncc=np.asarray(matrix).tolist(),
        diversity_kind='reference_free_2d_shape_proxy_not_3d_angular_coverage')


def evaluate_gate(context, images, policy=GatePolicy()):
    source = context.get('source') or {}
    sift = context.get('cryosift') or {}
    if not sift.get('success') or sift.get('job_uid') != source.get('job_uid') or sift.get('project_uid') != source.get('project_uid'):
        return dict(metric_version=VERSION, policy=asdict(policy), available=False, passed=False,
                    failed_conditions=['missing_or_wrong_source_cryosift'])
    records = context.get('class_statistics') or []
    if len({r['class_id'] for r in records}) != len(records):
        raise ValueError('Duplicate class population records')
    counts = {r['class_id']: r.get('particle_count', 0) for r in records}
    scores = {}
    for row in sift.get('class_scores', []):
        i, value = row['class_id'], row.get('score')
        if i in scores:
            raise ValueError('Duplicate class scores')
        if type(value) in (float, int) and math.isfinite(value):
            scores[i] = value
    ids = sorted(i for i, value in scores.items() if value < policy.cryosift_threshold and counts.get(i, 0) > 0)
    missing = [i for i in ids if i not in images]
    if missing:
        return dict(metric_version=VERSION, policy=asdict(policy), available=False, passed=False,
                    failed_conditions=['missing_qualified_class_images'], missing_class_ids=missing)
    try:
        matrix = pairwise_shape_ncc([images[i] for i in ids], policy)
    except ValueError as exc:
        return dict(metric_version=VERSION, policy=asdict(policy), available=False, passed=False,
                    failed_conditions=['invalid_qualified_class_images'], error=str(exc))
    result = gate_from_matrix(ids, counts, scores, matrix, policy)
    result.update(source=source, original_class_count=len(records),
                  scored_class_count=len(scores), cryosift_qualified_class_count=len(ids))
    if not ids:
        result['failed_conditions'].insert(0, 'no_classes_below_cryosift_threshold')
    return result


def load_class_images(project, job_uid):
    job = project.find_job(job_uid)
    if job.status != 'completed' or job.type != 'class_2D_new':
        raise ValueError('Completed Class 2D source required')
    dataset = job.load_output('class_averages')
    from pathlib import Path
    stacks, images = {}, {}
    for i, (path, index) in enumerate(zip(dataset['blob/path'], dataset['blob/idx'])):
        name = str(path)
        if name not in stacks:
            _, stack = job.download_mrc(Path(name).name)
            stack = np.asarray(stack)
            stacks[name] = stack[None] if stack.ndim == 2 else stack
        images[i] = stacks[name][int(index)]
    return images
