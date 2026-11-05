"""Standalone crYOLO size probe; never creates or changes CryoSPARC jobs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/hdd1/linweifan/cryosparc/cryolo_size_probe_20261028')
PROJECT = Path('/hdd1b/linweifan/cryosparc/projects/CS-empiar-10025-agent-full-dataset-4090-test')
SOURCE = ROOT / 'software/cryolo-1.9.9'
WEIGHTS = ROOT / 'software/gmodel_phosnet_202005_N63_c17.h5'
PYTHON = '/ssd1/lisongyang/envs/cryolo-size/bin/python'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare():
    import numpy as np
    from cryosparc.dataset import Dataset
    manifest = {'sample_seed': 20261028, 'images_per_dataset': 8,
                'diameter_prior': None, 'datasets': {}}
    for dataset, job in [('10025', None), ('10228', 'J1246'),
                         ('10764', 'J1031'), ('11280', 'J1442')]:
        if job:
            data = Dataset.load(PROJECT / job / (job + '_passthrough_exposures.cs'))
            candidates = []
            for row in data.rows():
                path = PROJECT / str(row['micrograph_blob/path'])
                if path.is_file():
                    candidates.append((str(path), float(row['micrograph_blob/psize_A'])))
        else:
            candidates = [(str(p), .6575) for p in sorted(Path(
                '/home/share/empiar/10025/data/14sep05c_averaged_196').glob('*.mrc'))]
        candidates.sort()
        if len(candidates) < 8:
            raise RuntimeError('Too few readable micrographs: ' + dataset)
        rng = np.random.RandomState(20261028)
        selected = sorted(rng.choice(len(candidates), 8, replace=False).tolist())
        folder = ROOT / dataset
        inputs = folder / 'input'
        inputs.mkdir(parents=True, exist_ok=True)
        entries = []
        for i, idx in enumerate(selected):
            path, pixel = candidates[idx]
            link = inputs / ('micrograph_%02d.mrc' % i)
            if not link.exists():
                link.symlink_to(path)
            if link.resolve() != Path(path).resolve():
                raise RuntimeError('Input link conflict: ' + str(link))
            entries.append(dict(source=path, staged=str(link), pixel_size_A=pixel))
        # The general model is trained on low-pass filtered images. No anchors,
        # minimum size or maximum size are provided to the estimator.
        config = {'model': {'architecture': 'PhosaurusNet', 'input_size': 1024,
                           'max_box_per_image': 10000,
                           'backend_weights': str(WEIGHTS),
                           'filter': [0.1, str(folder / 'filtered')]},
                  'train': {}, 'valid': {}}
        write(folder / 'config.json', config)
        manifest['datasets'][dataset] = dict(ctf_job=job, available=len(candidates), images=entries)
    write(ROOT / 'manifest.json', manifest)
    print(json.dumps({k: v['available'] for k, v in manifest['datasets'].items()}))


def predict(dataset):
    # Only bypass the optional desktop GUI adapter. Inference, filtering,
    # coordinate conversion and CBOX writing remain upstream crYOLO code.
    import types
    class HeadlessParser(argparse.ArgumentParser):
        pass
    original = argparse._ActionsContainer.add_argument
    def add_argument(self, *args, **kwargs):
        kwargs.pop('gooey_options', None)
        kwargs.pop('widget', None)
        return original(self, *args, **kwargs)
    argparse._ActionsContainer.add_argument = add_argument
    gui = types.ModuleType('gooey')
    gui.Gooey = lambda function=None, **kwargs: function or (lambda f: f)
    gui.GooeyParser = HeadlessParser
    sys.modules['gooey'] = gui
    sys.path.insert(0, str(SOURCE))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from cryolo import predict as upstream
    import cryolo.utils
    # No network/version query from a compute-node experiment.
    cryolo.utils.check_for_updates = lambda: None
    folder = ROOT / dataset
    parser = argparse.ArgumentParser()
    upstream.create_parser(parser, use_gooey=False)
    args = parser.parse_args(['-c', str(folder / 'config.json'), '-w', str(WEIGHTS),
                              '-i', str(folder / 'input'), '-o', str(folder / 'output'),
                              '-t', '0.3', '-pbs', '1', '--num_cpu', os.getenv('CRYOAGENT_CRYOLO_CPUS', '8')])
    upstream.main(args)


def read_cbox(path):
    columns = []
    active = False
    for line in path.read_text().splitlines():
        line = line.strip()
        if line == 'data_cryolo':
            active = True
            continue
        if active and line.startswith('data_'):
            break
        if not active or not line or line.startswith('#') or line == 'loop_':
            continue
        if line.startswith('_'):
            columns.append(line.split()[0])
            continue
        values = line.split()
        if len(values) == len(columns) and columns:
            yield dict(zip(columns, (float('nan') if x in ('<NA>', '?', '.')
                                      else float(x) for x in values)))


def analyze(dataset):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import mrcfile
    from scipy.ndimage import gaussian_filter
    folder = ROOT / dataset
    images = json.loads((ROOT / 'manifest.json').read_text())['datasets'][dataset]['images']
    metadata = {Path(x['staged']).stem: x for x in images}
    particles, by_image = [], {}
    for path in sorted((folder / 'output').rglob('*.cbox')):
        if path.stem not in metadata:
            raise RuntimeError('Unknown CBOX image: ' + str(path))
        item = metadata[path.stem]
        rows = []
        for row in read_cbox(path):
            width, height = row['_EstWidth'], row['_EstHeight']
            if not np.isfinite([width, height, row['_Confidence']]).all() or min(width, height) <= 0:
                continue
            value = dict(image=path.stem, confidence=row['_Confidence'],
                         estimated_width_px=width, estimated_height_px=height,
                         estimated_diameter_A=max(width, height) * item['pixel_size_A'],
                         estimated_short_axis_A=min(width, height) * item['pixel_size_A'],
                         center_x_px=row['_CoordinateX'] + row['_Width'] / 2,
                         center_y_px=row['_CoordinateY'] + row['_Height'] / 2)
            particles.append(value)
            rows.append(value)
        by_image[path.stem] = rows
    if not particles:
        raise RuntimeError('No estimated CBOX dimensions: ' + dataset)
    with (folder / 'particles.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(particles[0]))
        writer.writeheader()
        writer.writerows(particles)
    stats = {}
    for threshold in [.3, .5, .7]:
        subset = [x for x in particles if x['confidence'] >= threshold]
        diameters = [x['estimated_diameter_A'] for x in subset]
        stats[str(threshold)] = dict(count=len(subset),
            median_A=float(np.median(diameters)) if diameters else None,
            p10_p90_A=np.percentile(diameters, [10, 90]).tolist() if diameters else None,
            per_image_median_A={name: float(np.median([x['estimated_diameter_A'] for x in rows
                if x['confidence'] >= threshold])) for name, rows in by_image.items()
                if any(x['confidence'] >= threshold for x in rows)})
    summary = dict(dataset=dataset, statistics=stats, input_images=len(images),
        measurement='max(EstWidth, EstHeight) in native pixels times native pixel size',
        caveat='Predicted bounding extent, not independently validated molecular diameter; general-model training overlap unknown.',
        weights_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest())
    write(folder / 'summary.json', summary)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for threshold in [.3, .5, .7]:
        values = [x['estimated_diameter_A'] for x in particles if x['confidence'] >= threshold]
        if values:
            axes[0].hist(values, bins=40, histtype='step', density=True, label='confidence >= ' + str(threshold))
    axes[0].set_xlabel('Predicted bounding diameter (A)')
    axes[0].legend()
    values = stats['0.5']['per_image_median_A']
    axes[1].bar(range(len(values)), list(values.values()), color='#207e85')
    axes[1].set_xlabel('Micrograph index')
    axes[1].set_ylabel('Median diameter (A), confidence >= 0.5')
    fig.suptitle('EMPIAR-' + dataset + ': crYOLO general model, no supplied diameter')
    fig.tight_layout()
    fig.savefig(folder / 'size_distribution.png', dpi=160)
    plt.close(fig)
    for image in images[:2]:
        with mrcfile.open(image['source'], permissive=True) as mrc:
            raw = np.asarray(mrc.data, dtype=np.float32).squeeze()
        if raw.ndim != 2:
            raise RuntimeError('Not a single aligned micrograph')
        # Match a low-pass display without changing inference coordinates.
        display = gaussian_filter(raw, sigma=8.0 / image['pixel_size_A'])[::4, ::4]
        lo, hi = np.percentile(display, [1, 99])
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(display, cmap='gray', vmin=lo, vmax=hi, origin='upper',
                  extent=[0, raw.shape[1], raw.shape[0], 0])
        rows = [x for x in by_image.get(Path(image['staged']).stem, []) if x['confidence'] >= .5]
        if len(rows) > 100:
            rows = [rows[i] for i in np.linspace(0, len(rows)-1, 100).astype(int)]
        for row in rows:
            w, h = row['estimated_width_px'], row['estimated_height_px']
            # Upstream read_mrc flips the array, then rescale flips Y back.
            # Consequently exported MRC CBOX centers match native array rows.
            x, y = row['center_x_px'], row['center_y_px']
            ax.add_patch(Rectangle((x-w/2, y-h/2), w, h, edgecolor='#ffb23b', facecolor='none', lw=.6))
        length = 500 / image['pixel_size_A']
        ax.plot([raw.shape[1]*.05, raw.shape[1]*.05+length], [raw.shape[0]*.95]*2, color='white', lw=3)
        ax.text(raw.shape[1]*.05, raw.shape[0]*.93, '500 A', color='white')
        ax.set_title('EMPIAR-' + dataset + ': predicted extents, confidence >= 0.5 (at most 100 shown)')
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(folder / (Path(image['staged']).stem + '_overlay.png'), dpi=140)
        plt.close(fig)
    print(json.dumps(summary), flush=True)


def run():
    import h5py
    deadline = time.monotonic() + 3600
    while not WEIGHTS.exists() or WEIGHTS.stat().st_size != 202707359:
        if time.monotonic() > deadline:
            raise RuntimeError('Official weights download incomplete after one hour')
        print('Waiting for complete official weights; bytes=' + str(
            WEIGHTS.stat().st_size if WEIGHTS.exists() else 0), flush=True)
        time.sleep(30)
    with h5py.File(str(WEIGHTS), 'r') as model:
        print('Validated HDF5; root keys=' + str(list(model.keys())[:8]), flush=True)
    failures = {}
    for dataset in json.loads((ROOT / 'manifest.json').read_text())['datasets']:
        try:
            subprocess.run([PYTHON, '-u', __file__, 'predict', '--dataset', dataset], check=True)
            analyze(dataset)
        except Exception as error:
            failures[dataset] = str(error)
            print('FAILED ' + dataset + ': ' + str(error), flush=True)
        write(ROOT / 'run_status.json', dict(failures=failures, last_dataset=dataset))
    report()
    if failures:
        raise RuntimeError(str(failures))


def report():
    import numpy as np
    import mrcfile
    refs = {'10025': 'J370/J370_005_volume_mask_refine.mrc',
            '10228': 'J1383/J1383_005_volume_mask_refine.mrc',
            '10764': 'J2191/J2191_008_volume_mask_refine.mrc',
            '11280': 'J117/J117_009_volume_mask_refine.mrc'}
    reference = {}
    for dataset, relative in refs.items():
        path = PROJECT / relative
        if not path.is_file():
            continue
        with mrcfile.open(str(path), permissive=True) as mrc:
            mask = np.asarray(mrc.data)
            pixel = float(mrc.voxel_size.x)
            if not np.isfinite(pixel) or pixel <= 0:
                continue
            measures = []
            for threshold in [.25, .5, .75]:
                spans = []
                for axis in range(3):
                    profile = np.max(mask, axis=tuple(x for x in range(3) if x != axis))
                    indices = np.flatnonzero(profile >= threshold)
                    spans.append(float((indices[-1] - indices[0] + 1) * pixel) if len(indices) else None)
                measures.append(dict(threshold=threshold, axis_spans_A=spans))
            reference[dataset] = dict(path=str(path), voxel_A=pixel, measures=measures,
                caveat='Post-hoc reconstruction mask extent, NOT ground-truth diameter and never used for prediction.')
    write(ROOT / 'posthoc_mask_references.json', reference)
    lines = ['# Standalone crYOLO Size Probe', '',
        'Eight fixed random aligned micrographs per dataset; seed 20261028. No new CryoSPARC jobs.',
        'crYOLO 1.9.9 / TensorFlow 1.15.4 CPU; official low-pass general model. No supplied diameter/anchors/size filter.',
        'EstWidth/EstHeight are converted from native pixels to Angstroms; maximum axis reported.', '',
        '| EMPIAR | Picks >=0.5 | Median (A) | P10-P90 (A) | Median >=0.7 (A) | Post-hoc mask axes at 0.5 (A) |',
        '| --- | ---: | ---: | --- | ---: | --- |']
    for dataset in refs:
        path = ROOT / dataset / 'summary.json'
        if not path.exists():
            lines.append('| %s | pending/failed | | | | |' % dataset)
            continue
        data = json.loads(path.read_text())['statistics']
        main, strict = data['0.5'], data['0.7']
        ref = reference.get(dataset, {}).get('measures', [])
        dims = ref[1]['axis_spans_A'] if ref else []
        fmt = lambda value: ('%.1f' % value) if value is not None else 'NA'
        lines.append('| %s | %d | %s | %s | %s | %s |' % (dataset, main['count'],
            fmt(main['median_A']), ', '.join(map(fmt, main['p10_p90_A'] or [])),
            fmt(strict['median_A']), ', '.join(map(fmt, dims))))
    lines.extend(['', '## Interpretation Limits',
        '- Predictions are learned bounding extents, not a physical edge measurement.',
        '- Inspect overlays for complete particles, contaminants, orientation dependence and merged neighbors.',
        '- Mask extents include padding and depend on the reconstructed conformation; they are only an independent rough scale check.',
        '- General-model training overlap with these public datasets is unknown; this is not a held-out generalization benchmark.',
        '- Do not automatically replace production picker diameters or box sizes from this probe.', '',
        'Per-dataset folders contain raw CBOX, particles.csv, summary.json, size_distribution.png and two full-field overlays.',
        'Official documentation: https://cryolo-doc.readthedocs.io/en/stable/installation.html',
        'Estimated size export: https://cryolo-doc.readthedocs.io/en/latest/tutorials/tutorial_overview.html', ''])
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))


def reanalyze():
    previous = json.loads((ROOT / 'run_status.json').read_text())
    for dataset in json.loads((ROOT / 'manifest.json').read_text())['datasets']:
        analyze(dataset)
    report()
    write(ROOT / 'run_status.json', dict(failures={}, analysis_completed=True,
        recovery='Reused all four saved inference outputs; fixed optional CBOX <NA> parsing.',
        previous_run_status=previous))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'predict', 'analyze', 'run', 'report', 'reanalyze'])
    parser.add_argument('--dataset', choices=['10025', '10228', '10764', '11280'])
    options = parser.parse_args()
    if options.action in ('predict', 'analyze') and not options.dataset:
        parser.error('--dataset is required')
    if options.action in ('prepare', 'run', 'report', 'reanalyze'):
        globals()[options.action]()
    else:
        globals()[options.action](options.dataset)
