"""Per-task, micrograph-only size initialization; never reads reference maps."""
import json
import os
from pathlib import Path
import subprocess
import sys


def assets():
    base = Path(os.getenv('CRYOAGENT_CRYOLO_ASSETS', str(Path.home() / '.local/share/cryosage-tools/cryolo')))
    return (Path(os.getenv('CRYOAGENT_CRYOLO_PYTHON', sys.executable)),
            Path(os.getenv('CRYOAGENT_CRYOLO_SOURCE', str(base/'cryolo-1.9.9'))),
            Path(os.getenv('CRYOAGENT_CRYOLO_WEIGHTS', str(base/'gmodel_phosnet_202005_N63_c17.h5'))))


def estimate(task, root, project, ctf):
    from webapp.core import write
    from cryolo_initial_scale import snapshot_scale
    import numpy as np
    directory = root/'scale_probe'
    receipt = directory/'result.json'
    if receipt.exists():
        return json.loads(receipt.read_text()).get('initial_scale')
    python, source, weights = assets()
    if not (os.access(python, os.X_OK) and source.is_dir() and os.access(weights, os.R_OK)):
        write(receipt, dict(status='unavailable', fallback='model_visual_scale',
                            reason='Configure readable crYOLO Python/source/weights in the submitting user environment'))
        return None
    dataset = task['id']  # User-supplied display IDs must never become paths.
    data = ctf.load_output('exposures')
    indices = sorted(np.random.RandomState(20261028).choice(len(data), min(8, len(data)), replace=False).tolist())
    folder = directory/dataset
    inputs = folder/'input'
    inputs.mkdir(parents=True, exist_ok=True)
    entries = []
    for n, index in enumerate(indices):
        remote = str(data['micrograph_blob/path'][index])
        pixel = float(data['micrograph_blob/psize_A'][index])
        staged = inputs/('micrograph_%02d.mrc' % n)
        if not staged.exists():
            pending = staged.with_suffix('.pending.mrc')
            project.download_file(remote, pending)
            pending.replace(staged)
        entries.append(dict(source=str(staged), remote_source=remote, staged=str(staged), pixel_size_A=pixel))
    write(directory/'manifest.json', dict(datasets={dataset: dict(images=entries, ctf_job=ctf.uid)}))
    write(folder/'config.json', dict(model=dict(architecture='PhosaurusNet', input_size=1024,
        max_box_per_image=10000, backend_weights=str(weights), filter=[.1, str(folder/'filtered')]),
        train={}, valid={}, other=dict(log_path=str(folder/'logs'))))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='1',
                       CRYOAGENT_CRYOLO_CPUS=os.getenv('CRYOAGENT_CRYOLO_CPUS', '2'),
                       MPLCONFIGDIR=str(directory/'matplotlib'), KERAS_HOME=str(directory/'keras'))
    try:
        with (directory/'probe.log').open('a') as log:
            subprocess.run([str(python), '-u', '-m', 'webapp.cryolo_probe', str(directory), dataset,
                            str(source), str(weights)], check=True, timeout=3600, env=environment,
                           cwd=directory, stdout=log, stderr=log)
        initial = snapshot_scale(directory, dataset, task['config']['pixel_size_A'])
    except (subprocess.SubprocessError, ValueError, KeyError, OSError, RuntimeError) as exc:
        write(receipt, dict(status='failed', fallback='model_visual_scale', reason=str(exc)[:1000]))
        return None
    write(receipt, dict(status='measured', initial_scale=initial,
        limitation='A learned image-only extent prior, not a guaranteed physical diameter'))
    return initial


if __name__ == '__main__':
    from scripts import cryolo_size_probe as probe
    probe.ROOT = Path(sys.argv[1])
    probe.SOURCE = Path(sys.argv[3])
    probe.WEIGHTS = Path(sys.argv[4])
    probe.predict(sys.argv[2])
    probe.analyze(sys.argv[2])
