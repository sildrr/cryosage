"""Exercise the actual shell prelude without starting a controller."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


def check_routing_without_username_lookup(tmp_path, uid, optional_env):
    script = Path(__file__).resolve().parents[1] / 'controller.sbatch'
    prelude = script.read_text().split('exec "${CRYOAGENT_WEB_PYTHON', 1)[0]
    assert 'exec "${CRYOAGENT_WEB_PYTHON' in script.read_text()
    if optional_env:
        config = tmp_path / '.config/cryoagent'
        config.mkdir(parents=True)
        (config / 'cryosift.env').write_text(
            'export CRYOAGENT_GPU_LANE=h20_slurm\n'
            'export CRYOAGENT_GPU_LANES=h20_slurm\n'
            'export CRYOAGENT_AUTO_SELECT_LANE=1\n')
    env = dict(os.environ, HOME=str(tmp_path), TEST_UID=uid,
               CRYOAGENT_REPO_ROOT=str(script.parents[1]),
               CRYOAGENT_WEB_DYNAMIC_GPUS='1', CRYOAGENT_WEB_MAX_GPUS='8',
               CRYOAGENT_GPU_LANE='h20_slurm', CRYOAGENT_GPU_LANES='h20_slurm',
               CRYOAGENT_AUTO_SELECT_LANE='1')
    stub = '''
id() {
    if [[ "$*" == "-u" ]]; then printf '%s\\n' "$TEST_UID";
    else printf 'username lookup unavailable\\n' >&2; return 1; fi
}
'''
    inspect = '\nprintf "%s|%s" "$CRYOAGENT_WEB_DYNAMIC_GPUS" "$CRYOAGENT_WEB_MAX_GPUS"\n'
    result = subprocess.run(['bash', '-c', stub + prelude + inspect], env=env,
                            capture_output=True, text=True, check=True)
    assert not result.stderr
    assert result.stdout == '1|8'


class ControllerRoutingTests(unittest.TestCase):
    def test_missing_username_with_optional_env_variants(self):
        for uid in ['9036', '9031']:
            for optional_env in [False, True]:
                with self.subTest(uid=uid, optional_env=optional_env):
                    with tempfile.TemporaryDirectory() as directory:
                        check_routing_without_username_lookup(Path(directory), uid, optional_env)


if __name__ == '__main__':
    unittest.main()
