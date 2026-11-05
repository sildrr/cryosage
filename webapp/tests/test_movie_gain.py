from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from webapp import core, worker


class MovieGainTests(unittest.TestCase):
    def test_gain_is_validated_and_forwarded_only_when_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            gain=Path(d)/'gain.mrc';gain.write_bytes(b'header')
            payload=dict(input_type='movies',input_path='/input/*.tif',gainref_path=str(gain),
                pixel_size_A=1.0,voltage_kV=300,cs_mm=2.7,dose=40,target=3,max_jobs=40,external_consent=True)
            with patch.object(core,'validate_image_input',side_effect=lambda _:dict(input_path='/input/*.tif',input_file_count=2)), \
                    patch.object(core,'allowed_path',return_value=True):
                config=core.validate_request(payload)
                kind,params=worker.import_spec(config,Path(d))
                self.assertEqual(kind,'import_movies')
                self.assertEqual(params['gainref_path'],str(gain))
                del payload['gainref_path']
                config=core.validate_request(payload)
                self.assertNotIn('gainref_path',worker.import_spec(config,Path(d))[1])

    def test_gain_not_accepted_for_micrograph_input(self):
        with patch.object(core,'validate_image_input',return_value={}):
            with self.assertRaises(ValueError):
                core.validate_request(dict(input_type='micrographs',input_path='/input/*.mrc',
                    gainref_path='/input/gain.mrc',external_consent=True))

    def test_gain_cannot_escape_input_roots(self):
        with patch.object(core,'validate_image_input',return_value={}),patch.object(core,'allowed_path',return_value=False):
            with self.assertRaises(ValueError):
                core.validate_request(dict(input_type='movies',input_path='/input/*.tif',
                    gainref_path='/secret/gain.mrc',external_consent=True))


if __name__ == '__main__':
    unittest.main()
