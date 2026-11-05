import unittest
from enum import Enum
from types import SimpleNamespace
from unittest.mock import patch

from dynamic_candidates import (
    build_registry_candidate,
    registry_parameter_template,
    resolve_registry_connections,
)

class DynamicCandidateTests(unittest.TestCase):
    def test_orientation_diagnostics_volume_inputs_are_alternatives(self):
        from dynamic_candidates import registry_required_input_counts
        spec = SimpleNamespace(type='orientation_diagnostics', inputs=SimpleNamespace(root={
            'volume': SimpleNamespace(type='volume', slots=['map', 'map_half_A', 'map_half_B'], count_min=1),
            'volumes_all_classes': SimpleNamespace(type='volume_multi', slots=['map', 'map_half_A', 'map_half_B'], count_min=1)}))
        single = dict(source_type='volume', source_job_uid='J10', source_output='volume',
                      result_names=['map', 'map_half_A', 'map_half_B'])
        multi = dict(source_type='volume_multi', source_job_uid='J9', source_output='volumes_all_classes',
                     result_names=['map', 'map_half_A', 'map_half_B'])
        self.assertIsNone(resolve_registry_connections(spec, []))
        result = resolve_registry_connections(spec, [single])
        self.assertEqual(set(result), {'volume'})
        self.assertEqual(registry_required_input_counts(spec, result), {'volume': 1})
        result = resolve_registry_connections(spec, [multi])
        self.assertEqual(set(result), {'volumes_all_classes'})
        self.assertEqual(registry_required_input_counts(spec, result), {'volumes_all_classes': 1})
        self.assertEqual(set(resolve_registry_connections(spec, [single, multi])), {'volume'})
        with self.assertRaises(ValueError):
            registry_required_input_counts(spec, {'volume': [single], 'volumes_all_classes': [multi]})

    def test_obsolete_specs_are_blocked_and_modern_diagnostics_remain(self):
        from dynamic_candidates import build_registry_next_actions, obsolete_registry_job
        class Stability(str, Enum):
            OBSOLETE = 'obsolete'
        obsolete = SimpleNamespace(type='fsc3D', hidden=False, stability=Stability.OBSOLETE)
        modern = SimpleNamespace(type='orientation_diagnostics', hidden=False, stability='stable')
        self.assertTrue(obsolete_registry_job(obsolete))
        self.assertFalse(obsolete_registry_job(modern))
        client = SimpleNamespace(job_register=SimpleNamespace(specs=[obsolete, modern]))
        node = dict(status='completed', job_type='validation', cryosparc_job_uid='J9')
        with patch('dynamic_candidates.cryosparc_client', return_value=client), \
             patch('dynamic_candidates.available_output_sources', return_value=[]), \
             patch('dynamic_candidates.resolve_registry_connections', return_value={'volume': []}), \
             patch('dynamic_candidates.build_registry_candidate', side_effect=lambda n,s,r: {'job_type':s.type}):
            actions, blocked = build_registry_next_actions({'nodes': []}, node)
        self.assertEqual([a['job_type'] for a in actions], ['orientation_diagnostics'])
        self.assertEqual(blocked[0]['job_type'], 'fsc3D')
        self.assertEqual(blocked[0]['replacement_job_type'], 'orientation_diagnostics')

    def test_imported_particle_classification_binds_native_output_types(self):
        from dynamic_candidates import available_output_sources
        state = {"nodes": [{"status": "completed", "cryosparc_job_uid": "J2",
            "workflow_node_id": "J2", "logical_node_id": "class2d", "job_type": "class_2D_new",
            "outputs": {
                "particles": {"available": True, "type": "particle", "num_items": 184564,
                              "result_names": ["blob", "ctf", "alignments2D"]},
                "class_averages": {"available": True, "type": "template", "num_items": 50,
                                   "result_names": ["blob"]}}}]}
        spec = SimpleNamespace(type="select_2D", inputs=SimpleNamespace(root={
            "particles": SimpleNamespace(type="particle", slots=["blob"], count_min=1),
            "templates": SimpleNamespace(type="template", slots=["blob"], count_min=1)}))
        sources = available_output_sources(state)
        bindings = resolve_registry_connections(spec, sources)
        self.assertEqual(bindings['particles'][0]['source_output'], 'particles')
        self.assertEqual(bindings['templates'][0]['source_output'], 'class_averages')
        # Legacy normalized sources still distinguish common output names.
        for source in sources:
            source.pop('source_type')
        bindings = resolve_registry_connections(spec, sources)
        self.assertEqual(bindings['templates'][0]['source_output'], 'class_averages')

    def test_native_type_rejects_mismatched_blob_even_with_ambiguous_name(self):
        from dynamic_candidates import source_matches_input
        source = dict(source_type='particle', source_output='templates', result_names=['blob'])
        self.assertFalse(source_matches_input(source, 'template', {'blob'}))

    def test_inspect_picks_uses_local_automatic_policy(self):
        from dynamic_candidates import build_registry_candidate
        from job_specs import get_job_spec

        class Param:
            def __init__(self, default=None):
                self.default = default

        class RegistrySpec:
            type = "inspect_picks_v2"
            category = "particle_picking"
            tags = ["interactive"]
            interactive = True
            title = "Inspect Particle Picks"
            params = {"n_mic_to_plot": Param(10)}

        node = {"cryosparc_job_uid": "J1", "workflow_node_id": "J1"}
        action = build_registry_candidate(node, RegistrySpec(), {"particles": []})
        self.assertFalse(action["job_spec_metadata"]["interactive"])
        self.assertFalse(action["job_spec_metadata"]["requires_approval"])
        self.assertTrue(action["job_spec_metadata"]["queue_without_lane"])

    def test_select_2d_uses_local_automatic_policy(self):
        class RegistrySpec:
            type = "select_2D"
            category = "particle_selection"
            tags = ["interactive"]
            interactive = True
            title = "Select 2D Classes"
            params = {}

        node = {"cryosparc_job_uid": "J1", "workflow_node_id": "J1"}
        action = build_registry_candidate(node, RegistrySpec(), {"particles": []})
        metadata = action["job_spec_metadata"]
        self.assertFalse(metadata["interactive"])
        self.assertFalse(metadata["requires_approval"])
        self.assertTrue(metadata["queue_without_lane"])

    def test_registry_parameter_types_and_constraints_are_preserved(self):
        spec = SimpleNamespace(
            type="blob_picker_gpu",
            title="Blob Picker",
            tags=["gpuEnabled", "multiGpu"],
            interactive=False,
            params={
                "diameter": SimpleNamespace(type="number", anyOf=[], required_param=True, default=None, hidden=False, enum=None, ge=10, le=500),
                "use_circle": SimpleNamespace(type="boolean", anyOf=[], required_param=False, default=1, hidden=False, enum=None, ge=None, le=None),
                "diameter_max": SimpleNamespace(type=None, anyOf=[SimpleNamespace(type="number"), SimpleNamespace(type="null")], required_param=False, default=None, hidden=False, enum=None, ge=None, le=None),
            },
        )
        template = registry_parameter_template(spec)
        self.assertEqual(template["diameter"]["type"], "number")
        self.assertTrue(template["diameter"]["required"])
        self.assertEqual(template["diameter"]["minimum"], 10)
        self.assertEqual(template["diameter_max"]["type"], "number")

    def test_optional_registry_slots_do_not_block_particle_input(self):
        spec = SimpleNamespace(
            inputs=SimpleNamespace(root={
                "particles": SimpleNamespace(
                    type="particle",
                    slots=["blob", "?ctf", "?location", "?pick_stats"],
                    count_min=1,
                ),
            }),
        )
        sources = [{
            "source_job_uid": "J6",
            "source_output": "particles",
            "result_names": ["blob", "location", "ctf", "pick_stats"],
        }]
        connections = resolve_registry_connections(spec, sources)
        self.assertEqual(connections["particles"][0]["source_job_uid"], "J6")

    def test_primary_particles_are_preferred_over_unused_branch(self):
        spec = SimpleNamespace(
            type="homo_refine_new",
            inputs=SimpleNamespace(root={
                "particles": SimpleNamespace(
                    type="particle", slots=["blob"], count_min=1,
                ),
            }),
        )
        sources = [
            {"source_job_uid": "J9", "source_output": "particles_class_0", "result_names": ["blob"]},
            {"source_job_uid": "J9", "source_output": "particles_unused", "result_names": ["blob"]},
        ]
        connections = resolve_registry_connections(spec, sources)
        self.assertEqual(connections["particles"][0]["source_output"], "particles_class_0")

    def test_recovery_scope_excludes_future_job_outputs(self):
        from dynamic_candidates import available_output_sources
        state = {"nodes": [
            {"status": "completed", "cryosparc_job_uid": "J9", "workflow_node_id": "J9", "logical_node_id": "n9",
             "job_type": "homo_abinit", "outputs": {"particles_class_0": {"available": True, "result_names": ["blob"]}}},
            {"status": "completed", "cryosparc_job_uid": "J14", "workflow_node_id": "J14", "logical_node_id": "n14",
             "job_type": "homo_refine_new", "outputs": {"particles": {"available": True, "result_names": ["blob"]}}},
        ]}
        sources = available_output_sources(state, max_job_uid="J9")
        self.assertEqual([source["source_job_uid"] for source in sources], ["J9"])

    def test_excluded_particle_output_is_fallback_when_only_match(self):
        spec = SimpleNamespace(
            type="homo_refine_new",
            inputs=SimpleNamespace(root={
                "particles": SimpleNamespace(
                    type="particle", slots=["blob"], count_min=1,
                ),
            }),
        )
        sources = [{"source_job_uid": "J9", "source_output": "particles_rejected", "result_names": ["blob"]}]
        connections = resolve_registry_connections(spec, sources)
        self.assertEqual(connections["particles"][0]["source_output"], "particles_rejected")

    def test_movie_output_with_mscope_is_not_ctf_exposure(self):
        spec = SimpleNamespace(
            type="patch_ctf_estimation_multi",
            inputs=SimpleNamespace(root={
                "exposures": SimpleNamespace(
                    type="exposure", slots=["micrograph_blob", "mscope_params"], count_min=1,
                ),
            }),
        )
        sources = [{
            "source_job_uid": "J94", "source_output": "imported_movies", "num_items": 3657,
            "result_names": ["movie_blob", "mscope_params"],
        }]
        self.assertIsNone(resolve_registry_connections(spec, sources))

    def test_primary_exposure_output_is_preferred_over_incomplete_branch(self):
        spec = SimpleNamespace(
            type="patch_ctf_estimation_multi",
            inputs=SimpleNamespace(root={
                "exposures": SimpleNamespace(
                    type="exposure", slots=["micrograph_blob", "mscope_params"], count_min=1,
                ),
            }),
        )
        sources = [
            {"source_job_uid": "J95", "source_output": "micrographs", "num_items": 3656,
             "result_names": ["micrograph_blob", "mscope_params"]},
            {"source_job_uid": "J95", "source_output": "micrographs_incomplete", "num_items": 1,
             "result_names": ["micrograph_blob", "mscope_params"]},
        ]
        connections = resolve_registry_connections(spec, sources)
        self.assertEqual(connections["exposures"][0]["source_output"], "micrographs")

    def test_registry_gpu_metadata_is_executable(self):
        spec = SimpleNamespace(
            type="patch_ctf_estimation_multi", title="Patch CTF", tags=["gpuEnabled", "multiGpu"],
            interactive=False, category="ctf_estimation", params={},
        )
        candidate = build_registry_candidate(
            {"cryosparc_job_uid": "J6", "workflow_node_id": "J6"}, spec, {"exposures": []}
        )
        metadata = candidate["job_spec_metadata"]
        self.assertTrue(metadata["requires_gpu"])
        self.assertTrue(metadata["multi_gpu"])
        self.assertEqual(metadata["category"], "ctf_estimation")
        self.assertIn("default_lane", metadata)


if __name__ == "__main__":
    unittest.main()
