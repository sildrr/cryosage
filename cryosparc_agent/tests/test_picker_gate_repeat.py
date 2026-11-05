from unittest.mock import Mock, patch

import pytest

from picker_gate_repeat import FreshDiameterGateRepeat
from picker_blob_recovery import DiameterAwareRecovery
from sequential_picker_v4 import SequentialPickerV4
from sequential_picker_rollout import ProtocolExhausted


def bare():
    instance = object.__new__(FreshDiameterGateRepeat)
    instance.state = {}
    instance.persist = Mock()
    return instance


def test_clean_pilot_can_pass_without_recovery():
    run = bare()
    with patch.object(SequentialPickerV4, 'pilot', return_value={'accepted': True}), \
            patch.object(DiameterAwareRecovery, 'pilot') as recovery:
        assert run.pilot() == {'accepted': True}
        recovery.assert_not_called()


def test_only_initial_scientific_exhaustion_enters_recovery_once():
    run = bare()
    failure = ProtocolExhausted('Signal-aware pilot adjustments exhausted; no qualified pilot')
    with patch.object(SequentialPickerV4, 'pilot', side_effect=failure) as initial, \
            patch.object(DiameterAwareRecovery, 'pilot', return_value='recovered'):
        assert run.pilot() == 'recovered'
        assert run.pilot() == 'recovered'
        assert initial.call_count == 1
        run.persist.assert_called_once()


@pytest.mark.parametrize('failure', [ProtocolExhausted('Structured phase Job budget exhausted'),
                                    ProtocolExhausted('Blob output empty'), RuntimeError('network')])
def test_other_failures_do_not_start_more_experiments(failure):
    run = bare()
    with patch.object(SequentialPickerV4, 'pilot', side_effect=failure), \
            patch.object(DiameterAwareRecovery, 'pilot') as recovery:
        with pytest.raises(type(failure)):
            run.pilot()
        recovery.assert_not_called()
        assert not run.state.get('initial_pilot_exhausted')


def test_fresh_picker_uses_initial_stage_then_revised_diameter_stage():
    run = bare()
    with patch.object(SequentialPickerV4, 'sample_picker', return_value='initial') as initial, \
            patch.object(DiameterAwareRecovery, 'sample_picker', return_value='revised') as revised:
        assert run.sample_picker(50) == 'initial'
        run.state['initial_pilot_exhausted'] = True
        assert run.sample_picker(200) == 'revised'
        initial.assert_called_once_with(run, 50)
        revised.assert_called_once_with(200)
