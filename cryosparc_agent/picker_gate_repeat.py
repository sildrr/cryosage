"""Bootstrap a clean pilot before entering the amended diameter recovery path."""
from picker_blob_recovery import DiameterAwareRecovery
from sequential_picker_v4 import SequentialPickerV4
from sequential_picker_rollout import ProtocolExhausted
from cryolo_initial_scale import CryoloInitialScaleMixin


class FreshDiameterGateRepeat(CryoloInitialScaleMixin, DiameterAwareRecovery):
    def sample_picker(self, count):
        if not self.state.get('initial_pilot_exhausted'):
            return SequentialPickerV4.sample_picker(self, count)
        return super().sample_picker(count)

    def request_adjustment(self, tag, previous, history=()):
        if not self.state.get('initial_pilot_exhausted'):
            return SequentialPickerV4.request_adjustment(self, tag, previous, history)
        return super().request_adjustment(tag, previous, history)

    def pilot(self):
        if not self.state.get('initial_pilot_exhausted'):
            try:
                return SequentialPickerV4.pilot(self)
            except ProtocolExhausted as exc:
                # Only scientific exhaustion of the initial ladder opens recovery.
                # Submission budgets, empty picks and infrastructure failures must propagate.
                if str(exc) != 'Signal-aware pilot adjustments exhausted; no qualified pilot':
                    raise
                self.state['initial_pilot_exhausted'] = True
                self.persist()
        return super().pilot()
