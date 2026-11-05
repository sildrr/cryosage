"""Native parameter/input consistency checks before Job creation."""
from copy import deepcopy


def normalize_input_parameters(kind, parameters, connections):
    result = dict(parameters)
    if kind != 'class_3D':
        return result
    has_volume = bool((connections or {}).get('volume'))
    mode = result.get('class3D_init_mode', 'simple')
    if has_volume:
        if mode not in {None, 'simple', 'input'}:
            raise ValueError('3D Classification with a volume requires input initialization')
        result['class3D_init_mode'] = 'input'
    elif mode == 'input':
        raise ValueError('3D Classification input initialization requires a connected volume')
    return result


def normalize_volume_connections(kind, parameters, connections, *, repeat_allowed=False,
                                 require_class_count=True):
    result = deepcopy(connections or {})
    if kind != 'class_3D' or not result.get('volume'):
        return result
    classes = parameters.get('class3D_N_K')
    if classes is None and not require_class_count:
        return result
    if type(classes) is not int or not 2 <= classes <= 100:
        raise ValueError('3D Classification requires an integer class count between 2 and 100')
    value = result['volume']
    volumes = value if isinstance(value, list) else [value]
    if len(volumes) == 1 and classes > 1:
        if not repeat_allowed:
            raise ValueError('Native Registry does not authorize repeated initial-volume connections')
        volumes = [deepcopy(volumes[0]) for _ in range(classes)]
    if len(volumes) != classes:
        raise ValueError('3D Classification input volume count must equal class3D_N_K')
    def identity(item):
        if isinstance(item, dict):
            return item.get('source_job_uid'), item.get('source_output')
        return tuple(item)
    if not repeat_allowed and len({identity(item) for item in volumes}) != len(volumes):
        raise ValueError('Repeated initial volumes are not authorized by the native Registry')
    result['volume'] = volumes
    return result
