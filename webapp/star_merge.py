"""Safely combine compatible RELION particle tables without altering optics."""
import os
import shlex
import tempfile
from pathlib import Path


def _quote(token):
    if not token or any(c.isspace() for c in token) or token[0] in '#_;' or token.startswith(('data_', 'loop_', 'stop_')):
        if "'" not in token:
            return "'" + token + "'"
        if '"' not in token:
            return '"' + token + '"'
        raise ValueError('STAR values containing both quote types are not supported for merging')
    return token


def _loop(tokens, first):
    labels = []
    while first.startswith('_'):
        labels.append(first)
        first = next(tokens, None)
        if first is None:
            break
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('STAR loop has missing or duplicate column names')
    return labels, first


def _parse(path, emit, expected):
    with open(path, encoding='utf-8') as stream:
        lexer = shlex.shlex(stream, posix=True)
        lexer.whitespace_split = True
        tokens = iter(lexer)
        first = next(tokens, None)
        optics = None
        if first == 'data_optics':
            if next(tokens, None) != 'loop_':
                raise ValueError('Unsupported STAR optics structure')
            labels, token = _loop(tokens, next(tokens, ''))
            rows = []
            while token is not None and token != 'data_particles':
                row = [token]
                for _ in range(len(labels) - 1):
                    v = next(tokens, None)
                    if v is None or v == 'data_particles':
                        raise ValueError('Incomplete optics row')
                    row.append(v)
                rows.append(tuple(row)); token = next(tokens, None)
            optics = (tuple(labels), tuple(rows))
            if not rows:
                raise ValueError('STAR optics table is empty')
            first = token
        if first != 'data_particles' or next(tokens, None) != 'loop_':
            raise ValueError('Multiple STAR files must contain data_particles and optional data_optics loops')
        labels, token = _loop(tokens, next(tokens, ''))
        if '_rlnImageName' not in labels:
            raise ValueError('STAR particle table is missing _rlnImageName')
        signature = (optics, tuple(labels))
        if expected is not None and signature != expected:
            raise ValueError('STAR files have different optics metadata or particle columns; combine them with compatible optics first')
        if emit and expected is None:
            if optics:
                emit('data_optics\n\nloop_\n')
                for i, label in enumerate(optics[0], 1): emit(f'{label} #{i}\n')
                for row in optics[1]: emit(' '.join(_quote(v) for v in row) + '\n')
                emit('\n')
            emit('data_particles\n\nloop_\n')
            for i, label in enumerate(labels, 1): emit(f'{label} #{i}\n')
        count = 0
        while token is not None:
            if token.startswith('data_') or token in ('loop_', 'stop_'):
                raise ValueError('Unsupported additional STAR block or loop after particles')
            row = [token]
            for _ in range(len(labels)-1):
                v = next(tokens, None)
                if v is None or v.startswith('data_') or v in ('loop_', 'stop_'):
                    raise ValueError('Incomplete particle row')
                row.append(v)
            if emit: emit(' '.join(_quote(v) for v in row) + '\n')
            count += 1
            token = next(tokens, None)
        if not count:
            raise ValueError('STAR particle table is empty')
        return signature, count


def merge_particle_stars(paths, destination):
    """Validate all files, optionally writing an atomic merged task-local STAR."""
    temp = None
    stream = None
    try:
        if destination is not None:
            destination = Path(destination)
            fd, temp = tempfile.mkstemp(prefix='.merged-particles-', dir=destination.parent)
            stream = os.fdopen(fd, 'w', encoding='utf-8')
        signature = None
        counts = []
        for path in paths:
            signature, count = _parse(path, stream.write if stream else None, signature)
            counts.append(count)
        if stream:
            stream.flush(); os.fsync(stream.fileno()); stream.close(); stream = None
            os.replace(temp, destination); temp = None
        return counts
    finally:
        if stream: stream.close()
        if temp: os.unlink(temp)
