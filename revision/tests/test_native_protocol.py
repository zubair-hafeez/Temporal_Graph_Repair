import json

import pytest

from revision import native_protocol
from tgrepair.data import sha256


def test_native_confirmation_rejects_undeclared_jobs_and_changed_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(native_protocol, 'ROOT', tmp_path)
    (tmp_path / 'revision/docs').mkdir(parents=True)
    dependency = tmp_path / 'observed.txt'
    dependency.write_text('frozen observations')
    protocol = {'status': 'frozen_before_native_outcomes',
                'jobs': [['bpi2017', 0, .3, 30, 2048]],
                'implementation': {}, 'models': {}, 'inputs': {'observed.txt': sha256(dependency)}}
    (tmp_path / 'revision/docs/NATIVE_CONFIRMATION_PROTOCOL_V1.json').write_text(json.dumps(protocol))
    assert native_protocol.verify_protocol('bpi2017', 0, .3, 30, 2048) == protocol
    with pytest.raises(ValueError, match='Undeclared'):
        native_protocol.verify_protocol('bpi2017', 1, .3, 30, 2048)
    dependency.write_text('modified observations')
    with pytest.raises(ValueError, match='dependency changed'):
        native_protocol.verify_protocol('bpi2017', 0, .3, 30, 2048)
