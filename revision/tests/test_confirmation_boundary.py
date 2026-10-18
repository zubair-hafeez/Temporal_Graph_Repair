import json

import pytest

from revision import confirmation


def test_missing_freeze_keeps_confirmation_sealed(tmp_path, monkeypatch):
    monkeypatch.setattr(confirmation, 'ROOT', tmp_path)
    with pytest.raises(RuntimeError, match='remains sealed'):
        confirmation.verify_protocol()
    assert not list(tmp_path.iterdir())


def test_modified_job_cannot_create_a_confirmation_task(tmp_path, monkeypatch):
    monkeypatch.setattr(confirmation, 'ROOT', tmp_path)
    job = {'id': 'declared', 'thresholds': [.95, .95, 1.1]}
    with pytest.raises(ValueError, match='Undeclared coupled job'):
        confirmation.evaluate_job(dict(job, thresholds=[.5, .5, .5]), {'jobs': [job]})
    assert not list(tmp_path.iterdir())


def test_atomic_json_publication_leaves_one_complete_report(tmp_path):
    path = tmp_path / 'report.json'
    confirmation.write_json(path, {'completed': False, 'items': list(range(10))})
    confirmation.write_json(path, {'completed': True, 'items': list(range(20))})
    assert json.loads(path.read_text()) == {'completed': True, 'items': list(range(20))}
    assert list(tmp_path.iterdir()) == [path]
