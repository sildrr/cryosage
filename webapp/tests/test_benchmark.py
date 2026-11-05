import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
from webapp import benchmark
from benchmark_budget import BenchmarkBudgetReached, protect_client, budget_status


class FakeClient:
    def __init__(self, count=0):
        self.jobs = [SimpleNamespace(uid='J'+str(i+1)) for i in range(count)]
        self.fail = False

    def find_project(self, project):
        return SimpleNamespace(find_workspace=lambda workspace: SimpleNamespace(find_jobs=lambda: self.jobs))

    def create_job(self, project_uid, workspace_uid, type, **kwargs):
        job = SimpleNamespace(uid='J'+str(len(self.jobs)+1))
        self.jobs.append(job)
        if self.fail:
            raise RuntimeError('Connection failed after job creation')
        return job


@pytest.fixture
def capped(tmp_path, monkeypatch):
    path = tmp_path/'budget.json'
    path.write_text(json.dumps(dict(project='P1',workspace='W1',maximum=40)))
    monkeypatch.setenv('CRYOAGENT_BENCHMARK_BUDGET_FILE', str(path))
    return path


def test_actual_job_cap_includes_preexisting_pilots_and_failed_ids(capped):
    cs = protect_client(FakeClient(39))
    assert cs.create_job('P1', 'W1', 'homo_refine_new').uid == 'J40'
    with pytest.raises(BenchmarkBudgetReached):
        cs.create_job('P1', 'W1', 'validation')
    assert budget_status(cs)['used'] == 40
    assert len(list(capped.parent.glob('budget_receipts/*.json'))) == 1


def test_ambiguous_create_is_not_blindly_retried(capped):
    cs = protect_client(FakeClient())
    cs.fail = True
    with pytest.raises(RuntimeError, match='Connection failed'):
        cs.create_job('P1', 'W1', 'blob_picker_gpu')
    assert len(cs.jobs) == 1
    with pytest.raises(RuntimeError, match='Ambiguous prior'):
        cs.create_job('P1', 'W1', 'blob_picker_gpu')
    assert len(cs.jobs) == 1


def test_workspace_escape_and_double_install(capped):
    cs = protect_client(FakeClient())
    original = cs.create_job
    assert protect_client(cs).create_job is original
    with pytest.raises(ValueError, match='outside assigned'):
        cs.create_job('P1', 'W2', 'blob_picker_gpu')
    assert not cs.jobs


def test_no_profile_does_not_wrap_clients(monkeypatch):
    monkeypatch.delenv('CRYOAGENT_BENCHMARK_BUDGET_FILE', raising=False)
    cs = FakeClient()
    assert protect_client(cs) is cs
    assert not getattr(cs, '_benchmark_protected', False)
    assert budget_status() is None


def test_profile_matches_reference_and_uses_actual_jobs_separately():
    ref = json.loads((benchmark.BENCHMARK/'repeats/repeat03/codex/empiar11280/workspace/config.json').read_text())
    config = dict(empiar_id='11280', input_type='movies', input_path=ref['input_glob'],
        pixel_size_A=.856, voltage_kV=300, cs_mm=2.7, dose=68.4, target=3.5,
        symmetry=None, kb=True, max_jobs=40)
    result = benchmark.validate_profile(dict(benchmark_profile=benchmark.PROFILE,benchmark_repeat=1),config.copy())
    assert result['max_created_jobs'] == 40 and result['max_jobs'] == 80
    assert result['benchmark_reference_target'] == 3.96
    with pytest.raises(ValueError, match='Acquisition'):
        benchmark.validate_profile(dict(benchmark_profile=benchmark.PROFILE,benchmark_repeat=1),dict(config,dose=57.6))


def test_kb_bridge_cannot_fall_back_to_original_database(tmp_path, monkeypatch):
    db = tmp_path/'kb.sqlite'
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE docs(doc_id TEXT,kb_type TEXT,title TEXT,text TEXT,dataset_id TEXT,empiar_id TEXT,job_type TEXT,source_url TEXT)')
    con.execute("CREATE VIRTUAL TABLE docs_fts USING fts5(title,text,content='docs',content_rowid='rowid')")
    con.execute('INSERT INTO docs VALUES(?,?,?,?,?,?,?,?)',('safe','cryosparc_official_doc','Blob picker','generic particle picking','99999','99999','blob_picker','official'))
    con.execute("INSERT INTO docs_fts(docs_fts) VALUES('rebuild')")
    con.commit()
    con.close()
    monkeypatch.setenv('CRYOAGENT_BENCHMARK_KB_SNAPSHOT',str(db))
    monkeypatch.setenv('CRYOAGENT_BENCHMARK_KB_AUDIT',str(tmp_path/'queries.jsonl'))
    import kb_bridge
    monkeypatch.setattr(kb_bridge,'_load_module',lambda: pytest.fail('Original KB must not load'))
    result = kb_bridge.call_kb_tool('search_cryoem_kb',dict(query='particle'))
    assert [r['doc_id'] for r in result['results']] == ['safe']
    assert not kb_bridge.call_kb_tool('get_workflow',dict(dataset_id='11280'))['results']
    assert not kb_bridge.call_kb_tool('get_images',dict(dataset_id='11280'))['results']
    assert (tmp_path/'queries.jsonl').exists()
