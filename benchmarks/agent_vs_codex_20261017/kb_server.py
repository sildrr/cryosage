"""Read-only MCP retrieval against the frozen benchmark KB, not the original DB."""
import json
import os
import re
import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DB = Path(os.environ['BENCHMARK_KB_SNAPSHOT'])
AUDIT = Path(os.environ['BENCHMARK_KB_AUDIT'])
mcp = FastMCP('benchmark-heldout-kb')
DOC_TYPES = ('cryosparc_official_doc', 'cryosparc_job_type', 'cryosparc_job_param', 'cryosparc_job_io')


def query_docs(query, top_k=5, types=None, job_type=None, doc_id=None):
    top_k = min(max(int(top_k), 1), 10)
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    clauses, args = [], []
    if types:
        clauses.append('d.kb_type IN (' + ','.join('?' for _ in types) + ')')
        args.extend(types)
    if job_type:
        clauses.append('d.job_type = ?')
        args.append(job_type)
    if doc_id:
        clauses.append('d.doc_id = ?')
        args.append(doc_id)
    tokens = re.findall(r'[\w]+', query or '')[:16]
    join = ''
    if tokens:
        join = ' JOIN docs_fts f ON f.rowid=d.rowid'
        clauses.append('docs_fts MATCH ?')
        args.append(' OR '.join('"' + token + '"' for token in tokens))
    try:
        where = (' WHERE ' + ' AND '.join(clauses)) if clauses else ''
        rows = [dict(r) for r in con.execute(
            'SELECT d.doc_id,d.kb_type,d.title,d.dataset_id,d.empiar_id,d.job_type,d.source_url,'
            'substr(d.text,1,8000) AS text FROM docs d' + join + where + ' LIMIT ?', [*args, top_k])]
    finally:
        con.close()
    result = dict(source='frozen_heldout_kb', results=rows, top_k=top_k,
                  notice='Historical cases are fallible precedents, not mandatory settings. Target-specific source records are excluded. Use only this snapshot, not the source KB or linked local files.')
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT.open('a') as stream:
        stream.write(json.dumps(dict(query=query, job_type=job_type, doc_id=doc_id,
                                     returned_doc_ids=[r['doc_id'] for r in rows])) + '\n')
    return result


@mcp.tool()
def search_cryoem_kb(query: str, top_k: int = 5) -> dict:
    """Search allowed historical cases and general documents. Never query target identity."""
    return query_docs(query, top_k)


@mcp.tool()
def get_job_doc(job_type: str, query: str = '', top_k: int = 5) -> dict:
    """Read generic official documentation, parameters and IO specifications."""
    aliases = {'blob_picker_gpu': 'blob_picker', 'template_picker_gpu': 'template_picker',
               'class_2D_new': '2d_classification', 'homo_refine_new': 'homogeneous_refinement',
               'nonuniform_refine_new': 'non_uniform_refinement',
               'patch_ctf_estimation_multi': 'patch_ctf_estimation',
               'inspect_picks_v2': 'inspect_particle_picks',
               'extract_micrographs_multi': 'extract_from_micrographs',
               'homo_abinit': 'ab_initio_reconstruction'}
    result = query_docs(query, top_k, DOC_TYPES, job_type)
    if not result['results'] and job_type in aliases:
        result = query_docs(query, top_k, DOC_TYPES, aliases[job_type])
    if not result['results']:
        result = query_docs(query or aliases.get(job_type, job_type).replace('_', ' '), top_k, DOC_TYPES)
    return result


@mcp.tool()
def get_kb_record(doc_id: str) -> dict:
    """Read one allowed retrieved record from this filtered snapshot only."""
    return query_docs('', 1, doc_id=doc_id)


if __name__ == '__main__':
    mcp.run(transport='stdio')
