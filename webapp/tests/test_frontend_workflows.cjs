const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
const html = fs.readFileSync(path.join(__dirname, '../static/index.html'), 'utf8');
function setup() {
  const elements = new Map(), calls = [], messages = [], confirmations = [];
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      value: '', textContent: '', disabled: false, required: false, files: [], children: [],
      replaceChildren(...children) { this.children = children; },
      append(...children) { this.children.push(...children); },
      classList: { toggle() {} },
    });
    return elements.get(id);
  };
  const fields = Object.fromEntries(['input_type', 'dose', 'pixel_size_A', 'voltage_kV', 'cs_mm'].map(name => [name, element(name)]));
  fields.input_type.value = 'micrographs';
  for (const field of ['dose', 'pixel_size_A', 'voltage_kV', 'cs_mm']) fields[field].required = true;
  element('task-form').elements = fields;
  const context = vm.createContext({
    $: element, URLSearchParams, sessionStorage: {setItem() {}, removeItem() {}},
    node: (tag, text, className) => ({tag, text, className}),
    toast: text => messages.push(text), resetJobInspector() {},
    confirm: text => { confirmations.push(text); return context.confirmResult; },
    confirmResult: true, response: () => [],
    api: async (url, options) => { calls.push({url, options}); return context.response(url, options); },
  });
  vm.runInContext("let token='test', currentTask='', selected=null, loading=false, uploading=false, archivedView=false, listGeneration=0, inspectorKey='', inspectedTask='', inspectedJob=''; let rendered=[]; function render(task){selected=task;rendered.push(task);}", context);
  vm.runInContext(source.slice(source.indexOf('function updateInputMode()'), source.indexOf('function render(task)')), context);
  return {context, element, fields, calls, messages, confirmations, run: code => vm.runInContext(code, context)};
}

async function main() {
  assert.equal(/[\p{Script=Han}]/u.test(source + html), false, 'Static UI must be English');
  assert.match(html, /lang="en"/);
  assert.match(html, /src="\/showcase\/10025-shape.png"/);
  assert.doesNotMatch(html, /<section class="how">|<svg/);
  assert.match(html, /mailto:lisongya21@mails.tsinghua.edu.cn/);
  assert.match(source, /decision\?\.reason\|\|decision\?\.reason_note/);
  assert.match(source, /reason\+='\\n\\nRaw planner reason:\\n'\+decision.planner_reason/);

  const h = setup();
  h.fields.input_type.value = 'particles'; h.run('updateInputMode()');
  assert.equal(h.element('particle-star-path').required, true);
  assert.equal(h.element('particle-star-path').disabled, false);
  assert.equal(h.element('input-path').required, true);
  assert.equal(h.fields.dose.required, false);
  for (const field of ['pixel_size_A', 'voltage_kV', 'cs_mm']) assert.equal(h.fields[field].required, true);
  h.fields.input_type.value = 'movies'; h.run('updateInputMode()');
  assert.equal(h.fields.dose.required, true);
  assert.equal(h.element('particle-star-path').disabled, true);

  h.run('renderUsage({api_calls:0,input_tokens:1234,output_tokens:23,total_tokens:1257,cached_tokens:100})');
  assert.equal(h.element('usage-api-calls').textContent, '0');
  assert.equal(h.element('usage-input-tokens').textContent, '1,234');
  h.run('renderUsage(null)');
  assert.equal(h.element('usage-total-tokens').textContent, '—');
  h.run('renderUsage({source:"ledger",report_scope:"exact_http_attempts",has_usage:true,api_calls:2,api_calls_complete:true,tokens_complete:false,input_tokens:0,output_tokens:0,total_tokens:0,cached_tokens:0,calls_without_usage:2,failed_calls:1,failed_calls_complete:true,note:"Unreported tokens are unknown."})');
  assert.equal(h.element('usage-api-calls').textContent, '2');
  assert.equal(h.element('usage-total-tokens').textContent, '—');
  assert.match(h.element('usage-note').textContent, /HTTP attempts since ledger tracking began/);
  assert.match(h.element('usage-note').textContent, /Unreported tokens are unknown/);
  assert.match(h.element('usage-note').textContent, /Calls without provider usage: 2/);
  assert.match(h.element('usage-note').textContent, /Failed API calls: 1/);
  h.run('renderUsage({source:"legacy",has_usage:false,api_calls:0,api_calls_complete:false,tokens_complete:false,total_tokens:0,report_scope:"approximate_recorded_responses"})');
  assert.equal(h.element('usage-api-calls').textContent, '—');
  assert.equal(h.element('usage-total-tokens').textContent, '—');
  assert.match(h.element('usage-note').textContent, /Approximate historical/);
  h.run('renderUsage({source:"ledger",has_usage:false,api_calls:0,api_calls_complete:true,tokens_complete:true,total_tokens:0,input_tokens:0,output_tokens:0,cached_tokens:0})');
  assert.equal(h.element('usage-api-calls').textContent, '0');
  assert.equal(h.element('usage-total-tokens').textContent, '0');
  h.run('renderUsage({source:"legacy",has_usage:true,api_calls:3,api_calls_complete:false,tokens_complete:false,total_tokens:123,failed_calls:0,failed_calls_complete:false})');
  assert.equal(h.element('usage-total-tokens').textContent, '123');
  assert.match(h.element('usage-note').textContent, /partial recorded subtotals/);
  assert.match(h.element('usage-note').textContent, /Failed API calls: unknown/);

  h.fields.input_type.value = 'particles'; h.run('updateInputMode()');
  h.element('input-path').value = '/mounted/stacks';
  h.element('files').files = [{name:'particles.star',size:20}];
  h.context.response = () => ({group:'group',file_path:'/uploads/group/particles.star',input_path:'/uploads/group/*'});
  await h.element('files').onchange();
  assert.equal(h.element('input-path').value, '/mounted/stacks', 'STAR-only upload preserves folder');
  assert.equal(h.element('particle-star-path').value, '/uploads/group/particles.star');

  for (const names of [['particles.star','stack.mrcs'], ['stack.mrcs','particles.star']]) {
    h.element('files').files = names.map(name => ({name,size:20}));
    h.context.response = url => {
      const filename = new URLSearchParams(url.split('?')[1]).get('filename');
      return {group:'flat',file_path:'/uploads/flat/'+filename,input_path:'/uploads/flat/*'};
    };
    await h.element('files').onchange();
    assert.equal(h.element('input-path').value, '/uploads/flat/*.mrcs');
    assert.equal(h.element('particle-star-path').value, '/uploads/flat/particles.star');
  }
  h.element('input-path').value = '/original/stacks';
  h.element('particle-star-path').value = '/original/particles.star';
  h.context.response = () => ({group:'missing',input_path:'/uploads/missing/*'});
  h.element('files').files = [{name:'particles.star',size:20}];
  await h.element('files').onchange();
  assert.equal(h.element('input-path').value, '/original/stacks');
  assert.equal(h.element('particle-star-path').value, '/original/particles.star');
  assert.equal(h.element('submit-task').disabled, false);

  h.context.response = () => [];
  h.calls.length = 0; h.element('task-filter').value = 'archived';
  h.element('task-filter').onchange();
  await new Promise(setImmediate);
  assert.equal(h.calls[0].url, '/api/tasks?archived=true');
  assert.equal(h.element('task-detail').hidden, true);
  h.element('task-filter').value = 'normal'; h.element('task-filter').onchange();
  await new Promise(setImmediate);
  assert.equal(h.calls.at(-1).url, '/api/tasks');

  for (const archived of [false, true]) {
    h.run('selected={id:"test",archived:'+archived+'}');
    h.calls.length = 0;
    await h.run('manageTask("archive")');
    assert.equal(h.calls[0].url, '/api/tasks/test/archive');
    assert.deepEqual(JSON.parse(h.calls[0].options.body), {archived:!archived});
  }
  h.run('selected={id:"test",archived:false}');
  h.calls.length = 0; h.context.confirmResult = false;
  await h.run('manageTask("delete")');
  assert.equal(h.calls.length, 0);
  assert.match(h.confirmations.at(-1), /CryoSPARC.*source files remain untouched/);
  h.context.confirmResult = true;
  await h.run('manageTask("delete")');
  assert.equal(h.calls[0].url, '/api/tasks/test/delete');
  assert.equal(h.calls[0].options.method, 'POST');
  assert.equal(h.calls[0].options.body, '{}');

  const stale = setup();
  let resolveOld;
  stale.context.response = url => url.includes('archived=true') ? [] : new Promise(resolve => { resolveOld=resolve; });
  const pending = stale.run('refresh()');
  stale.element('task-filter').value = 'archived'; stale.element('task-filter').onchange();
  resolveOld([{id:'old',config:{name:'Old'}}]); await pending;
  await new Promise(setImmediate);
  assert.equal(stale.run('rendered.length'), 0, 'Old list must not leak into archived view');
  assert.equal(stale.calls.at(-1).url, '/api/tasks?archived=true');
  console.log('Frontend workflow regressions passed (English UI, particles, uploads, usage, archive/delete, stale refresh).');
}
main().catch(error => { console.error(error); process.exitCode=1; });
