const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    textContent: 'old detail', value: 'old target', children: ['old image'],
    replaceChildren(...children) { this.children = children; this.textContent = ''; },
    append(child) { this.children.push(child); },
  });
  return elements.get(id);
}
let closed = 0;
const context = vm.createContext({
  $: element, document: {querySelectorAll: () => [{close() { closed++; }}]},
  node: (tag, text) => ({tag, text}), timestamp: String,
  drawFSC() {}, appendGallery() {}, toast() { throw new Error('Unexpected toast'); },
});
vm.runInContext(`
  let inspectedJob='', detailSerial=0, inspectorTask='', inspectorKey='';
  let gallerySignature='old', galleryGeneration=0;
  const galleryObservers=new Set();
  let currentTask='old', token='test', loading=false, listGeneration=0, archivedView=false;
  let renderCalls=0;
  function render(){renderCalls++;}
`, context);
const reset = source.slice(source.indexOf('function resetJobInspector('), source.indexOf('const baseRender=render;'));
const load = source.slice(source.indexOf('async function loadDetails('), source.indexOf("document.querySelectorAll('[data-control]').forEach(button=>button.onclick"));
const refresh = source.slice(source.indexOf('async function refresh()'), source.indexOf('function render(task)'));
vm.runInContext(reset + load + refresh, context);

async function main() {
  let requests = 0;
  context.api = async () => { requests++; throw new Error('No request expected'); };
  await vm.runInContext("loadDetails({id:'new',progress:{nodes:[]}})", context);
  assert.equal(requests, 0);
  for (const id of ['job-times', 'job-parameters', 'fsc-chart', 'evidence-gallery', 'control-history']) {
    assert.deepEqual(element(id).children, []);
  }
  assert.equal(element('operator-message').value, '');
  assert.equal(element('operator-target').value, '');
  assert.equal(closed, 1);
  const emptyReason = element('job-reason').textContent;

  let resolveActivity;
  context.api = () => new Promise(resolve => { resolveActivity = resolve; });
  const pending = vm.runInContext("inspectedJob='J1';loadDetails({id:'old',progress:{nodes:[]}})", context);
  assert.notEqual(element('job-reason').textContent, emptyReason);
  await vm.runInContext("inspectedJob='';loadDetails({id:'new',progress:{nodes:[]}})", context);
  resolveActivity({});
  await pending;
  assert.equal(element('job-reason').textContent, emptyReason);
  assert.deepEqual(element('evidence-gallery').children, []);

  let resolveTasks;
  context.api = () => new Promise(resolve => { resolveTasks = resolve; });
  const pendingRefresh = vm.runInContext('refresh()', context);
  vm.runInContext("currentTask='new'", context);
  resolveTasks([{id: 'old'}]);
  await pendingRefresh;
  assert.equal(vm.runInContext('currentTask', context), 'new');
  assert.equal(vm.runInContext('renderCalls', context), 0);
  assert.match(source, /sessionStorage\.setItem\('cryosage-task',currentTask\);render\(task\);\s*toast/);
  console.log('Inspector switching regression tests passed');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
