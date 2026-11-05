const $ = (id) => document.getElementById(id);
let token = sessionStorage.getItem('cryosage-token') || '';
let currentTask = sessionStorage.getItem('cryosage-task') || '';
let selected = null;
let loading = false;
let uploading = false;
let toastTimer;
let archivedView = false;
let listGeneration = 0;
const stateNames = {queued:'Queued',running:'Running',completed:'Completed',finished:'Finished',failed:'Failed',building:'Created',needs_attention:'Needs Attention',controller_failed:'Controller Failed',submission_failed:'Submission Failed',launched:'Starting',waiting:'Waiting for Input',killed:'Stopped',user_stopped:'Stopped by User'};
function toast(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,6000);}
async function api(path, options={}){const response=await fetch(path,{...options,headers:{Authorization:`Bearer ${token}`,...(options.headers||{})}});const data=await response.json();if(!response.ok)throw new Error(data.error||'Request failed');return data;}
function node(tag,text,className){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(className)el.className=className;return el;}
function route(){const work=location.hash.startsWith('#workflow');$('home').hidden=work;$('workflow').hidden=!work;$('home-link').classList.toggle('active',!work);$('workflow-link').classList.toggle('active',work);if(work&&token)refresh();}
addEventListener('hashchange',route);
$('unlock').onclick=()=>{$('auth-dialog').showModal();};$('close-auth').onclick=()=>$('auth-dialog').close();
$('auth-form').onsubmit=async(e)=>{e.preventDefault();token=$('access-token').value.trim();try{await api('/api/tasks');sessionStorage.setItem('cryosage-token',token);$('auth-dialog').close();$('access-token').value='';$('unlock').textContent='Console Unlocked';toast('Console Unlocked');refresh();}catch(err){token='';sessionStorage.removeItem('cryosage-token');toast(err.message);}};
async function showcase(){
  try{
    const results=await api('/api/showcase');$('showcase').replaceChildren();
    for(const result of results){
      const card=node('article',undefined,'result-card'),copy=node('div');
      copy.append(node('p',`EMPIAR-${result.id}`,'eyebrow'),node('h3',result.name),node('p','Final reconstruction · FSC resolution'));
      const score=node('div',result.resolution.toFixed(2),'resolution');score.append(node('small',' Å'));card.append(copy,score);
      const previews=node('div',undefined,'showcase-previews');
      for(const [kind,label] of [['fsc','Final FSC Curve'],['shape','3D Density Map · Single View']]){
        const figure=node('figure',undefined,`showcase-${kind}`),button=node('button',undefined,'showcase-image');
        const img=node('img');img.src=`/showcase/${encodeURIComponent(result.id)}-${kind}.png`;img.alt=`EMPIAR-${result.id} · ${label}`;img.loading='lazy';
        button.setAttribute('aria-label',`${img.alt}, click to enlarge`);
        button.onclick=()=>showFullImage(img.src,img.alt);
        img.onerror=()=>{button.replaceChildren(node('span','Result image unavailable'));button.disabled=true;};
        button.append(img);figure.append(button,node('figcaption',label));previews.append(figure);
      }
      card.append(previews,node('p','Density isosurfaces preview shape only; they do not validate structural correctness.','showcase-note'));
      const foot=node('div',undefined,'provenance');foot.append(node('span',`${result.project}/${result.workspace}/${result.job} · ${result.metric}`));
      const source=node('button','View Source','quiet');source.onclick=()=>toast(`Source: ${result.source}`);foot.append(source);card.append(foot);$('showcase').append(card);
    }
  }catch(err){$('showcase').textContent='Recorded results are temporarily unavailable.';}
}
function updateInputMode(){
  const form=$('task-form'),particles=form.elements.input_type.value==='particles';
  $('particle-star-field').hidden=!particles;
  $('particle-star-path').disabled=!particles;
  $('particle-star-path').required=particles;
  $('input-path').required=true;
  $('input-path-label').textContent=particles?'MRCS Stack Folder on Server':'Server Data Path';
  $('input-path').placeholder=particles?'/server/particles/stacks':'/home/share/empiar/10025/data/*.mrc';
  $('input-path-note').textContent=particles?'Use an existing server MRCS folder and its corresponding STAR file, especially for nested image references. Flat MRCS + STAR uploads are supported; directory trees are not uploaded or flattened.':'Wildcards supported. For large datasets, use an existing server path.';
  form.elements.dose.required=!particles;
  $('dose-note').textContent=particles?'Dose is optional for particles; the backend uses its default when omitted.':'Dose is required for micrographs and movies.';
}
$('task-form').elements.input_type.onchange=updateInputMode;
updateInputMode();
$('example').onclick=()=>{
  const values={name:'20S proteasome · EMPIAR-10025',empiar_id:'10025',input_type:'micrographs',input_path:'/home/share/empiar/10025/data/14sep05c_averaged_196/*.mrc',pixel_size_A:.6575,voltage_kV:300,cs_mm:2.7,dose:53,target:2.8,symmetry:'D7'};
  for(const [key,value]of Object.entries(values))$('task-form').elements[key].value=value;
  updateInputMode();toast('Historical example parameters filled in. Verify before submitting.');
};
$('files').onchange=async()=>{
  if(!token){$('auth-dialog').showModal();return;}
  const files=[...$('files').files];if(!files.length)return;
  const particles=$('task-form').elements.input_type.value==='particles';
  if(particles&&files.some(file=>! /\.(mrcs|star)$/i.test(file.name))){toast('Particle uploads support flat MRCS stacks and one corresponding STAR file. Use server paths for other stack layouts.');return;}
  if(files.some(file=>/\.star$/i.test(file.name))&&!particles){toast('Select Particles mode before uploading STAR metadata.');return;}
  if(files.filter(file=>/\.star$/i.test(file.name)).length>1){toast('Upload one STAR metadata file per workflow.');return;}
  uploading=true;$('submit-task').disabled=true;$('task-form').elements.input_type.disabled=true;
  let group,uploadedPath='',uploadedStar='';
  try{
    for(let i=0;i<files.length;i++){
      const file=files[i];
      if(file.size>8*1024**3)throw new Error('File exceeds 8 GiB. Use a server path instead.');
      $('upload-status').textContent=`Uploading ${i+1}/${files.length}: ${file.name} (keep this page open)`;
      const query=new URLSearchParams({filename:file.name});if(group)query.set('group',group);
      const data=await api(`/api/uploads?${query}`,{method:'POST',body:file});group=data.group;
      if(/\.star$/i.test(file.name)){
        if(!data.file_path)throw new Error('The upload response did not include the STAR file_path. Enter its server path manually.');
        uploadedStar=data.file_path;
      }else{
        if(!data.input_path)throw new Error('The upload response did not include a data path.');
        if(particles&&!data.file_path)throw new Error('The upload response did not include the MRCS file_path. Enter its server folder manually.');
        uploadedPath=particles?data.file_path.slice(0,data.file_path.lastIndexOf('/')+1)+'*.mrcs':data.input_path;
      }
    }
    // Do not replace an existing MRCS folder when uploading only STAR metadata.
    if(uploadedPath)$('input-path').value=uploadedPath;
    if(uploadedStar)$('particle-star-path').value=uploadedStar;
    $('upload-status').textContent=`${files.length} file(s) uploaded. Verify paths, input type, and acquisition parameters.`;
  }catch(err){
    toast(err.message);$('upload-status').textContent='Upload incomplete. Re-select the files or enter server paths; partial upload paths were not applied.';
  }finally{uploading=false;$('submit-task').disabled=false;$('task-form').elements.input_type.disabled=false;}
};
$('task-form').onsubmit=async(e)=>{
  e.preventDefault();if(uploading)return;if(!token){$('auth-dialog').showModal();return;}
  const form=new FormData(e.target),body=Object.fromEntries(form);
  body.kb=form.has('kb');body.external_consent=form.has('external_consent');
  if(body.input_type==='particles'&&!body.dose)delete body.dose;
  $('submit-task').disabled=true;
  try{
    const health=await api('/api/health');
    if(health.pipeline_version!=='web-pilot-to-full-v1')throw new Error('Restart this account\'s web service before submitting: the backend has not loaded the pilot-to-full strategy update.');
    const task=await api('/api/tasks',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    listGeneration++;archivedView=false;$('task-filter').value='normal';
    currentTask=task.id;sessionStorage.setItem('cryosage-task',currentTask);render(task);
    toast(`Workflow submitted, Slurm ${task.slurm_job}`);await refresh();
  }catch(err){toast(err.message);}finally{$('submit-task').disabled=false;}
};
function clearTask(){
  selected=null;currentTask='';sessionStorage.removeItem('cryosage-task');
  $('task-detail').hidden=true;$('empty-state').hidden=false;
  resetJobInspector('','');inspectorKey='';inspectedTask='';inspectedJob='';
}
async function refresh(){
  if(!token)return;
  if(loading){listGeneration++;return;}
  loading=true;
  const generation=listGeneration,requestedTask=currentTask,requestedView=archivedView;
  try{
    const tasks=await api('/api/tasks'+(requestedView?'?archived=true':''));
    if(generation!==listGeneration||currentTask!==requestedTask)return;
    $('task-list').replaceChildren();
    if(!tasks.length){clearTask();$('task-list').append(node('span',requestedView?'No archived workflows.':'No submitted workflows.','field-note'));return;}
    if(!tasks.some(t=>t.id===currentTask))currentTask=tasks[0].id;
    sessionStorage.setItem('cryosage-task',currentTask);
    for(const task of tasks){
      const button=node('button',task.config.name,'task-tab'+(task.id===currentTask?' selected':''));
      button.onclick=()=>{currentTask=task.id;sessionStorage.setItem('cryosage-task',currentTask);render(task);refresh();};
      $('task-list').append(button);
    }
    render(tasks.find(t=>t.id===currentTask));
  }catch(err){if(generation===listGeneration&&currentTask===requestedTask)toast(err.message);}
  finally{loading=false;if(generation!==listGeneration)refresh();}
}
$('task-filter').onchange=()=>{
  archivedView=$('task-filter').value==='archived';listGeneration++;clearTask();refresh();
};
function renderUsage(usage={}){
  const emptyLedger=usage?.source==='ledger'&&usage.api_calls===0&&usage.api_calls_complete===true&&usage.tokens_complete===true;
  const noHistory=usage?.has_usage===false&&!emptyLedger;
  for(const field of ['api_calls','input_tokens','output_tokens','total_tokens','cached_tokens']){
    const value=usage?.[field];
    const incomplete=field==='api_calls'?usage?.api_calls_complete===false:usage?.tokens_complete===false;
    const known=typeof value==='number'&&Number.isFinite(value)&&value>=0&&!noHistory&&!(value===0&&incomplete);
    const element=$('usage-'+field.replaceAll('_','-'));
    element.textContent=known?value.toLocaleString('en-US'):'—';
    element.title=incomplete?'Recorded subtotal only. Missing usage is unknown, not zero.':'Backend-reported usage';
  }
  if(!usage){$('usage-note').textContent='Usage data unavailable.';return;}
  const scope={exact_http_attempts:'HTTP attempts since ledger tracking began',approximate_recorded_responses:'Approximate historical recorded responses'}[usage.report_scope]||usage.report_scope;
  const notes=[];
  if(scope)notes.push('Scope: '+scope+'.');
  if(usage.note)notes.push(usage.note);
  if(usage.api_calls_complete===false)notes.push('API call count is partial; unrecorded calls may be missing.');
  if(usage.tokens_complete===false)notes.push('Token totals are partial recorded subtotals. Unreported consumption is unknown, not zero.');
  else if(usage.tokens_complete===true)notes.push('Token usage is complete for recorded attempts.');
  if(Number.isFinite(usage.calls_without_usage))notes.push('Calls without provider usage: '+usage.calls_without_usage.toLocaleString('en-US')+'.');
  if(Number.isFinite(usage.failed_calls)){
    const failed=usage.failed_calls===0&&usage.failed_calls_complete===false?'unknown':usage.failed_calls.toLocaleString('en-US');
    notes.push('Failed API calls: '+failed+(usage.failed_calls_complete===false?' (partial records).':'.'));
  }
  if(usage.unreadable_files>0)notes.push('Unreadable usage records/files: '+usage.unreadable_files.toLocaleString('en-US')+'.');
  if(noHistory)notes.push('No usable call history is available; dashes do not mean zero consumption.');
  $('usage-note').textContent=notes.join(' ')||'Usage data unavailable.';
}
let managingTask=false;
async function manageTask(action){
  if(!selected||managingTask)return;
  const task=selected,archived=!(task.archived??archivedView);
  if(action==='delete'&&!confirm('Delete this web workflow? This only hides/removes the workflow from the web app. CryoSPARC jobs, project data, and source files remain untouched. Running workflows cannot be deleted.'))return;
  managingTask=true;$('archive-task').disabled=true;$('delete-task').disabled=true;
  try{
    await api(`/api/tasks/${encodeURIComponent(task.id)}/${action}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(action==='archive'?{archived}:{})});
    listGeneration++;clearTask();toast(action==='delete'?'Web workflow removed. CryoSPARC data and source files were not deleted.':archived?'Workflow archived.':'Workflow restored.');
    await refresh();
  }catch(err){toast(err.message);}
  finally{managingTask=false;if(selected)render(selected);}
}
$('archive-task').onclick=()=>manageTask('archive');
$('delete-task').onclick=()=>manageTask('delete');
function render(task){selected=task;renderUsage(task.usage);$('archive-task').textContent=(task.archived??archivedView)?'Unarchive Workflow':'Archive Workflow';$('archive-task').disabled=managingTask;$('delete-task').disabled=managingTask||['running','queued','launched','paused','waiting','building'].includes(task.status);$('empty-state').hidden=true;$('task-detail').hidden=false;$('task-id').textContent=task.project?`${task.project}/${task.workspace} · SLURM ${task.slurm_job}`:`SLURM ${task.slurm_job}`;$('task-name').textContent=task.config.name;$('task-status').textContent=stateNames[task.status]||task.status;const progress=task.progress||{},nodes=progress.nodes||[];$('best-resolution').textContent=progress.best_refine?`${progress.best_refine.resolution_A.toFixed(2)} Å`:'—';$('target-resolution').textContent=`${task.config.target} Å`;$('completed-jobs').textContent=nodes.filter(n=>n.status==='completed').length;$('progress-note').textContent=`${progress.updated_at?`Updated: ${new Date(progress.updated_at).toLocaleString('en-US')}.`:'Waiting for the controller or first status query.'} Scheduler status: ${task.scheduler_status||'see job list'}. Page refreshes every 10 seconds; backend queries actual jobs every 30 seconds.`;
if(task.pipeline_version){const pilot=task.pilot_progress;const phase=task.config.input_type==='particles'?'Particle-input reconstruction':pilot?.full_data_promoted?'Full-data reconstruction and optimization':pilot?.pilot_accepted?'Pilot accepted; comparing routes':'Preprocessing and adaptive pilot exploration';$('progress-note').textContent+=` Strategy: ${phase}. Actual Job budget: ${nodes.length}/${task.config.max_created_jobs}.`;}
const warnings=[];if(task.monitor_warning)warnings.push('The latest progress query failed; showing the last valid snapshot.');if(task.error)warnings.push(task.error);if(['needs_attention','controller_failed'].includes(task.status))warnings.push('This workflow needs attention and may not have met its target. Available best refinement results can still be downloaded.');$('warnings').textContent=warnings.join(' ');$('warnings').hidden=!warnings.length;$('job-rows').replaceChildren();for(const n of nodes){const row=node('tr');row.append(node('td',n.id),node('td',n.type),node('td',stateNames[n.status]||n.status,`status-${n.status}`),node('td',typeof n.resolution_A==='number'?n.resolution_A.toFixed(2):'—'));$('job-rows').append(row);}if(!nodes.length){const row=node('tr');const td=node('td','No job snapshot yet; the workflow may be queued in Slurm or importing data.');td.colSpan=4;row.append(td);$('job-rows').append(row);}const result=task.result;$('deliverables').hidden=!result?.artifacts?.length;if(result){$('result-note').textContent=result.status==='available'?`${result.job_uid} · ${result.resolution_A.toFixed(2)} Å · ${result.map_kind==='sharpened'?'sharpened map':'unsharpened map'}. ${result.target_met?'Target resolution met.':'Target resolution not yet met.'} Structural correctness still requires human validation.`:'No downloadable refinement map is available. See the report.';$('download-map').disabled=result.status!=='available';$('open-cryonet').disabled=result.map_kind!=='sharpened';}$('cryonet-status').textContent=task.cryonet?`CryoNet: ${task.cryonet.status}${task.cryonet.jobid?' · '+task.cryonet.jobid:''}`:'';}
async function download(name){if(!selected)return;try{const response=await fetch(`/api/tasks/${selected.id}/artifacts/${name}`,{headers:{Authorization:`Bearer ${token}`}});if(!response.ok)throw new Error('Download failed');const url=URL.createObjectURL(await response.blob());const link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),60000);}catch(err){toast(err.message);}}
$('download-map').onclick=()=>download('density.mrc');$('download-report').onclick=()=>download('report.md');$('refresh').onclick=refresh;$('open-cryonet').onclick=()=>$('cryonet-dialog').showModal();$('close-cryonet').onclick=()=>$('cryonet-dialog').close();$('submit-cryonet').onclick=async()=>{if(!selected)return;if(!$('cryonet-consent').checked){toast('Confirm permission to upload to CryoNet.');return;}$('submit-cryonet').disabled=true;try{await api(`/api/tasks/${selected.id}/cryonet`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({fasta:$('fasta').value,external_consent:true})});$('cryonet-dialog').close();toast('CryoNet handoff started. Refresh later for submission status.');refresh();}catch(err){toast(err.message);}finally{$('submit-cryonet').disabled=false;}};
Object.assign(stateNames,{paused:'Paused at Decision Point',user_stopped:'Stopped by User'});
let inspectedJob='',inspectedTask='',detailSerial=0;
let inspectorTask='',inspectorKey='';
const imageURLs=new Map();
let gallerySignature='',galleryGeneration=0;
const galleryObservers=new Set();
function resetJobInspector(taskId,job) {
  detailSerial++;
  gallerySignature='';galleryGeneration++;
  for(const observer of galleryObservers)observer.disconnect();
  galleryObservers.clear();
  for(const id of ['job-times','job-parameters','fsc-chart','evidence-gallery','control-history'])$(id).replaceChildren();
  $('inspector-title').textContent=job?`${job} · Job Details and Visual Evidence`:'Job Details and Visual Evidence';
  $('job-reason').textContent=job?'Loading job details…':'Waiting for the first job in this workflow.';
  if(inspectorTask!==taskId){$('operator-message').value='';$('operator-target').value='';}
  inspectorTask=taskId;
  document.querySelectorAll('.image-dialog').forEach(dialog=>dialog.close());
}
const baseRender=render;
render=function(task){baseRender(task);const nodes=task.progress?.nodes||[];if(inspectedTask!==task.id){inspectedTask=task.id;inspectedJob=nodes.at(-1)?.id||'';}if(!nodes.some(n=>n.id===inspectedJob))inspectedJob=nodes.at(-1)?.id||'';[...$('job-rows').rows].forEach((row,i)=>{if(!nodes[i])return;row.dataset.job=nodes[i].id;row.classList.toggle('selected-job',nodes[i].id===inspectedJob);row.tabIndex=0;row.onclick=()=>{inspectedJob=nodes[i].id;render(task);};row.onkeydown=e=>{if(e.key==='Enter'){inspectedJob=nodes[i].id;render(task);}};});const previousTarget=$('operator-target').value;$('operator-target').replaceChildren(node('option','Keep current job; let the model choose the route'));$('operator-target').options[0].value='';for(const n of nodes.filter(n=>n.status==='completed')){const option=node('option',`${n.id} · ${n.type}`);option.value=n.id;$('operator-target').append(option);}$('operator-target').value=previousTarget;const active=['running','paused','queued'].includes(task.status);document.querySelectorAll('[data-control]').forEach(button=>{button.disabled=button.dataset.control==='resume'?!task.project:(active&&!task.controls_supported)||!active;});$('control-note').textContent=task.controls_supported?'Commands take effect at the next decision point after the current job finishes; running jobs are not killed. Guidance persists until replaced.':active?'This older controller cannot receive live commands. Resume with the updated controller to enable them.':'This workflow has stopped. Select a completed job and choose Continue / Resume to start the updated controller. New compute and API costs apply.';loadDetails(task);};
function timestamp(value){if(!value)return 'Not yet / Not recorded';const date=new Date(value);return Number.isNaN(date.getTime())?String(value):date.toLocaleString();}
function drawFSC(curve){$('fsc-chart').replaceChildren();if(!curve)return;const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 520 180');svg.setAttribute('role','img');svg.setAttribute('aria-label','Actual CryoSPARC FSC Curve');const n=curve.values.length,line=document.createElementNS(ns,'polyline');line.setAttribute('points',curve.values.map((v,i)=>`${35+i*460/Math.max(1,n-1)},${145-Math.max(-.1,Math.min(1,v))*115}`).join(' '));line.setAttribute('fill','none');line.setAttribute('stroke','#205e4d');line.setAttribute('stroke-width','2');svg.append(line);const cutoff=document.createElementNS(ns,'path');cutoff.setAttribute('d',`M35 ${145-.143*115}H495`);cutoff.setAttribute('stroke','#b66a3e');cutoff.setAttribute('stroke-dasharray','4 4');svg.append(cutoff);const label=document.createElementNS(ns,'text');label.setAttribute('x','35');label.setAttribute('y','170');label.setAttribute('font-size','10');label.textContent='Spatial frequency → / Dashed FSC=0.143';svg.append(label);$('fsc-chart').append(svg,node('p',`Actual curve field: ${curve.name} · ${curve.values.length} samples`));}
async function galleryImage(path) {
  if(imageURLs.has(path))return imageURLs.get(path);
  const response=await fetch(path,{headers:{Authorization:`Bearer ${token}`}});
  if(!response.ok)throw new Error('Image unavailable. Please retry later.');
  const url=URL.createObjectURL(await response.blob());
  imageURLs.set(path,url);
  if(imageURLs.size>160){const oldest=imageURLs.keys().next().value;URL.revokeObjectURL(imageURLs.get(oldest));imageURLs.delete(oldest);}
  return url;
}
function showFullImage(url, caption) {
  const dialog=node('dialog',undefined,'image-dialog'),close=node('button','Close','button secondary'),img=node('img');
  img.src=url;img.alt=caption;
  close.onclick=()=>dialog.close();
  dialog.onclose=()=>dialog.remove();
  dialog.append(close,node('p',caption),img);document.body.append(dialog);dialog.showModal();
}
function appendGallery(title, images, generation) {
  const gallery=$('evidence-gallery'),section=node('section',undefined,'gallery-section');
  section.append(node('h4',`${title} · ${images.length} images`));
  const grid=node('div',undefined,'gallery-grid'),more=node('button','Load More Images','button secondary');
  section.append(grid,more);gallery.append(section);
  let offset=0;
  function batch(){
    if(generation!==galleryGeneration)return;
    for(const image of images.slice(offset,offset+12)){
      const figure=node('figure'),button=node('button','Loading…','image-open'),caption=node('figcaption',image.caption);
      if(image.created_at)caption.append(node('div',timestamp(image.created_at)));
      figure.append(button,caption);grid.append(figure);
      // Only fetch figures near the viewport, not every refinement iteration at once.
      const observer=new IntersectionObserver(async entries=>{
        if(!entries.some(e=>e.isIntersecting))return;observer.disconnect();galleryObservers.delete(observer);
        if(generation!==galleryGeneration)return;
        try{const url=await galleryImage(image.path);if(generation!==galleryGeneration)return;
          const img=node('img');img.src=url;img.alt=image.caption;button.replaceChildren(img);
          button.onclick=()=>{if(generation===galleryGeneration)showFullImage(url,image.caption);};
        }catch(err){if(generation!==galleryGeneration)return;button.textContent=err.message;button.onclick=async()=>{if(generation!==galleryGeneration)return;try{const url=await galleryImage(image.path);if(generation===galleryGeneration)showFullImage(url,image.caption);}catch(e){if(generation===galleryGeneration)toast(e.message);}};}
      },{rootMargin:'200px'});
      galleryObservers.add(observer);
      observer.observe(button);
    }
    offset+=12;more.hidden=offset>=images.length;
    more.textContent=`Load More Images (${Math.max(0,images.length-offset)} remaining)`;
  }
  more.onclick=batch;batch();
}
async function loadDetails(task){
  const key=JSON.stringify([task.id,inspectedJob]);
  if(key!==inspectorKey){inspectorKey=key;resetJobInspector(task.id,inspectedJob);}
  const serial=++detailSerial,job=inspectedJob;if(!job)return;
  try{
    const activity=await api(`/api/tasks/${task.id}/activity`);if(serial!==detailSerial)return;
    const existing=(task.progress?.nodes||[]).find(n=>n.id===job)||{};
    let detail={...existing,...activity.job_details[job]};
    try{detail={...detail,...await api(`/api/tasks/${task.id}/jobs/${job}`)};}catch{detail.dashboard_warning='Job dashboard unavailable. Please refresh later.';}
    if(serial!==detailSerial)return;
    $('inspector-title').textContent=`${job} · Job Details and Visual Evidence`;$('job-times').replaceChildren();
    for(const [field,label]of [['queued_at','Queued At'],['started_at','Started At'],['completed_at','Completed At'],['failed_at','Failed At']]){
      const cell=node('div');cell.append(node('span',label),node('strong',timestamp(detail.timestamps?.[field])));$('job-times').append(cell);
    }
    const decision=activity.decisions.findLast(d=>d.created_jobs.includes(job));
    let reason=decision?.reason||decision?.reason_note||'No model reason was recorded.';
    if(typeof reason!=='string')reason=JSON.stringify(reason,null,2);
    if(job===task.import_job)reason='The system imports data using the supplied paths and acquisition parameters; this is not a model decision.';
    if(decision?.planner_reason)reason+='\n\nRaw planner reason:\n'+decision.planner_reason;
    $('job-reason').textContent=reason;
    $('job-parameters').textContent=JSON.stringify({parameters:detail.parameters||decision?.parameters||{},cluster_job_id:detail.cluster_job_id,cluster_status:detail.cluster_status,errors:detail.errors},null,2);
    drawFSC(detail.fsc_curve);
    const dashboard=(detail.dashboard_images||[]).map(image=>({...image,path:`/api/tasks/${task.id}/jobs/${job}/assets/${image.asset_id}`}));
    const evidence=(decision?.images||[]).map(image=>({...image,path:`/api/tasks/${task.id}/images/${image.file}`}));
    const signature=JSON.stringify([task.id,job,dashboard,evidence,detail.dashboard_warning]);
    if(signature!==gallerySignature){
      gallerySignature=signature;galleryGeneration++;
      for(const observer of galleryObservers)observer.disconnect();galleryObservers.clear();
      $('evidence-gallery').replaceChildren();
      if(detail.dashboard_warning)$('evidence-gallery').append(node('p',detail.dashboard_warning,'field-note'));
      if(dashboard.length)appendGallery('Original CryoSPARC Job Dashboard (Newest First; Click to Enlarge)',dashboard,galleryGeneration);
      else if(!detail.dashboard_warning)$('evidence-gallery').append(node('p','No CryoSPARC image events are available for this job yet.','field-note'));
      if(evidence.length)appendGallery('Input Evidence Used for the Model Decision',evidence,galleryGeneration);
    }
    $('control-history').replaceChildren();
    for(const command of activity.control.commands.slice(-5)){
      const status={pending:'Pending Next Decision Point',applied:'Accepted at Decision Point',rejected:'Rejected'}[command.status]||command.status;
      $('control-history').append(node('p',`${timestamp(command.submitted_at)} · ${command.action} · ${status}\n${command.message||''}${command.target_node?'\nContinue from '+command.target_node+' outputs':''}`));
    }
  }catch(err){if(serial===detailSerial)$('job-reason').textContent='Details temporarily unavailable: '+err.message;}
}
document.querySelectorAll('[data-control]').forEach(button=>button.onclick=async()=>{if(!selected)return;const action=button.dataset.control;if(action==='stop'&&!confirm('Stop the workflow after the current job finishes? Existing results will remain; the running GPU job will not be killed.'))return;if(action==='resume'&&!['running','paused','queued'].includes(selected.status)&&!confirm('Resuming submits a new Slurm controller and incurs compute and API costs. Continue?'))return;button.disabled=true;try{await api(`/api/tasks/${selected.id}/control`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,message:$('operator-message').value,target_node:['guide','resume'].includes(action)?$('operator-target').value:null})});toast('Command saved; waiting for the backend to apply it at a safe decision point.');refresh();}catch(err){toast(err.message);}finally{button.disabled=false;}});
const renderWithServiceStatus=render;
render=function(task){
  renderWithServiceStatus(task);
  const service=task.service_supervision;
  const labels={waiting_for_cryosparc:'Waiting for CryoSPARC Recovery',waiting_for_model_api_network:'Waiting for Model API / Network Recovery',service_stabilizing:'Checking Service Recovery'};
  if(service&&labels[service.status]&&task.status!=='controller_failed'){
    $('task-status').textContent=labels[service.status];
    $('warnings').textContent=`The controller is retained. Health checks run every ${service.poll_seconds||120} seconds; execution resumes after two consecutive healthy checks. Existing CryoSPARC jobs are not cancelled.`;
    $('warnings').hidden=false;
  }
};
const renderWithPilotApproval=render;
render=function(task){
  renderWithPilotApproval(task);
  let panel=$('pilot-diversity-approval');
  if(!panel){panel=node('div');panel.id='pilot-diversity-approval';panel.className='notice';$('warnings').after(panel);}
  panel.replaceChildren();
  const override=task.pilot_diversity_override;
  panel.hidden=!override?.available||!['needs_attention','controller_failed'].includes(task.status);
  if(panel.hidden)return;
  const route=override.best_route;
  panel.append(node('p',`Particle quality passed (${route.qualified_particles.toLocaleString()} good particles), but only ${route.supported_group_count} supported shape groups were found. Best pilot: ${route.best_refine}, ${route.resolution_A.toFixed(2)} A. Human approval does not validate structural correctness.`));
  const button=node('button','Approve Diversity Exception & Continue to Full Data');
  button.className='button secondary';
  button.onclick=async()=>{
    if(!confirm(`Continue ${task.project}/${task.workspace} from ${route.class_job} despite insufficient shape diversity? This uses the remaining original Job budget and incurs compute/API costs.`))return;
    button.disabled=true;
    try{
      await api(`/api/tasks/${task.id}/control`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'approve_pilot_diversity',message:$('operator-message').value||'Human approved the best existing pilot route despite insufficient shape diversity.'})});
      toast('Human exception recorded; controller resubmitted.');refresh();
    }catch(err){toast(err.message);button.disabled=false;}
  };
  panel.append(button);
};
if(token)$('unlock').textContent='Console Unlocked';showcase();route();setInterval(()=>{if(!document.hidden&&location.hash.startsWith('#workflow'))refresh();},10000);
