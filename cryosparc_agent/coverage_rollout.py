"""Opt-in pilot-to-full rollout with source-scoped recovery of weak views.

This experiment reuses explicitly declared own-pilot results, not a blind
post-CTF benchmark. Relative route promotion is not scientific validation.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from pilot_reclassification_experiment import PilotExperiment
from pilot_objective_gate import GatePolicy, evaluate_gate, load_class_images


POLICY = (
    'Build a coherent core reconstruction, and explore weaker excluded classes separately. '
    'CryoSift lower is better: <3 core quality evidence; 3..4.5 is an exploratory recovery pool, '
    'not automatically junk and not automatically usable. Do not discard a coherent rare '
    'view solely for its score. Prior failure can reflect insufficient samples. '
    'Use only current source class IDs; compare whole images, particle support, complete '
    'boundaries, centering, diversity and reference compatibility. A 2D shape group is '
    'not a measured 3D direction. Own lowpassed reference projections are supporting '
    'evidence, not ground truth: do not force novel views to copy an uncertain reference. '
    'Keep core and recovery populations separate until a reconstruction comparison. '
    'Always execute the best nonempty bounded diagnostic route rather than request '
    'human input solely because absolute scores are poor. This does not validate '
    'particles or promise target resolution. Never loop unchanged selections/refinements.'
)


def heldout_indices(all_uids, pilot_uids, count=200):
    all_uids,pilot_uids=np.asarray(all_uids),np.asarray(pilot_uids)
    if len(np.unique(all_uids))!=len(all_uids) or not np.isin(pilot_uids,all_uids).all():
        raise ValueError('Micrograph identity is ambiguous or pilot is outside source')
    indices=np.flatnonzero(~np.isin(all_uids,pilot_uids))[:count]
    if len(indices)!=count:
        raise ValueError('Not enough independent micrographs for confirmation/ramp')
    return indices


def class_pool(context, role, excluded=()):
    counts={r['class_id']:r.get('particle_count',0) for r in context['class_statistics']}
    sift=context.get('cryosift') or {}
    source=context['source']
    if (not sift.get('success') or sift.get('project_uid')!=source['project_uid'] or
            sift.get('job_uid')!=source['job_uid']):
        raise ValueError('Source-scoped CryoSift required; missing scores are not poor scores')
    scores={r['class_id']:r['score'] for r in sift['class_scores']
            if type(r.get('score')) in (int,float) and math.isfinite(r['score'])}
    valid={i for i,n in counts.items() if n>0 and i in scores and i not in excluded}
    if role=='recovery':
        return sorted(i for i in valid if 3.<=scores[i]<=4.5)
    if role=='recovered':
        # Improved rare views can score BELOW 3 after reclassification.
        return sorted(i for i in valid if scores[i]<=4.5)
    if role!='core':
        raise ValueError('Unknown class role')
    strict=[i for i in valid if scores[i]<3.]
    if sum(counts[i] for i in strict)>=1000:
        return sorted(strict,key=lambda i:(scores[i],-counts[i]))
    # A diagnostic core can be relatively best when absolute quality is poor.
    return sorted(valid,key=lambda i:(scores[i],-counts[i]))


def validate_selection(response,pool):
    ids=response.get('selected_class_ids')
    if (not isinstance(ids,list) or any(type(i) is not int or i not in pool for i in ids) or
            len(ids)!=len(set(ids)) or not isinstance(response.get('reason'),str) or not response['reason'].strip()):
        raise ValueError('Unique source-scoped integer IDs and a nonempty reason required')
    return response


def union_from_source(source, datasets):
    ids=np.unique(np.concatenate([np.asarray(d['uid']) for d in datasets]))
    if len(np.unique(source['uid']))!=len(source) or not np.isin(ids,source['uid']).all():
        raise ValueError('Union contains foreign or ambiguous particles')
    return source.take(np.flatnonzero(np.isin(source['uid'],ids)))


def external_slots(output_spec,wanted):
    """Slot names are not necessarily dtypes (e.g. blob_selected -> blob)."""
    definitions={}
    for item in [*output_spec.get('slots',[]),*output_spec.get('results',[])]:
        if isinstance(item,dict) and item.get('name') and item.get('dtype'):
            definitions[item['name']]=item['dtype']
    missing=set(wanted)-definitions.keys()
    if missing:
        raise ValueError('Missing authoritative External slot dtype: '+','.join(sorted(missing)))
    return [dict(name=name,dtype=definitions[name],required=True) for name in sorted(wanted)]


class CoverageRollout(PilotExperiment):
    allowed_job_types={'blob_picker_gpu','template_picker_gpu','inspect_picks_v2','extract_micrographs_multi',
        'class_2D_new','select_2D','volume_tools','reference_select_2D','homo_abinit',
        'homo_refine_new','nonuniform_refine_new'}

    def __init__(self,root):
        super().__init__(root)
        if self.manifest.get('protocol')!='coverage-rollout-v1':
            raise ValueError('Explicit isolated coverage-rollout manifest required')
        self.allowed_samples=(50,200,self.manifest['full_micrographs'])
        self.policy=GatePolicy()
        self.cryosift_threshold=3.

    def submit(self,slug,kind,sources,params,wait=True):
        if kind=='template_picker_gpu' and params.get('num_process') not in self.allowed_samples:
            raise ValueError('Unexpected template picking population')
        if not self.state['steps'].get(slug,{}).get('job_uid'):
            created={s.get('job_uid') for s in self.state['steps'].values() if s.get('job_uid')}
            if len(created)>=self.manifest.get('max_seed_jobs',40):
                raise RuntimeError('Seed/rollout phase budget exhausted; preserve state for review')
        return super().submit(slug,kind,sources,params,wait)

    def model(self,slug,messages,model,validate):
        return super().model(slug,[*messages,dict(role='user',content=POLICY)],model,validate)

    def external(self,slug,datatype,output,connections,dataset):
        """One durable External job; save only source-derived, unchanged data."""
        if not len(dataset) or len(np.unique(dataset['uid']))!=len(dataset):
            raise ValueError('External output must be nonempty with unique UIDs')
        record=self.state['steps'].setdefault(slug,{})
        digest=hashlib.sha256(np.asarray(dataset['uid']).tobytes()).hexdigest()
        if record.get('uid_sha256',digest)!=digest:
            raise ValueError('External population changed on resume')
        record.update(uid_sha256=digest,particle_or_exposure_count=len(dataset))
        title='Coverage rollout '+slug
        if not record.get('job_uid'):
            if record.get('creation_intent'):
                matches=self.service('reconcile '+slug,lambda:[j for j in self.workspace.find_jobs() if j.title==title])
                if len(matches)!=1:
                    raise RuntimeError('Ambiguous external creation; no blind retry')
                record['job_uid']=matches[0].uid
            else:
                if len({s.get('job_uid') for s in self.state['steps'].values() if s.get('job_uid')})>=self.manifest.get('max_seed_jobs',40):
                    raise RuntimeError('Rollout External phase budget exhausted')
                record['creation_intent']=True
                self.persist()
                job=self.cs.create_external_job(self.project.uid,self.workspace.uid,title=title,
                    desc='Explicit source-derived rollout subset/bank; no historical orientation-labelled particle selection.')
                record['job_uid']=job.uid
            self.persist()
        job=self.service('external '+slug,lambda:self.project.find_external_job(record['job_uid']))
        slots=sorted({field.split('/')[0] for field in dataset.fields() if '/' in field})
        if job.status=='building':
            output_types={}
            for name,(uid,out) in connections.items():
                source_job=self.project.find_job(uid)
                source=source_job.load_output(out)
                source_fields={f.split('/')[0] for f in source.fields() if '/' in f}
                required=set(slots)&source_fields
                source_slots=external_slots(source_job.model.model_dump()['spec']['outputs'][out],required)
                for slot in source_slots:
                    if slot['name'] in output_types and output_types[slot['name']]!=slot['dtype']:
                        raise ValueError('Incompatible External source slot dtypes')
                    output_types[slot['name']]=slot['dtype']
                spec=job.model.model_dump()['spec']
                if name not in spec['inputs']:
                    job.connect(name,uid,out,slots=source_slots)
                elif [(v['job_uid'],v['output']) for v in spec['inputs'][name].get('connections',[])]!=[(uid,out)]:
                    raise ValueError('External source identity changed')
            if output not in job.model.model_dump()['spec']['outputs']:
                if set(slots)-output_types.keys():
                    raise ValueError('External output fields lack a declared source')
                job.add_output(datatype,output,slots=[dict(name=name,dtype=output_types[name],required=True) for name in slots])
            job.start()
        if job.status in {'waiting','running'}:
            job.save_output(output,dataset)
            job.stop()
        if job.status!='completed':
            raise RuntimeError('Incomplete External output; reconcile without duplication')
        saved=self.service('verify '+slug,lambda:job.load_output(output))
        if not np.array_equal(saved['uid'],dataset['uid']):
            raise ValueError('External saved UIDs differ')
        record['status']='completed'
        self.persist()
        return job

    def readout(self,slug,job):
        c=self.classes(slug,job)
        images=self.service('class image '+job.uid,lambda:load_class_images(self.project,job.uid))
        g=evaluate_gate(c,images,self.policy)
        self.write(self.root/'evidence'/slug/'objective_gate.json',g)
        return c,g

    def reference(self,slug,classes,volume):
        from refinement_optimization import reference_sampling
        size=self.service('reference sampling',lambda:reference_sampling(self.project.uid,classes,volume))
        low=self.submit(slug+'_lowpass','volume_tools',dict(volume=(volume,'volume')),dict(vol_lpf_A=20.,vol_resample_N=size))
        return self.submit(slug+'_match','reference_select_2D',dict(particles=(classes,'particles'),
            templates=(classes,'class_averages'),volume=(low.uid,'volume')),
            dict(select_mode='thresholds only',pearson_thres=.3,hog_thres=.3,res_filter=30.,max_res=20.,lp_filter=True,lp_filter_A=20.))

    def reference_images(self,job):
        import base64
        images=[]
        events=self.service('reference plots',lambda:self.cs.api.jobs.get_event_logs(self.project.uid,job.uid))
        for event in reversed(events):
            if event.type!='image':
                continue
            asset=next((a for a in event.imgfiles if a.filetype=='png'),None)
            if asset is None:
                continue
            data=self.service('reference image',lambda:self.cs.api.assets.download(asset.fileid).dumps())
            if len(data)<3*1024**2:
                images.append(dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(data).decode())))
            if len(images)>=2:
                break
        return images

    def review(self,slug,context,role,excluded=(),reference=None):
        from upstream_evidence import metadata
        pool=class_pool(context,role,excluded)
        if not pool:
            return []
        prompt=dict(role=role,allowed_class_ids=pool,context=metadata(context),
            instruction='Choose source-scoped classes for '+role+'. '
                'Only a core-role request requires a nonempty executable core. '
                'For recovery/recovered, supplement the EXISTING CORE from the separately labelled source. '
                'Do NOT invent a new provisional core from this candidate job and then omit those classes. '
                'If you regard a candidate as a good provisional core view, it is still eligible here '
                'when it complements the actual existing core. Prefer intact additional shapes, '
                'including compact views absent from that core, not high-quality duplicates. '
                'A recovery class at score 4.0 can be informative; it is not automatically trustworthy. '
                'Reference lowpass/correlations are permissive support, not an authoritative orientation label. '
                'Return {"selected_class_ids":[integers],"reason":"evidence and uncertainty"}.')
        content=[dict(type='text',text=json.dumps(prompt)),
            dict(type='image_url',image_url=dict(url=context['contact_sheet']['data_url']))]
        if role!='core' and self.state.get('current_core'):
            core=self.state['current_core']
            core_context=json.loads(Path(core['context_file']).read_text())
            content += [dict(type='text',text='EXISTING CORE REFERENCE. Only these source-scoped IDs '
                'are retained; other tiles are not core. Do not copy these IDs to the candidate source. '+json.dumps(core)),
                dict(type='image_url',image_url=dict(url=core_context['contact_sheet']['data_url']))]
        if reference:
            d=reference.load_output('templates_selected')
            prompt['reference_suggested_source_ids']=[int(i) for i in d['blob/idx']]
            content[0]['text']=json.dumps(prompt)
            content+=self.reference_images(reference)
        try:
            response=self.model(slug,[dict(role='system',content=POLICY),dict(role='user',content=content)],
                'openai/gpt-6.1-sol',lambda r:validate_selection(r,set(pool)))
            ids=response['selected_class_ids']
        except RuntimeError as exc:
            if 'Model contract failed three times' not in str(exc):
                raise
            self.write(self.root/'decisions'/slug/'bounded_contract_fallback.json',dict(error=str(exc),role=role))
            ids=[]
        if role=='core':
            counts={r['class_id']:r.get('particle_count',0) for r in context['class_statistics']}
            # Lack of an absolute score pass is not a quality-only terminal state.
            for i in pool:
                if ids and sum(counts[v] for v in ids)>=min(1000,sum(counts[v] for v in pool)):
                    break
                if i not in ids:
                    ids.append(i)
            path=self.root/'evidence'/slug/'core_context.json'
            self.write(path,context)
            self.state['current_core']=dict(source=context['source'],class_ids=ids,context_file=str(path))
            self.persist()
        return ids

    def select(self,slug,source,ids):
        if not ids:
            raise ValueError('Cannot create an empty Select 2D')
        return self.submit(slug,'select_2D',dict(particles=(source,'particles'),templates=(source,'class_averages')),
            dict(selected_templates=','.join(map(str,ids))))

    def bank(self):
        if self.state.get('bank_job') and self.state.get('bank_revision')==2:
            return self.project.find_job(self.state['bank_job'])
        if self.state.get('bank_job'):
            if any(k.startswith('confirm_template_') for k in self.state['steps']):
                raise RuntimeError('Old bank already used for Template confirmation; do not silently change the experiment')
            self.state.setdefault('superseded_banks',[]).append(dict(job=self.state.pop('bank_job'),
                provenance=self.state.pop('bank_provenance',[]),reason='Recovery review lacked explicit actual core context'))
            self.persist()
        datasets=[]
        sources={}
        provenance=[]
        for uid,role in [('J2286','core'),('J2295','recovered'),('J2296','recovered')]:
            job=self.project.find_job(uid)
            context,_=self.readout('seed_'+uid,job)
            match=self.reference('seed_'+uid,uid,'J2290') if role!='core' else None
            revision='_v2' if role!='core' else ''
            ids=self.review('seed_review'+revision+'_'+uid,context,role,reference=match)
            if not ids:
                continue
            selected=self.select('seed_select'+revision+'_'+uid,uid,ids)
            data=selected.load_output('templates_selected').filter_fields(lambda field:field=='uid' or field.startswith('blob/'))
            datasets.append(data)
            sources['source_'+uid]=(selected.uid,'templates_selected')
            provenance.append(dict(source=uid,role=role,ids=ids,selection=selected.uid))
        if not datasets:
            raise ValueError('No executable template bank')
        data=datasets[0].append(*datasets[1:])
        bank=self.external('template_bank_v2','template','templates',sources,data)
        self.state.update(bank_job=bank.uid,bank_provenance=provenance,bank_revision=2)
        self.persist()
        return bank

    def template_parameters(self):
        if self.state.get('template_parameters'):
            return self.state['template_parameters']
        content=[dict(type='text',text='Choose Template Picker whole-particle diameter in Angstroms from '
            'the pilot class images. The physical field of view is 376.64 A per tile '
            '(440 native pixels at 0.856 A/px). Class averages are downsampled to 128 pixels '
            'at 2.9425 A/px, and rendered in 192-pixel tiles: do NOT treat displayed pixels '
            'as native pixels. '
            'The prior Blob minimum 120 A is NOT a measured diameter. Return JSON '
            '{"diameter":number,"reason":"image-based estimate with uncertainty"}. '
            'No numeric min_distance override; use the native dimensionless default.')]
        for uid in ['J2286','J2295','J2296']:
            c=json.loads((self.root/'evidence'/('seed_'+uid)/'context.json').read_text())
            content.append(dict(type='image_url',image_url=dict(url=c['contact_sheet']['data_url'])))
        def validate(r):
            d=r.get('diameter')
            if type(d) not in (int,float) or not math.isfinite(d) or not 30<=d<=350:
                raise ValueError('Finite whole-particle diameter 30..350 A required')
            return r
        response=self.model('template_diameter',[dict(role='system',content=POLICY),dict(role='user',content=content)],
            'openai/gpt-6.1-sol',validate)
        self.state['template_parameters']=dict(diameter=response['diameter'])
        self.persist()
        return self.state['template_parameters']

    def route(self,tag,micrographs,count,kind,bank,thresholds=None):
        from vision_inputs import build_pick_inspection_visual_context
        if kind=='template':
            params=dict(self.template_parameters(),num_process=count)
            inputs=dict(micrographs=micrographs,templates=(bank.uid,'templates'))
            typ='template_picker_gpu'
        else:
            params=dict(diameter=120,diameter_max=220,num_process=count)
            inputs=dict(micrographs=micrographs)
            typ='blob_picker_gpu'
        picker=self.submit(tag+'_picker',typ,inputs,params)
        if picker.outputs['particles'].num_items==0:
            return dict(kind=kind,count=count,executable=False,reason='No picked particles')
        if thresholds is None:
            visual=self.service('Inspect evidence',lambda:build_pick_inspection_visual_context(self.project.uid,picker.uid,max_micrographs=3))
            thresholds=self.stage(tag+'_inspect_parameters','inspect_picks_threshold',visual)
        raw=picker.load_output('particles',slots=['pick_stats'])
        ncc=np.asarray(raw['pick_stats/ncc_score']); power=np.asarray(raw['pick_stats/power'])
        expected=float(np.mean((ncc>=thresholds['ncc_score_thresh'])&(power>=thresholds['lpower_thresh_min'])&
                               (power<=thresholds['lpower_thresh_max'])))
        if tag!='confirm_blob' and expected<.1:
            visual=self.service('drifted Inspect evidence',lambda:build_pick_inspection_visual_context(self.project.uid,picker.uid,max_micrographs=3))
            thresholds=self.stage(tag+'_inspect_drift_review','inspect_picks_threshold',dict(visual,
                frozen_thresholds=thresholds,expected_retention=expected,
                warning='One bounded review: earlier raw thresholds reject over 90%; do not assume weak views are junk.'))
        inspection=self.submit(tag+'_inspect','inspect_picks_v2',dict(micrographs=(picker.uid,'micrographs'),particles=(picker.uid,'particles')),
            {**{k:thresholds[k] for k in ('ncc_score_thresh','lpower_thresh_min','lpower_thresh_max')},
             'calibrate_ncc':False,'calibrate_pow':False,'do_auto_cluster':False})
        if not inspection.outputs['particles'].num_items:
            return dict(kind=kind,count=count,executable=False,reason='Empty Inspect; alternate route required')
        extracted=self.submit(tag+'_extract','extract_micrographs_multi',dict(micrographs=(inspection.uid,'micrographs'),
            particles=(inspection.uid,'particles')),dict(box_size_pix=440,num_extract=count,compute_num_gpus=1))
        number=extracted.outputs['particles'].num_items
        if number<100:
            return dict(kind=kind,count=count,executable=False,reason='Fewer than 100 extracted particles')
        k=min(100,max(10,number//500))
        classes=self.submit(tag+'_class','class_2D_new',dict(particles=(extracted.uid,'particles')),
            dict(class2D_K=k,class2D_num_full_iter=4,random_seed=26026,compute_num_gpus=1))
        context,gate=self.readout(tag+'_classes',classes)
        return dict(kind=kind,count=count,executable=True,picker=picker.uid,inspect=inspection.uid,extract=extracted.uid,
            classes=classes.uid,particles=number,thresholds={k:thresholds[k] for k in ('ncc_score_thresh','lpower_thresh_min','lpower_thresh_max')},
            score=context['score_summary'],gate=gate,evidence_slug=tag+'_classes')

    def choose_route(self,routes):
        viable={k:v for k,v in routes.items() if v.get('executable')}
        if not viable:
            raise RuntimeError('Neither confirmation route has executable particles; no fake promotion')
        content=[dict(type='text',text=POLICY+' Compare matched independent 50-image routes. '
            'Consider supported quality groups, coherent additional shapes, particle yield and contamination, '
            'not just mean CryoSift. Return {"route":"blob|template","reason":"evidence",'
            '"validated":false}. validated is informational; it cannot override quantitative flags.\n'+json.dumps(viable))]
        for name,v in viable.items():
            c=json.loads((self.root/'evidence'/('confirm_'+name+'_classes')/'context.json').read_text())
            content+=[dict(type='text',text='SOURCE '+name+' '+v['classes']),
                dict(type='image_url',image_url=dict(url=c['contact_sheet']['data_url']))]
        def validate(r):
            if r.get('route') not in viable:
                raise ValueError('Choose an executable advertised route')
            return r
        r=self.model('confirmation_route_choice',[dict(role='system',content=POLICY),dict(role='user',content=content)],
            'openai/gpt-6.1-sol',validate)
        return r['route']

    def reconstruct(self,full):
        job=self.project.find_job(full['classes'])
        context=json.loads((self.root/'evidence'/full['evidence_slug']/'context.json').read_text())
        core_ids=self.review('full_core_review',context,'core')
        core=self.select('full_core_select',job.uid,core_ids)
        initial=self.submit('full_core_abinit','homo_abinit',dict(particles=(core.uid,'particles_selected')),
            dict(abinit_K=1))
        homo=self.submit('full_core_homo','homo_refine_new',dict(particles=(core.uid,'particles_selected'),
            volume=(initial.uid,'volume_class_0')),dict(refine_symmetry='C1'))
        if self.manifest.get('production_pipeline') and self.resolution(homo) <= self.manifest['target_resolution_A']:
            return homo
        nu=self.submit('full_core_nu','nonuniform_refine_new',dict(particles=(homo.uid,'particles'),volume=(homo.uid,'volume')),
            dict(refine_symmetry='C1'))
        checkpoints=[homo,nu]
        if min(self.resolution(j) for j in checkpoints)<=self.manifest['target_resolution_A']:
            return min(checkpoints,key=self.resolution)
        pool=class_pool(context,'recovery',core_ids)
        if pool:
            tail=self.select('full_tail_select',job.uid,pool)
            n=tail.outputs['particles_selected'].num_items
            if n>=100:
                tail_class=self.submit('full_tail_reclass','class_2D_new',dict(particles=(tail.uid,'particles_selected')),
                    dict(class2D_K=min(50,max(5,n//500)),class2D_num_full_iter=4,random_seed=26027,compute_num_gpus=1))
                tail_context,_=self.readout('full_tail_classes',tail_class)
                matched=self.reference('full_tail_reference',tail_class.uid,nu.uid)
                supplemental_ids=self.review('full_tail_review',tail_context,'recovered',reference=matched)
                if supplemental_ids:
                    selected=self.select('full_tail_compatible_select',tail_class.uid,supplemental_ids)
                    original=job.load_output('particles')
                    core_data=core.load_output('particles_selected')
                    extra=selected.load_output('particles_selected')
                    union=union_from_source(original,[core_data,extra])
                    combined=self.external('full_core_tail_union','particle','particles',dict(source=(job.uid,'particles')),union)
                    expanded=self.submit('full_union_nu','nonuniform_refine_new',dict(particles=(combined.uid,'particles'),volume=(nu.uid,'volume')),
                        dict(refine_symmetry='C1',refine_res_init=30.,refine_gs_resplit=True))
                    checkpoints.append(expanded)
                    self.state['union_comparison']=dict(core=nu.uid,expanded=expanded.uid,added=len(union)-len(core_data),
                        note='Native fresh half-set split and 30 A initial lowpass; not an isolated matched-half-set statistical treatment test.')
                    self.persist()
        best=min(checkpoints,key=self.resolution)
        if self.resolution(best)<=4. and self.resolution(best)>self.manifest['target_resolution_A']:
            from refinement_optimization import OPTICS
            optics=self.submit('full_optics_nu','nonuniform_refine_new',dict(particles=(best.uid,'particles'),volume=(best.uid,'volume')),
                dict(refine_symmetry='C1',refine_gs_resplit=False,**OPTICS))
            checkpoints.append(optics)
        return min(checkpoints,key=self.resolution)

    @staticmethod
    def resolution(job):
        fsc=job.model.model_dump()['spec']['outputs']['volume'].get('latest_summary_stats',{}).get('fsc_info',{})
        value=fsc.get('radwn_final_A')
        return float(value) if isinstance(value,(int,float)) and math.isfinite(value) and value>0 else math.inf

    def run(self):
        if self.state.get('rollout_complete'):
            return self.state['best_refine']
        bank=self.bank()
        ctf=self.project.find_job('J1442').load_output('exposures')
        pilot=self.project.find_job('J2281').load_output('micrographs')
        indices=heldout_indices(ctf['uid'],pilot['uid'])
        holdout=ctf.take(indices[:50])
        ramp=ctf.take(indices)
        confirmation=self.external('heldout50','exposure','exposures',dict(source=('J1442','exposures')),holdout)
        ramp_input=self.external('ramp200','exposure','exposures',dict(source=('J1442','exposures')),ramp)
        self.write(self.root/'independence.json',dict(pilot_uids=list(map(int,pilot['uid'])),
            confirmation_uids=list(map(int,holdout['uid'])),ramp_uids=list(map(int,ramp['uid'])),
            confirmation_disjoint_from_pilot=True,confirmation_included_in_ramp=True))
        routes=self.state.setdefault('confirmation',{})
        for kind in ('blob','template'):
            if kind not in routes:
                routes[kind]=self.route('confirm_'+kind,(confirmation.uid,'exposures'),50,kind,bank,
                    dict(ncc_score_thresh=.2,lpower_thresh_min=650,lpower_thresh_max=1400) if kind=='blob' else None)
                self.persist()
        if not self.state.get('chosen_route'):
            self.state['chosen_route']=self.choose_route(routes)
            self.persist()
        chosen=self.state['chosen_route']
        if 'ramp' not in self.state:
            self.state['ramp']=self.route('ramp',(ramp_input.uid,'exposures'),200,chosen,bank,routes[chosen]['thresholds'])
            self.persist()
        ramp_result=self.state['ramp']
        if not ramp_result.get('executable'):
            alternative='blob' if chosen=='template' else 'template'
            if not routes[alternative].get('executable'):
                raise RuntimeError('Ramp has no executable continuation, including the alternate confirmation route')
            if 'ramp_alternate' not in self.state:
                self.state['ramp_alternate']=self.route('ramp_alternate',(ramp_input.uid,'exposures'),200,
                    alternative,bank,routes[alternative]['thresholds'])
                self.persist()
            ramp_result=self.state['ramp_alternate']
            if not ramp_result.get('executable'):
                raise RuntimeError('Neither 200-image ramp has executable particles')
            self.state['chosen_route']=alternative
            chosen=alternative
            self.persist()
        self.state['promotion_quality_status']=('objective_pass_on_confirmation_and_ramp'
            if routes[chosen]['gate'].get('passed') and ramp_result['gate'].get('passed')
            else 'exploratory_relative_best_not_validated')
        self.persist()
        if 'full' not in self.state:
            thresholds=ramp_result['thresholds']
            self.state['full']=self.route('full',('J1442','exposures'),self.manifest['full_micrographs'],chosen,bank,thresholds)
            self.persist()
        if not self.state['full'].get('executable'):
            raise RuntimeError('Full route has no executable particles; cannot fabricate reconstruction')
        best=self.reconstruct(self.state['full'])
        self.state.update(rollout_complete=True,status='seeded_for_planner',best_refine=best.uid,best_resolution_A=self.resolution(best))
        self.persist()
        return best.uid
