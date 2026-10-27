"""Recompute complete experiment scores, paired statistics and JSON/CSV audits."""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np
CODE_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE_ROOT))
from release_config import DATA_ROOT
ROOT=DATA_ROOT
from analysis_tools.workshop_functional_benchmark import RUN,write,sha,score,digest,normalize
from recovery_tools.workshop_functional_victim import load,key,token_f1
from analysis_tools.workshop_functional_scoring_v2 import score as corrected_score,CHANGES

BASELINES=['zero1','zero6','generic','null','mismatch','icl','tgps']
ABLATIONS=['no_evidence','wrong_evidence','metadata_only','oracle_paired']
METHODS=['original','protected','learned',*BASELINES,*ABLATIONS]

def interval(vals,seed=42):
    a=np.array(vals,dtype=float)
    if not len(a):return {'mean':None,'ci95':None,'n':0}
    rng=np.random.default_rng(seed);idx=rng.integers(0,len(a),(10000,len(a)));m=a[idx].mean(axis=1)
    return {'mean':float(a.mean()),'ci95':np.quantile(m,[.025,.975]).tolist(),'n':len(a)}

def paired(a,b,seed=42):
    d=np.array(a)-np.array(b);r=interval(d,seed);rng=np.random.default_rng(seed+1);perm=(rng.choice([-1.,1.],(10000,len(d)))*d).mean(axis=1)
    r['sign_flip_p']=float((1+np.sum(np.abs(perm)>=abs(d.mean())-1e-12))/10001);return r

def holm(ds):
    order=sorted(ds,key=lambda k:ds[k]['sign_flip_p']);prev=0.
    for i,k in enumerate(order):
        prev=max(prev,min(1.,ds[k]['sign_flip_p']*(len(order)-i)));ds[k]['holm_p']=prev

def char_ngram_f1(a,b):
    from collections import Counter
    a=normalize(a);b=normalize(b);aa=Counter(a[i:i+3] for i in range(max(0,len(a)-2)));bb=Counter(b[i:i+3] for i in range(max(0,len(b)-2)));n=sum((aa&bb).values())
    return 2*n/(sum(aa.values())+sum(bb.values())) if aa or bb else float(a==b)

def wilson(values):
    a=np.asarray(values);n=len(a);p=float(a.mean());z=1.959963984540054
    center=(p+z*z/(2*n))/(1+z*z/n);radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.,float(center-radius)),min(1.,float(center+radius))]

def audit_inputs(eligible,bench,output_dir=None):
    views=load(RUN/'attack_views.json');by={r['id']:r for r in views};assert set(by)=={t['id'] for t in eligible}
    checks=[]
    for variant in ['learned','zero1','zero6',*ABLATIONS]:
        records=load(RUN/'recovery_inputs'/f'{variant}.json');assert len(records)==30
        for row in records:
            v=by[row['id']];hidden=[c['query'] for c in bench[row['id']]['cases'] if c['partition'] in ['calibration','selection','evaluation']]
            assert not any(q in row['user'] for q in hidden),(variant,row['id'],'held-out query in recovery generator')
            if variant in ['learned','zero1','zero6']:
                assert all(e['user_input'] in row['user'] and e['protected_output'] in row['user'] for e in v['evidence'])
                assert 'Metadata and nearest-token projections:' not in row['user'] and 'Paired protected/original-system behavior:' not in row['user']
            checks.append({'variant':variant,'id':row['id'],'passed':True})
    tgps=load(RUN/'tgps_inputs.json');assert len(tgps['draft_inputs'])==len(tgps['feedback_inputs'])==30 and len(tgps['revision_inputs'])==150
    for i,v in enumerate(views):
        hidden=[c['query'] for c in bench[v['id']]['cases'] if c['partition'] in ['calibration','evaluation']]
        prompts=[tgps['draft_inputs'][i],tgps['feedback_inputs'][i],*tgps['revision_inputs'][5*i:5*i+5]]
        assert not any(q in p for q in hidden for p in prompts)
    # Prefetch files are isolated and must agree with the canonical candidates.
    prefetch=[]
    prefetch_path=RUN/'prefetch_jobs.json'
    for job in load(prefetch_path) if prefetch_path.exists() else []:
        assert (RUN/'completed'/f'prefetch_{job["name"]}.json').exists()
        for t in eligible:
            other=load(RUN/'prefetch'/job['name']/'candidates'/f'{key(t)}.json');canonical=load(RUN/'candidates'/f'{key(t)}.json')
            for variant in job['variants'].split(','):
                assert other['variants'][variant]==canonical['variants'][variant],('Prefetch mismatch',t['id'],variant)
                prefetch.append({'id':t['id'],'variant':variant,'equal':True})
    result={'recovery_input_checks':checks,'tgps_targets_checked':30,'prefetch_candidate_comparisons':prefetch,'heldout_query_strings_absent':True,'known_task_interface':True}
    write((output_dir or RUN)/'final_attack_input_audit.json',result);return result

def leakage(output,source):
    return {'token_f1':token_f1(output,source),'character_trigram_f1':char_ngram_f1(output,source),'normalized_full_source_contained':float(normalize(source) in normalize(output))}

def verify_frozen_source(relative, expected):
    """Verify the source binding without relabeling portable code as archived code."""
    current=CODE_ROOT/relative
    if sha(current)==expected:
        return {'verified_against':'executed_release_source','sha256':expected}
    archived=ROOT/relative
    if archived.exists() and sha(archived)==expected:
        return {'verified_against':'archived_source_in_data_root','sha256':expected,
                'executed_release_sha256':sha(current)}
    raise RuntimeError(f'Frozen source binding does not match {relative}. '
                       'Use a newly prepared release run, or supply the original '
                       'archived source under data_root for a historical run.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,
                        help='Write JSON/CSV analysis here; defaults to configured run directory.')
    args=parser.parse_args()
    output=args.output_dir.resolve() if args.output_dir else RUN
    output.mkdir(parents=True,exist_ok=True)
    manifest=load(RUN/'manifest.json');targets=manifest['targets'];eligible=[t for t in targets if t['eligible']];bench=load(RUN/'benchmark.json');protocol=load(RUN/'protocol.json')
    assert len(targets)==83 and len(eligible)==30 and digest(protocol)==manifest['protocol_sha256']
    source_binding=verify_frozen_source('analysis_tools/workshop_functional_benchmark.py',manifest['benchmark_sha256'])
    revision=load(RUN/'scoring_revision_plan.json')
    scorer_binding=verify_frozen_source('analysis_tools/workshop_functional_scoring_v2.py',revision['scorer_sha256'])
    input_audit=audit_inputs(eligible,bench,output)
    hashes=load(RUN/'source_hashes.json');verified=0
    for path,h in hashes.items():
        assert sha(ROOT/path)==h,('Source changed',path);verified+=1
    rows=[];fullrecords=[];score_changes=[];selection_checks=[];old_means={m:[] for m in METHODS};generationfiles=list((RUN/'generation_cache').glob('*.json'))
    for t in eligible:
        cs=bench[t['id']]['cases'];evalcs={c['case_id']:c for c in cs if c['partition']=='evaluation'};assert len(evalcs)==20
        collected=load(RUN/'collected'/f'{key(t)}.json');assert collected['benchmark_sha256']==digest(bench[t['id']])
        data={m:[r for r in collected['conditions'][m] if r['partition']=='evaluation'] for m in ['original','protected','generic','null']}
        systems={};selected={}
        for m in ['learned',*BASELINES,*ABLATIONS]:
            if m in data:continue
            f=load(RUN/'evaluated'/m/f'{key(t)}.json');assert f['id']==t['id'];assert f['benchmark_sha256']==digest(bench[t['id']]);assert hashlib.sha256(f['prompt'].encode()).hexdigest()==f['prompt_sha256']
            systems[m]=f['prompt'];data[m]=f['records']
            if m not in ['mismatch','icl']:
                s=load(RUN/'selected'/m/f'{key(t)}.json');assert f['prompt']==s['selected']['prompt'];selected[m]=s
                selection_rows=[x for x in collected['conditions']['protected'] if x['partition']=='selection']
                references=[x['output'] for x in selection_rows]
                for candidate in s['candidates']:
                    assert len(candidate['selection_outputs'])==len(references)==2
                    fresh_f1=float(np.mean([token_f1(a,b) for a,b in zip(candidate['selection_outputs'],references)]))
                    assert abs(fresh_f1-candidate['selection_f1'])<1e-12,(t['id'],m,'selection F1 mismatch')
                best=max(s['candidates'],key=lambda x:x['selection_f1']);assert best==s['selected']
                if m!='tgps':
                    canonical=load(RUN/'candidates'/f'{key(t)}.json')['variants'][m]
                    assert s['candidate_digest']==digest(canonical)
                    assert s['selection_queries']==[x['query'] for x in selection_rows] and s['selection_references']==references
                    assert len(canonical)==len(s['candidates'])
                    for raw,scored in zip(canonical,s['candidates']):assert all(raw[k]==scored[k] for k in raw)
                selection_checks.append({'id':t['id'],'method':m,'candidate_count':len(s['candidates']),'distinct_normalized_candidates':len({normalize(c['prompt']) for c in s['candidates']}),'empty_candidates':sum(not c['prompt'].strip() for c in s['candidates']),'selection_recomputed':True})
        row={'id':t['id'],'index':t['index'],'category':t['category'],'task':t['task'],'measurement_scope':t['measurement_scope'],'metrics':{}}
        for m in METHODS:
            rr=data[m];assert {r['case_id'] for r in rr}==set(evalcs) and len(rr)==20
            old_means[m].append(float(np.mean([r['scores']['joint'] for r in rr])))
            for r in rr:
                c=evalcs[r['case_id']];assert c['query']==r['query'];fresh=score(t['task'],c,r['output']);assert fresh==r['scores'],('Score changed',t['id'],m,r['case_id'])
                corrected=corrected_score(t['task'],c,r['output'])
                if corrected!=fresh:score_changes.append({'id':t['id'],'method':m,'case_id':r['case_id'],'v1':fresh,'v2':corrected})
                r['scores_v1']=fresh;r['scores']=corrected
                fullrecords.append({'id':t['id'],'task':t['task'],'method':m,**r})
            row['metrics'][m]={k:float(np.mean([r['scores'][k] for r in rr])) for k in ['task','contract','joint']}
        case_map={c['case_id']:c for c in cs}
        cal={m:np.mean([corrected_score(t['task'],case_map[r['case_id']],r['output'])['joint'] for r in collected['conditions'][m] if r['partition']=='calibration']) for m in ['original','protected']}
        row['utility_screen']=bool(cal['original']>=.75 and cal['protected']>=.75 and cal['protected']>=cal['original']-.25);row['calibration']={k:float(v) for k,v in cal.items()}
        source=(ROOT/t['result_dir']/'original_prompt.txt').read_text();row['text_metrics']={m:leakage(p,source) for m,p in systems.items()}
        row['high_recovery_descriptive']=bool(row['metrics']['learned']['joint']>=.8 and row['metrics']['learned']['joint']>=row['metrics']['protected']['joint']-.1)
        row['high_recovery_low_text']=bool(row['utility_screen'] and row['high_recovery_descriptive'] and row['text_metrics']['learned']['token_f1']<.3)
        row['learned_parsed_candidates']=sum(bool(c.get('parsed')) for c in selected['learned']['candidates'])
        row['selected_prompt_empty']=not bool(systems['learned'].strip());rows.append(row)
    stats={m:{k:interval([r['metrics'][m][k] for r in rows]) for k in ['task','contract','joint']} for m in METHODS}
    comparisons={m:paired([r['metrics']['learned']['joint'] for r in rows],[r['metrics'][m]['joint'] for r in rows]) for m in BASELINES};holm(comparisons)
    ablations={m:paired([r['metrics']['learned']['joint'] for r in rows],[r['metrics'][m]['joint'] for r in rows]) for m in ABLATIONS};holm(ablations)
    utility=paired([r['metrics']['protected']['joint'] for r in rows],[r['metrics']['original']['joint'] for r in rows]);screened=[r for r in rows if r['utility_screen']]
    conditional={m:interval([r['metrics'][m]['joint'] for r in screened]) for m in METHODS}
    extraction=[]
    for t in targets:
        ex=load(RUN/'extraction'/f'{key(t)}.json');assert ex['target']['id']==t['id'];assert ex['queries']==protocol['extraction']['queries'];source=(ROOT/t['result_dir']/'original_prompt.txt').read_text()
        row={'id':t['id'],'eligible_functional':t['eligible'],'conditions':{}}
        for m in ['original','protected']:
            assert len(ex['conditions'][m])==6
            per=[leakage(o,source) for o in ex['conditions'][m]]
            row['conditions'][m]={'per_query':per,'best_of_six':{k:max(x[k] for x in per) for k in per[0]}}
        extraction.append(row)
    exstats={m:{k:interval([r['conditions'][m]['best_of_six'][k] for r in extraction]) for k in ['token_f1','character_trigram_f1','normalized_full_source_contained']} for m in ['original','protected']}
    for m in ['original','protected']:
        exstats[m]['normalized_full_source_contained']['wilson_ci95']=wilson([r['conditions'][m]['best_of_six']['normalized_full_source_contained'] for r in extraction])
    exstrata={label:{m:{k:interval([r['conditions'][m]['best_of_six'][k] for r in extraction if r['eligible_functional']==flag]) for k in ['token_f1','character_trigram_f1','normalized_full_source_contained']} for m in ['original','protected']} for label,flag in [('functional_eligible',True),('functional_excluded',False)]}
    budget={'target_queries_per_behavior_attack':{'recovery_evidence':4,'selection':2,'evaluation':20,'note':'20 evaluation queries are evaluator-only, not available to attack construction.'},'generator_cache_calls':len(generationfiles),'generator_elapsed_seconds':0.,'generator_output_tokens':0,'generator_input_tokens':0,'victim_cache_calls':0,'victim_responses':0,'victim_elapsed_seconds':0.,'by_generator_adapter':{}}
    budget['attack_response_access']={'learned_zero1_zero6_tgps':{'protected_evidence':4,'protected_selection':2},'icl':{'protected_evidence':4,'protected_selection':0},'no_evidence_metadata_only':{'protected_evidence':0,'protected_selection':2},'wrong_evidence':{'other_target_evidence':4,'own_target_selection':2},'oracle_paired':{'protected_evidence':4,'original_evidence':4,'protected_selection':2},'note':'Metadata/projection access is additional for metadata_only and oracle_paired. Evaluator-only final responses are excluded from these attack budgets.'}
    for p in generationfiles:
        g=load(p);budget['generator_elapsed_seconds']+=g['elapsed_seconds'];budget['generator_input_tokens']+=sum(g['input_tokens']);budget['generator_output_tokens']+=sum(g['output_tokens']);a=g['binding']['adapter'] or 'base';budget['by_generator_adapter'][a]=budget['by_generator_adapter'].get(a,0)+len(g['outputs'])
    for p in (RUN/'cache').glob('*/*.json'):
        c=load(p);assert c['outputs_sha256']==digest(c['outputs']);budget['victim_cache_calls']+=1;budget['victim_responses']+=len(c['outputs']);budget['victim_elapsed_seconds']+=c['elapsed_seconds']
    sensitivity={'primary_scoring_version':2,'changes':CHANGES,'all_original_scores_validated':True,'v1_joint_means':{m:float(np.mean(v)) for m,v in old_means.items()},'changed_output_scores':len(score_changes),'changed_outputs':score_changes,'scorer_v2_sha256':sha(CODE_ROOT/'analysis_tools/workshop_functional_scoring_v2.py')}
    write(output/'scoring_revision_audit.json',sensitivity)
    write(output/'selection_audit.json',selection_checks)
    result={'protocol':protocol,'scoring_revision':sensitivity,'n_functional_targets':len(rows),'n_functional_cases_per_target':20,'n_utility_screened':len(screened),'n_extraction_targets':len(extraction),'functional':stats,'learned_minus_baseline':comparisons,'learned_minus_ablation':ablations,'protected_minus_original':utility,'utility_screened_descriptive':conditional,'extraction':exstats,'budget':budget,'high_recovery_low_text_count':sum(r['high_recovery_low_text'] for r in rows),'selected_empty_count':sum(r['selected_prompt_empty'] for r in rows),'source_hashes_verified':verified}
    result['extraction_by_functional_eligibility']=exstrata
    result['input_audit_summary']={'recovery_inputs_checked':len(input_audit['recovery_input_checks']),'tgps_targets_checked':30,'prefetch_comparisons':len(input_audit['prefetch_candidate_comparisons'])}
    write(output/'metrics.json',result);write(output/'per_target_metrics.json',rows);write(output/'extraction_metrics.json',extraction);write(output/'scored_outputs.json',fullrecords)
    with (output/'functional_summary.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['method','n_targets','task_pass','contract_pass','joint_pass','joint_ci95_low','joint_ci95_high'])
        for m in METHODS:w.writerow([m,len(rows),stats[m]['task']['mean'],stats[m]['contract']['mean'],stats[m]['joint']['mean'],*stats[m]['joint']['ci95']])
    with (output/'per_target.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['id','task','scope','utility_screen',*METHODS,'learned_source_f1'])
        for r in rows:w.writerow([r['id'],r['task'],r['measurement_scope'],r['utility_screen'],*[r['metrics'][m]['joint'] for m in METHODS],r['text_metrics']['learned']['token_f1']])
    analysis_source_hashes={str(p.relative_to(CODE_ROOT)):sha(p) for p in [
        CODE_ROOT/'analysis_tools/report_workshop_functional_suite.py',
        CODE_ROOT/'analysis_tools/workshop_functional_benchmark.py',
        CODE_ROOT/'analysis_tools/workshop_functional_scoring_v2.py']}
    files=['metrics.json','functional_summary.csv','per_target.csv','per_target_metrics.json',
           'selection_audit.json','scoring_revision_audit.json','final_attack_input_audit.json']
    write(output/'final_validation.json',{
        'five_groups_complete':True,'functional_targets':len(eligible),
        'functional_methods':len(METHODS),'final_scored_responses':len(fullrecords),
        'extraction_responses':len(targets)*6*2,'source_hashes_verified':verified,
        'frozen_benchmark_verified':True,'all_saved_scores_recomputed':True,
        'selections_recomputed':len(selection_checks),
        'benchmark_source_binding':source_binding,'scorer_source_binding':scorer_binding,
        'analysis_source_sha256':analysis_source_hashes,
        'artifact_paths_relative_to':'output directory',
        'artifacts':{name:sha(output/name) for name in files}})
    print(json.dumps({'complete':True,'functional':{m:stats[m]['joint']['mean'] for m in METHODS},
                      'utility_screened':len(screened),'output_dir':str(output)},indent=2))


if __name__=='__main__':main()
