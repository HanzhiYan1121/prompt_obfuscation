"""Resumable victim-only stages for the five frozen-model experiments."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'recovery_tools')]
from release_config import DATA_ROOT, model_path, configure_hf_environment
ROOT=DATA_ROOT
from analysis_tools.workshop_functional_benchmark import RUN,write,digest,sha,score,EXTRACTION_QUERIES

def load(p):return json.loads(Path(p).read_text())
def key(target):return f'{target["index"]:02}'
def token_f1(a,b):
    from collections import Counter
    import re
    aa=Counter(re.findall(r'\w+|[^\w\s]',a.lower()));bb=Counter(re.findall(r'\w+|[^\w\s]',b.lower()));n=sum((aa&bb).values())
    return 2*n/(sum(aa.values())+sum(bb.values())) if aa or bb else 1.

class Victim:
    def __init__(self):
        configure_hf_environment()
        import torch
        from src.model import Model
        self.torch=torch
        torch.manual_seed(20260917)
        self.model=Model(model_path('victim'),'4bit')
        self.model.name_or_path='meta-llama/Meta-Llama-3.1-8B-Instruct';self.model.tokenizer.name_or_path='meta-llama/Meta-Llama-3.1-8B-Instruct'
        self.loaded=time.time()

    def generate(self,target,condition,queries,system=None,maximum=256):
        """Cache binds actual inputs, frozen soft artifacts and decoding settings."""
        from torch.utils.data import DataLoader
        from data.utils import TextDataset,create_collate_fn
        from src.output_generation import generate_model_responses,generate_model_responses_replace
        d=ROOT/target['result_dir'];soft=condition=='protected'
        if system is None:system=(d/'original_prompt.txt').read_text() if condition=='original' else ''
        params=load(d/'params.json') if soft else None
        gen={'do_sample':False,'temperature':None,'top_p':None,'top_k':None,'max_new_tokens':maximum,'num_return_sequences':1}
        binding={'target_id':target['id'],'condition':condition,'queries':queries,'system':system,'soft_sha256':sha(d/'best_candidate.pt') if soft else None,'source_sha256':target['source_sha256'],'params_sha256':sha(d/'params.json') if soft else None,'generation':gen,'batch_size':10,'model':'Llama-3.1-8B-Instruct NF4 BF16','seed':20260917}
        h=digest(binding);p=RUN/'cache'/key(target)/(h+'.json')
        if p.exists():
            old=load(p);assert old['binding']==binding and len(old['outputs'])==len(queries)
            assert old['outputs_sha256']==digest(old['outputs']);return old['outputs']
        self.torch.manual_seed(20260917)
        collate=create_collate_fn(self.model.tokenizer,params['system_prompt'] if soft else system)
        loader=DataLoader(TextDataset(queries),batch_size=10,collate_fn=collate)
        started=time.time()
        if soft:
            emb=self.torch.load(d/'best_candidate.pt',map_location='cpu',weights_only=True)
            ids=self.model.tokenizer(params['system_prompt'],return_tensors='pt',add_special_tokens=False).input_ids[0]
            raw=generate_model_responses_replace(self.model,loader,gen,emb,ids,True,params['obf_sys_prompt_len'],self.model.tokenizer.pad_token_id)
        else:raw=generate_model_responses(self.model,loader,gen)
        outputs=[str(x[0]) for x in raw];assert len(outputs)==len(queries)
        write(p,{'binding':binding,'outputs':outputs,'outputs_sha256':digest(outputs),'elapsed_seconds':time.time()-started,'runner_sha256':sha(__file__),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'time':time.time()})
        print(f'GENERATED {key(target)} {condition} n={len(queries)} seconds={time.time()-started:.1f}',flush=True)
        return outputs

def rows(shard,nshards,eligible=True):
    ts=load(RUN/'manifest.json')['targets']
    return [r for r in ts if (r['eligible'] or not eligible) and r['index']%nshards==shard]

def collect(args):
    b=load(RUN/'benchmark.json');v=None
    for t in rows(args.shard,args.nshards):
        path=RUN/'collected'/f'{key(t)}.json'
        if path.exists():continue
        v=v or Victim();cs=b[t['id']]['cases'];q=[c['query'] for c in cs]
        out={}
        for name,system in [('original',None),('protected',None),('null',''),('generic','You are a helpful assistant.')]:
            answers=v.generate(t,name,q,system)
            out[name]=[{'case_id':c['case_id'],'partition':c['partition'],'query':c['query'],'output':a,'scores':score(t['task'],c,a)} for c,a in zip(cs,answers)]
        write(path,{'target':t,'benchmark_sha256':digest(b[t['id']]),'conditions':out})
        print('COLLECTED',t['id'],flush=True)

def extract(args):
    v=None
    for t in rows(args.shard,args.nshards,False):
        path=RUN/'extraction'/f'{key(t)}.json'
        if path.exists():continue
        v=v or Victim();out={c:v.generate(t,c,EXTRACTION_QUERIES,maximum=512) for c in ['original','protected']}
        write(path,{'target':t,'queries':EXTRACTION_QUERIES,'conditions':out})
        print('EXTRACTED',t['id'],flush=True)

def select(args):
    v=None
    for t in rows(args.shard,args.nshards):
        cp=RUN/'candidates'/f'{key(t)}.json'
        if not cp.exists():raise RuntimeError(f'Missing candidates: {cp}')
        candidates=load(cp);col=load(RUN/'collected'/f'{key(t)}.json');ss=[r for r in col['conditions']['protected'] if r['partition']=='selection'];q=[r['query'] for r in ss];refs=[r['output'] for r in ss]
        for name,cs in candidates['variants'].items():
            if args.variants and name not in args.variants.split(','):continue
            path=RUN/'selected'/name/f'{key(t)}.json'
            if path.exists():
                assert load(path)['candidate_digest']==digest(cs);continue
            v=v or Victim();scored=[]
            for i,c in enumerate(cs):
                ans=v.generate(t,f'select_{name}_{i}',q,c['prompt']);sf=sum(token_f1(a,b) for a,b in zip(ans,refs))/len(refs)
                scored.append({**c,'selection_outputs':ans,'selection_f1':sf,'candidate_index':i})
            best=max(scored,key=lambda x:x['selection_f1'])
            write(path,{'id':t['id'],'method':name,'candidate_digest':digest(cs),'selection_queries':q,'selection_references':refs,'candidates':scored,'selected':best,'selection_rule':'highest mean token F1 vs protected, first-index tie break; no gold/functional scoring'})
            print('SELECTED',key(t),name,flush=True)

def evaluate(args):
    b=load(RUN/'benchmark.json');v=None;ts=[x for x in load(RUN/'manifest.json')['targets'] if x['eligible']]
    learned={t['id']:load(RUN/'selected/learned'/f'{key(t)}.json')['selected']['prompt'] for t in ts}
    for t in rows(args.shard,args.nshards):
        cs=[c for c in b[t['id']]['cases'] if c['partition']=='evaluation'];q=[c['query'] for c in cs]
        ev=[r for r in load(RUN/'collected'/f'{key(t)}.json')['conditions']['protected'] if r['partition']=='evidence']
        systems={}
        for p in sorted((RUN/'selected').glob(f'*/{key(t)}.json')):systems[p.parent.name]=load(p)['selected']['prompt']
        donor=None;idx=ts.index(t)
        for j in range(1,len(ts)):
            d=ts[(idx+j)%len(ts)]
            if learned[d['id']].strip() and learned[d['id']].strip()!=learned[t['id']].strip():donor=d;break
        if donor is None:raise RuntimeError('No genuinely different nonempty mismatch donor')
        systems['mismatch']=learned[donor['id']]
        systems['icl']='Perform the same task and follow the response conventions demonstrated below. Apply the inferred rules to the new user input.\n\n'+'\n\n'.join(f'User input:\n{r["query"]}\nResponse:\n{r["output"]}' for r in ev)
        for name,system in systems.items():
            if args.variants and name not in args.variants.split(','):continue
            path=RUN/'evaluated'/name/f'{key(t)}.json'
            if path.exists():
                assert load(path)['prompt_sha256']==hashlib.sha256(system.encode()).hexdigest();continue
            v=v or Victim();ans=v.generate(t,name,q,system)
            write(path,{'id':t['id'],'task':t['task'],'method':name,'prompt':system,'prompt_sha256':hashlib.sha256(system.encode()).hexdigest(),'benchmark_sha256':digest(b[t['id']]),'mismatch_donor_id':donor['id'] if name=='mismatch' else None,'records':[{'case_id':c['case_id'],'query':c['query'],'output':a,'scores':score(t['task'],c,a)} for c,a in zip(cs,ans)]})
            print('EVALUATED',key(t),name,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['collect','extract','select','evaluate']);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--nshards',type=int,default=1);ap.add_argument('--variants',default='');args=ap.parse_args()
    assert 0<=args.shard<args.nshards
    globals()[args.stage](args)
    write(RUN/'completed'/f'{args.stage}_{args.shard}of{args.nshards}_{args.variants or "all"}.json',{'time':time.time(),'runner_sha256':sha(__file__),'stage':args.stage,'shard':args.shard,'nshards':args.nshards,'variants':args.variants})

if __name__=='__main__':main()
