"""Audit the fixed MiMo RL release; download data only, never execute dataset code.

Requires requests and pyarrow. Run from the repository root:
  python scripts/analyze-mimo-rl.py --cache /tmp/mimo-rl-analysis \
    --output TechReport/MiMo/data/MiMoV2.6-RL-oss-summary.json
The first run downloads five Parquets and 925 verifier metadata files.
"""
import argparse
import collections, concurrent.futures, json, pathlib, statistics, time
import requests
import pyarrow.parquet as pq
parser=argparse.ArgumentParser()
parser.add_argument('--cache', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
args=parser.parse_args()
ROOT=args.cache
(ROOT/'meta').mkdir(parents=True, exist_ok=True)
REV='639865fd3374018d6cb29b9fb82dd531406fcf5f'
BASE=f'https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss/resolve/{REV}/'
for domain in ['code','cyber','general','webdev','music']:
 path=ROOT/(domain+'.parquet')
 if not path.exists():
  remote='general/train.parquet' if domain=='general' else domain+'.parquet'
  response=requests.get(BASE+remote,timeout=120)
  response.raise_for_status()
  pq.read_table(__import__('io').BytesIO(response.content))
  path.write_bytes(response.content)
data={k:pq.read_table(ROOT/(k+'.parquet')).to_pylist() for k in ['code','cyber','general','webdev','music']}
def inst(r):return json.loads(r['extra_info']['instance_json'])
def prompt(r):return ''.join(m['content'] for m in r['prompt'])
def stats(a):
 a=sorted(a);return {'n':len(a),'min':min(a),'median':statistics.median(a),'p90':a[int((len(a)-1)*.9)],'max':max(a)}
summary={'revision':REV,'audit_date':'2026-09-29','length_unit':'Unicode characters; no tokenizer; prompt concatenates message contents','domains':{}}
for k,rows in data.items():
 ps=[prompt(r) for r in rows]
 summary['domains'][k]={'rows':len(rows),'prompt_chars':stats([len(p) for p in ps]),'unique_prompts':len(set(ps)),'sources':dict(collections.Counter(r['data_source'] for r in rows)),'empty_ground_truth':sum(r['reward_model']['ground_truth']=='' for r in rows),'reward_styles':dict(collections.Counter(r['reward_model']['style'] for r in rows))}
code=data['code'];ratios=[len(inst(r)['test_patch'])/len(prompt(r)) for r in code]
summary['code_tests']={'patch_chars':stats([len(inst(r)['test_patch']) for r in code]),'per_task_ratio':stats(ratios),'longer_than_prompt':sum(x>1 for x in ratios),'unique_images':len(set(inst(r)['docker_image'] for r in code))}
rows=[r for r in data['general'] if inst(r)['dataset_type']=='general_agent']
def fetch(r):
 j=inst(r);uid=j['env_task_dir'].split('/')[-1];p=ROOT/'meta'/(uid+'.json')
 if not p.exists():
  for trial in range(4):
   try:
    res=requests.get(BASE+'general/'+j['env_task_dir']+'/verifier_meta.json',timeout=30);res.raise_for_status();meta=res.json();p.write_text(json.dumps(meta,ensure_ascii=False));break
   except Exception:
    if trial==3:raise
    time.sleep(1+trial)
 return r,json.loads(p.read_text())
pairs=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
 for i,pair in enumerate(ex.map(fetch,rows)):
  pairs.append(pair)
  if (i+1)%100==0:print('loaded',i+1,flush=True)
methods=collections.Counter();tiers=collections.Counter();entries=[];patterns=collections.Counter()
for r,m in pairs:
 items=m.get('items',[]);methods.update(x.get('method','missing') for x in items);tiers.update(x.get('tier','missing') for x in items)
 # Semantic rubric text only; exclude identifiers, file paths, JSON syntax, executable checks.
 txt=''.join(str(x.get(f,'') or '') for x in items for f in ['question','pass_anchor','desc'])
 patterns[tuple(sorted(set(x.get('method','missing') for x in items)))]+=1
 entries.append({'id':inst(r)['instance_id'],'prompt_chars':len(prompt(r)),'rubric_chars':len(txt),'ratio':len(txt)/len(prompt(r)),'items':len(items),'rule_code_chars':len(m.get('check_code','') or ''),'gates':sum(bool(x.get('gate')) for x in items)})
summary['general_rubrics']={'tasks':len(entries),'rubric_text_fields':['question','pass_anchor','desc'],'prompt_chars':stats([x['prompt_chars'] for x in entries]),'rubric_chars':stats([x['rubric_chars'] for x in entries]),'per_task_ratio':stats([x['ratio'] for x in entries]),'longer_than_prompt':sum(x['ratio']>1 for x in entries),'items_per_task':stats([x['items'] for x in entries]),'methods':dict(methods),'tiers':dict(tiers),'method_combinations':{'+'.join(k):v for k,v in patterns.items()},'tasks_with_gates':sum(x['gates']>0 for x in entries),'tasks_with_check_code':sum(x['rule_code_chars']>0 for x in entries)}
summary['music']={k:dict(collections.Counter(str(r['extra_info'][k]) for r in data['music'])) for k in ['lang','length','nvoice_want','meter']}
summary['cyber_identity']={'unique_instance_ids':len(set(inst(r)['instance_id'] for r in data['cyber'])),'unique_images':len(set(inst(r)['docker_image'] for r in data['cyber']))}
summary['webdev_categories']=dict(collections.Counter(inst(r)['category'] for r in data['webdev']))
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(ROOT/'general-per-task.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n')
print('Summary saved:',args.output)
