import os,sys
os.environ.setdefault("NPROMPTS","16")
sys.argv=["x"]
exec(open("/mmfs1/home/evanly/RL-on-HRM-Text/rl/compare_prompts.py").read().replace(
 'for path in ["rl-data/dapo_prompts.jsonl","rl-data/openr1_dapo_prompts.jsonl"]:\n    evalfile("/mmfs1/home/evanly/RL-on-HRM-Text/"+path)',
 'evalfile("/mmfs1/home/evanly/RL-on-HRM-Text/rl-data/openr1_dapo_prompts.jsonl")'))
