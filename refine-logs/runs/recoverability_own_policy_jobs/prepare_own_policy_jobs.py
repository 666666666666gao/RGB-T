import json
import numpy as np
from pathlib import Path
from trackit.datasets.MMOT.specialization.memory_mapped.dataset import MultiModalObjectTrackingDataset_MemoryMapped
root=Path('/data/gb/outputs/recoverability_own_policy_jobs_20261003')
root.mkdir(exist_ok=True)
split=json.loads(Path('/data/gb/outputs/c1_initial_seed42/split.json').read_text())
dataset=MultiModalObjectTrackingDataset_MemoryMapped.load('/data/wangwj/dataset/LasHeR','trackit/datasets/cache/MultiModalObjectTrackingDataset_MemoryMapped/filtered/lasher-train-b00458d848ef249c438f993815ddbb19.np')
by_name={dataset[i].get_name():dataset[i] for i in range(len(dataset))}
hard=json.loads(Path('/data/gb/setup/own_policy_hard_requests.json').read_text())['jobs']
uniform=json.loads(Path('/data/gb/outputs/recoverability_train_s42_20261003/config.json').read_text())['jobs'][:128]
validation=json.loads(Path('/data/gb/outputs/recoverability_validation_s100042_20261003/config.json').read_text())['jobs']
for partition, jobs in [('train',hard+uniform),('validation',validation)]:
    selected=[]
    seen=set()
    skipped=[]
    for job in jobs:
        name,q=job['sequence'],int(job['query_frame'])
        assert name in split[partition] and 0<q<=1024
        boxes=by_name[name].get_all_bounding_boxes()
        valid=np.isfinite(boxes).all(1)&(boxes[:,2:]>boxes[:,:2]).all(1)
        if not valid[0] or not valid[q:q+4].all() or len(boxes[q:q+4])!=4:
            skipped.append(job)
            continue
        if (name,q) not in seen:
            selected.append(job)
            seen.add((name,q))
    receipt={'source':'Existing TRAIN-only hard missing-candidate audit plus128uniform TRAIN jobs; exact existing128 held-out jobs for validation','partition':partition,'jobs':selected,'requested':len(jobs),'unique_valid':len(selected),'invalid_label_requests_excluded':skipped,'after256':sum(j['query_frame']>256 for j in selected),'past_GT_enters_online_decisions':False}
    (root/(partition+'.json')).write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='jobs'}))