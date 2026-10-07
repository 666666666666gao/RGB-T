"""CPU receipt/schema verification before queued event-training dependencies."""
import argparse
import json
from pathlib import Path

from research.event_rollout_data import EventRolloutData


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',nargs='+',required=True)
    args=p.parse_args()
    for root in args.roots:
        config=json.loads((Path(root)/'config.json').read_text())
        dataset=EventRolloutData([root],config['partition'])
        assert len(dataset)==config['clips']
        print(json.dumps(dict(root=root,clips=len(dataset),videos=len(dataset.names),partition=config['partition'],
                              schema='causal_sequence_prefix_v1',exact_prefix_receipt=True)),flush=True)


if __name__=='__main__':main()
