"""Run reviewed collection sources without changing active evaluation files."""
import argparse
import importlib.util
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', required=True)
    args, collection_args = parser.parse_known_args()
    root = Path(args.source_directory)
    for name in ('collect_recoverability', 'recoverability_tracker'):
        qualified = 'research.' + name
        spec = importlib.util.spec_from_file_location(qualified, root/(name+'.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    sys.argv = [sys.argv[0], *collection_args]
    sys.modules['research.collect_recoverability'].main()


if __name__ == '__main__':
    main()
