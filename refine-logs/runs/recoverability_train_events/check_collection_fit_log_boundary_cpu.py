"""Replay the actual log-path expressions without starting collection or training."""
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = Path(r'C:\Users\gb\.codex_tmp\gola_setup_20261002').resolve()


def expression(name, variable):
    tree = ast.parse((HERE / name).read_text(encoding='utf-8'))
    nodes = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == variable for target in node.targets)]
    assert len(nodes) == 1
    return compile(ast.Expression(nodes[0]), name, 'eval')


def main():
    collector = expression('collection_pipeline.py', 'log_path')
    original_fit = expression('fit_pipeline.py', 'log')
    corrected_fit = expression('complete_fit_after_M0.py', 'log')
    with tempfile.TemporaryDirectory(prefix='train_events_log_boundary_', dir=WORKSPACE) as folder:
        setup = Path(folder).resolve()
        assert setup.is_relative_to(WORKSPACE)
        old = [eval(collector, {'SETUP': setup, 'phase': 'full', 'gpu': gpu}) for gpu in range(4)]
        for path in old:
            path.write_bytes(b'preserved completed collection log\n')
        first = eval(original_fit, {'SETUP': setup, 'stage': 'full', 'arm': {'gpu': 0}})
        assert first == old[0]
        red = subprocess.run([sys.executable, '-c', 'from pathlib import Path; Path(' + repr(str(first)) + ").open('x')"],
                             capture_output=True, text=True)
        assert red.returncode == 1 and 'FileExistsError' in red.stderr
        fresh = [eval(corrected_fit, {'SETUP': setup, 'arm': {'gpu': gpu}}) for gpu in range(4)]
        assert not set(old) & set(fresh) and len(set(fresh)) == 4
        for path in fresh:
            with path.open('x') as stream:
                stream.write('new complete fit log\n')
        assert all(path.read_bytes() == b'preserved completed collection log\n' for path in old)
        assert all(path.read_text() == 'new complete fit log\n' for path in fresh)
        receipt = {'status': 'PASS', 'original_actual_source_log_expression_failed_with_FileExistsError': True,
                   'corrected_actual_source_log_expression_all_four_create_exclusively': True,
                   'all_four_collection_logs_unchanged': True,
                   'red_exit_code': red.returncode, 'CUDA_or_NN_imports': 0, 'optimizer_steps': 0,
                   'scope': 'Actual AST log expressions and exclusive pathlib creation in an isolated temporary directory; no remote/model execution.'}
    (HERE / 'actual_log_boundary_CPU_regression.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
