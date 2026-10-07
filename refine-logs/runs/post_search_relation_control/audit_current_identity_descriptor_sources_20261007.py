"""Verify current descriptor source paths on CPU; no model or result replay."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

parent = Path('C:/Users/gb/.codex_tmp/gola_setup_20261002')
repo = parent / 'GOLA-source'
base = repo / 'refine-logs/runs/post_search_relation_control'
output = base / 'identity_descriptor_source_audit.json'
assert not output.exists()
revision = '7691742b3fc121d80684aef262e56d04ef7d757f'
names = (
    'trackit/models/methods/GOLA/modules/patch_embed.py',
    'trackit/models/methods/GOLA/gola_full_finetune.py',
    'research/candidate_learning.py', 'research/collect_recoverability.py',
    'research/recoverability_tracker.py', 'research/recoverability_modules.py')
sources = {}
for name in names:
    raw = (repo / name).read_bytes()
    original = subprocess.run(['git', 'show', revision + ':' + name], cwd=repo,
                              capture_output=True, check=True).stdout
    # Two upstream Windows files differ from Git only by CRLF/LF, as verified.
    assert raw.replace(b'\r\n', b'\n') == original.replace(b'\r\n', b'\n')
    sources[name] = dict(working_raw_sha256=hashlib.sha256(raw).hexdigest(),
                         scientific_Git_raw_sha256=hashlib.sha256(original).hexdigest(),
                         raw_bytes_equal=raw == original,
                         source_content_equal_after_CRLF_normalization=True)

def method(name, class_name, method_name):
    tree = ast.parse((repo / name).read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)

def call_names(node):
    return [ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)]

patch = method(names[0], 'PatchEmbedNoSizeCheck', 'forward')
assert call_names(patch) == ['self.proj', 'x.flatten(2).transpose', 'self.norm', 'x.flatten']
roi = method(names[3], 'InstanceExtractor', 'extract_with_output')
assert 'self.base.patch_embed' in call_names(roi) and 'instance_pool' in call_names(roi)
extractor = method(names[2], 'FrozenCandidateExtractor', 'extract_with_output')
assert 'self.base._fusion' in call_names(extractor)
fusion = method(names[1], 'GOLABaseline_DINOv2', '_fusion')
assert 'self.blocks[i]' in call_names(fusion) and 'self.norm' in call_names(fusion)
projection = method(names[5], 'InstanceMemory', '__init__')
assert 'nn.Sequential' in call_names(projection) and 'self.projection.load_state_dict' in call_names(projection)
fits = json.loads((base / 'actual_four_full_fit_intake.json').read_text())
assert all(not value['completion']['frozen_modules'] for value in fits['full_fits'].values())
record = dict(
    status='SOURCE_DATAFLOW_VERIFIED_NOT_PERFORMANCE_DIAGNOSIS',
    scientific_revision=revision, source_files=sources,
    patch_embed_forward_calls=call_names(patch),
    facts={
        'instance_anchor': 'Each sensor is separately patch-embedded and foreground-pooled, before position/type addition and Transformer blocks.',
        'candidate_instance_descriptor': 'Area-overlap weighted pooling of each sensor patch_embed output within the predicted candidate box; not a Transformer-encoded instance ROI.',
        'C1_candidate_features': 'Selected search locations after joint template/RGB/TIR attention and final norm; the returned search_i stream already depends on jointly observed inputs.',
        'memory_projection': 'A separately constructed LayerNorm/Linear/GELU initialized with C1 projection weights; no alias to frozen C1 projection. All four fit configurations have no frozen A/B/C module.',
        'stored_history': 'The current accepted candidate instance descriptor is stored; other candidates do not form persistent identity tracklets in this path.'},
    code_locations={
        'patch_projection': names[0] + ':22',
        'joint_transformer': names[1] + ':95',
        'sensor_instance_ROI': names[3] + ':44',
        'sensor_initial_anchor': names[4] + ':36',
        'selected_instance_update': names[4] + ':138',
        'independent_projection': names[5] + ':23'},
    not_proven=[
        'These feature choices explain the completed native score deficits.',
        'Transformer ROI descriptors or a new instance encoder improve accuracy.',
        'Every projection parameter actually changed; current receipts verify module-level changes.',
        'Jointly attended stream outputs are independent sensor reliability evidence.'],
    future_evidence_order='Close current bidirectional complete evaluation first; before persistent candidate memory, compare the current descriptor with stronger region representations under the same memory/candidate/visual budget, and keep modality identity and reliability claims separate.',
    no_neural_operation=True, no_current_NN_source_recipe_or_selection_change=True,
    new_formal_accuracy=False)
output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=record['status'], active_scientific_source_matches=6,
                     no_neural_operation=True, new_formal_accuracy=False)), flush=True)
