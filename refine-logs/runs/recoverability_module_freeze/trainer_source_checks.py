"""Source/stdlib fixtures only: no torch import, neural execution, SSH, or GPU."""
import argparse
import ast
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
TRAINER = "research/train_recoverability.py"
MODULES = "research/recoverability_modules.py"


def parsed(path, baseline=False):
    source = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT, text=True) if baseline else (ROOT / path).read_text()
    return ast.parse(source, filename=path)


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def assignment(body, name):
    return next(node for node in body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))


def field(node, name):
    return next(value for key, value in zip(node.keys, node.values) if isinstance(key, ast.Constant) and key.value == name)


def run_nodes(nodes, namespace):
    exec(compile(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]), TRAINER, "exec"), namespace)


def evaluate(node, namespace):
    return eval(compile(ast.Expression(body=copy.deepcopy(node)), TRAINER, "eval"), namespace)


def same(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


class Scalar:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def clone(self):
        return Scalar(self.value)

    def square(self):
        return Scalar(self.value ** 2)

    def sum(self):
        return self

    def sqrt(self):
        return Scalar(math.sqrt(self.value))

    def __float__(self):
        return float(self.value)


class Parameter(Scalar):
    def __init__(self, value, requires_grad=True):
        super().__init__(value)
        self.requires_grad = requires_grad
        self.grad = None

    def numel(self):
        return 1


class Module:
    def __init__(self, count, requires_grad=True):
        self.params = [Parameter(float(index), requires_grad) for index in range(count)]

    def parameters(self):
        return iter(self.params)

    def requires_grad_(self, requires_grad):
        for parameter in self.params:
            parameter.requires_grad = requires_grad
        return self

    def state_dict(self):
        return {str(index): parameter for index, parameter in enumerate(self.params)}


def main():
    tree, base, modules, base_modules = parsed(TRAINER), parsed(TRAINER, True), parsed(MODULES), parsed(MODULES, True)
    trainer, old_trainer = function(tree, "main"), function(base, "main")
    freeze = next(node for node in trainer.body if isinstance(node, ast.For) and ast.unparse(node.iter) == "args.frozen_modules")
    gradient_loop = next(node for node in ast.walk(trainer) if isinstance(node, ast.For) and ast.unparse(node.target) == "(name, module)" and ast.unparse(node.iter) == "groups.items()")
    old_gradient_loop = next(node for node in ast.walk(old_trainer) if isinstance(node, ast.For) and ast.unparse(node.target) == "(name, module)" and ast.unparse(node.iter) == "groups.items()")
    changed = assignment(trainer.body, "changed")
    changed_index = trainer.body.index(changed)
    final_assertions = trainer.body[changed_index + 1:changed_index + 3]
    assert all(isinstance(node, ast.Assert) for node in final_assertions)

    # Remove exactly the reviewed additions; all remaining trainer AST must equal HEAD.
    normalized = copy.deepcopy(trainer)
    normalized.body = [node for node in normalized.body if not (isinstance(node, ast.For) and ast.unparse(node.iter) == "args.frozen_modules")]
    new_config, old_config = assignment(normalized.body, "config").value.right, assignment(old_trainer.body, "config").value.right
    config_index = next(index for index, key in enumerate(new_config.keys) if key.value == "trainable_parameters")
    new_config.values[config_index] = copy.deepcopy(field(old_config, "trainable_parameters"))
    normalized_gradient_loop = next(node for node in ast.walk(normalized) if isinstance(node, ast.For) and ast.unparse(node.target) == "(name, module)" and ast.unparse(node.iter) == "groups.items()")
    assert isinstance(normalized_gradient_loop.body[0], ast.If)
    assert same(normalized_gradient_loop.body[0].orelse[0], old_gradient_loop.body[0])
    normalized_gradient_loop.body = normalized_gradient_loop.body[0].orelse + normalized_gradient_loop.body[1:]
    index = normalized.body.index(assignment(normalized.body, "changed"))
    old_index = old_trainer.body.index(assignment(old_trainer.body, "changed"))
    normalized.body[index + 1:index + 3] = [copy.deepcopy(old_trainer.body[old_index + 1])]
    receipt = assignment(normalized.body, "receipt").value
    retained = [(key, value) for key, value in zip(receipt.keys, receipt.values) if key.value not in ("frozen_modules", "frozen_module_weights_unchanged")]
    receipt.keys, receipt.values = map(list, zip(*retained))
    assert same(normalized, old_trainer)
    normalized_args = copy.deepcopy(function(tree, "arguments"))
    normalized_args.body = [node for node in normalized_args.body if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and node.value.args and isinstance(node.value.args[0], ast.Constant) and node.value.args[0].value == "--frozen-modules")]
    assert same(normalized_args, function(base, "arguments"))
    for name in ("load_data", "forward", "evaluate"):
        assert same(function(tree, name), function(base, name))
    assert same(modules, base_modules)

    # These are named routing fixtures; the scalars are not neural parameters.
    toy_torch = SimpleNamespace(equal=lambda left, right: left.value == right.value,
                                stack=lambda values: Scalar(sum(value.value for value in values)))
    results = []
    for variant, frozen in (("ABC_default", []), ("BC", ["A"]), ("AC", ["B"]), ("AB", ["C"]), ("C", ["A", "B"])):
        memory, search, action, c1 = Module(9), Module(4), Module(4), Module(8, False)
        model = SimpleNamespace(memory=memory, search=search, action=action, c1=c1,
                                parameters=lambda: (parameter for module in (memory, search, action, c1) for parameter in module.parameters()))
        namespace = {"model": model, "args": SimpleNamespace(frozen_modules=frozen), "torch": toy_torch, "np": SimpleNamespace(isfinite=math.isfinite)}
        run_nodes([assignment(trainer.body, "groups"), freeze, assignment(trainer.body, "initial"), assignment(trainer.body, "parameters")], namespace)
        active = [name for name in ("A", "B", "C") if name not in frozen]
        groups = namespace["groups"]
        expected = [parameter for name in active for parameter in groups[name].parameters()]
        assert set(map(id, namespace["parameters"])) == set(map(id, expected))
        assert all(id(parameter) not in set(map(id, namespace["parameters"])) for parameter in c1.parameters())
        logged_counts = evaluate(field(assignment(trainer.body, "config").value.right, "trainable_parameters"), namespace)
        assert logged_counts == {name: (0 if name in frozen else len(module.params)) for name, module in groups.items()}
        for name in active:
            for parameter in groups[name].parameters():
                parameter.grad = Scalar(.25)
                parameter.value += .125
        namespace.update(gradients={}, max_gradients={name: 0. for name in groups})
        run_nodes([gradient_loop, changed, *final_assertions], namespace)
        assert namespace["changed"] == {name: name in active for name in groups}
        assert all(namespace["gradients"][name] == 0 for name in frozen)
        rejected = []
        mutations = [("missing_active_weight_change", lambda: namespace["changed"].__setitem__(active[0], False), final_assertions)]
        if frozen:
            mutations += [("frozen_weight_change", lambda: namespace["changed"].__setitem__(frozen[0], True), final_assertions),
                          ("frozen_gradient_present", lambda: setattr(groups[frozen[0]].params[0], "grad", Scalar(.1)), [gradient_loop])]
        for label, mutate, nodes in mutations:
            namespace["changed"] = {name: name in active for name in groups}
            mutate()
            try:
                run_nodes(nodes, namespace)
            except AssertionError:
                rejected.append(label)
            else:
                raise AssertionError(f"{variant}: did not reject {label}")
        results.append({"variant": variant, "frozen": frozen, "active": active,
                        "fixture_optimizer_parameter_objects": len(expected), "rejected_invalid_receipts": rejected})

    # Execute only the argument parser and decision-input wrapper, without importing the trainer.
    namespace = {"argparse": argparse, "__doc__": "Source-only fixture"}
    run_nodes([function(tree, "arguments")], namespace)
    sys.argv = [TRAINER, "--train", "train", "--validation", "validation", "--output", "fixture"]
    default_args = namespace["arguments"]()
    assert default_args.frozen_modules == []
    assert (default_args.seed, default_args.lr, default_args.threshold, default_args.weight_decay) == (42, 1e-4, .03, 1e-4)
    decision_fields = ast.literal_eval(assignment(modules.body, "DECISION_FIELDS").value)
    label_fields = ast.literal_eval(assignment(tree.body, "LABEL_FIELDS").value)
    assert set(decision_fields).isdisjoint(label_fields)
    namespace = {"DECISION_FIELDS": decision_fields}
    run_nodes([function(tree, "forward")], namespace)
    data = {name: object() for name in decision_fields + label_fields}
    captured = namespace["forward"](lambda batch: batch, data)
    for name in label_fields:
        data[name] = object()
    assert namespace["forward"](lambda batch: batch, data) == captured
    assert tuple(captured) == decision_fields

    selection = next(node for node in ast.walk(trainer) if isinstance(node, ast.If) and ast.unparse(node.test) == "metrics['utility'] > best")
    saves = []
    namespace = {"best": .5, "save": lambda epoch, metrics: saves.append(epoch)}
    for epoch, utility in enumerate((.5, .49, .51, .51, .50), 1):
        namespace.update(epoch=epoch, metrics={"utility": utility})
        run_nodes([selection], namespace)
    assert saves == [3]
    initial_index = trainer.body.index(assignment(trainer.body, "initial"))
    assert trainer.body.index(freeze) < initial_index < trainer.body.index(assignment(trainer.body, "parameters"))
    parent_load = next(node for node in trainer.body if isinstance(node, ast.If) and ast.unparse(node.test) == "args.init_checkpoint")
    assert trainer.body.index(parent_load) < trainer.body.index(freeze)
    assert changed_index < next(index for index, node in enumerate(trainer.body) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "checkpoint" for target in node.targets))
    assert "torch" not in sys.modules

    report = {"status": "PASS", "scope": "source AST and stdlib scalar fixtures only", "neural_execution": False,
              "ssh": False, "gpu_execution": False, "runtime_certification": False,
              "default_ABC_ast_equivalence_after_removing_exact_freeze_patch": True,
              "canonical_modules_ast_equal_HEAD": True, "variants": results,
              "decision_label_fields_disjoint": True, "forward_label_mutation_does_not_change_inputs": True,
              "strict_best_selection_fixture_saved_epochs": saves,
              "parent_load_before_freeze_before_snapshot_before_optimizer": True,
              "completion_change_checks_precede_best_reload": True,
              "limitations": ["Scalar fixtures validate routing/assertion logic, not autograd or real state tensors.",
                              "Actual M0 3 epochs/9 updates and independent 128-output CPU audit remain pending.",
                              "No deployment, full training, tracking result, or cause of prior negative results is certified."]}
    (RUN / "trainer_source_checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
