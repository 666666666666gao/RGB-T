# RGB-T research on pretrained GOLA

This repository extends official GOLA commit `339c737cda6a24be667b6e5abdc721e8d6046f05`.
Complete pretrained GOLA-B remains frozen; C1 learns candidate quality and ranking.
Both requested benchmarks have now been evaluated in full with corrected C1
coordinate decoding, actual dataset annotations and unchanged epoch-3 weights.
LasHeR covers 245 sequences / 220703 frames; RGBT234 covers 234 / 116649.

| Dataset / variant | PR or MPR (%) | NPR (%) | SR or MSR (%) |
|---|---:|---:|---:|
| LasHeR / pretrained baseline | 76.598449 | 73.101973 | 61.037425 |
| LasHeR / C1 v2 | 77.860973 | 74.154518 | 61.951314 |
| RGBT234 / pretrained baseline | 91.778858 | N/A | 69.232593 |
| RGBT234 / C1 v2 | 91.817009 | N/A | 69.234047 |

LasHeR differences are +1.2625/+1.0525/+0.9139 percentage points;
RGBT234 is essentially tied (+0.0382/+0.0015). All five individual 95% paired
sequence bootstrap intervals include zero (5000 shared resamples, seed42,
fixed checkpoints); this run does not establish stable improvement.
[Complete intervals and sequence win/loss counts](refine-logs/runs/paired_sequence_bootstrap_v2.json).

Complete [LasHeR report](refine-logs/runs/lasher_v2_complete_report.json) and
[RGBT234 report](refine-logs/runs/rgbt234_v2_complete_report.json) include all
19/12 attributes, every sequence, native curves, failure/recovery event records,
paired frame harms/rescues, candidate recall and rejection counts, template-write
localization proxies, ten calibration bins, latency percentiles and peak memory.
CSV and PNG artifacts use the same dataset_v2 prefix in `refine-logs/runs`.
LasHeR same-frame C1 reselection rescues477 and harms584; RGBT234 rescues239
and harms99. Localization proxies are not complete distractor identity labels.
Timing includes decode/crop/forward/selection/update, excludes initialization;
different concurrent HDD loads prevent a fair acceleration claim.
[Actual parameter counts](refine-logs/runs/model_parameter_counts.json): frozen
base88093445, C1 added117889 (0.1338%), C3 cached utility head119041.

Corrected C2 internal validation is negative: 16 TRAIN-held-out sequences,
4041 valid frames; C1 meanIoU0.710122 versus C2 0.704633 and box-only0.704921.
C3 v3 preserves C1 Hann selection at zero residual; all512 TRAIN/256 internal
validation choices match C1 exactly. Its 20epochs/320 optimizer steps completed,
with actual new-parameter updates and frozen base. Best remains **epoch0**;
epoch1-5 tie, later epochs do not improve it. Strict reload reproduces every
validation metric exactly. Initial utility0.759345 equals C1; last utility0.759062
is worse, with0 improved/1 worsened/1 reselected clip despite reduced MSE.
[All v3 epochs](refine-logs/runs/c3_hann_v3_metrics.csv),
[reload and whole-data control](refine-logs/runs/c3_hann_v3_reload_verification.json).
Previous v2 results are retained; their epoch0 gain used a different initial
pure-quality policy and cannot be attributed to training. Teacher generation
and cached optimization cost are separately reported.
These internal C2/C3 results are not official online tracking PR/SR.

Learned A compression, B multi-future prediction and coupled C state recovery
are now implemented. Fresh source review and actual GPU M0 passed: A/B/C
parameters changed, frozen C1 had no gradients, and bounded online smoke checks
passed. Small validation choices did not improve and the smoke videos had no
branch switches. All3072 TRAIN clips (3032 unique sequence/query pairs) and
512 held-out clips are complete. Joint training finished30epochs/1440steps,
batch64, with A/B/C changed and frozen C1 exactly unchanged. Cached validation
choices stayed equal to C1 in all31epoch records; cached best is epoch0,
so there is no demonstrated learned selection gain. Strict initial/best/last
reload and all held-out calibration checks passed.
New branch-template and learned-slot
provenance/forecast diagnostics passed source review and GPU correspondence.
Sampling uses full causal prefixes and batch32. Full98TRAIN-held-out videos
(49418frames each) compare initial, trained last and trained box-only online;
checkpoint selection uses sequence-weighted IoU before official tests. All98 videos are complete: initial .737194672342, trained last .739523904518, last box-only .738778577820. Epoch30 was locked before official testing; these are internal localization diagnostics, not PR/SR or a baseline improvement.
Four GPUs now run full ABC and matched box-only on both core benchmarks, with native scores/all attributes/sequences/curves/events, branch and slot provenance, forecast calibration, complete costs and paired uncertainty. The complete98 internal report shows last-minus-initial +0.232923pp (95% paired CI [-0.345903,0.896431]) and last-minus-box +0.074533pp ([-0.210859,0.423822]); both include zero, and some repair diagnostics worsen. [All internal metrics and events](refine-logs/runs/abc_internal_full_report.json) and [every internal sequence](refine-logs/runs/abc_internal_per_sequence.csv) retain the negative results. Full ABC benchmark results remain pending. The target is +2pp on all five overall native metrics. Existing v1 artifacts and initial C3
negative results are retained as history. Six full internal zero-head control
videos (1401 frames) had exactly equal raw trajectories after the minimal
float32 box-scaling correction; this limited control is not a full benchmark
zero-head equivalence claim. One detailed handoff contains current results,
protocols, negative findings, code reviews, dataset paths and remaining work.

A TRAIN-only search audit found189 failed sampled queries,151 without a correct
candidate. Of88 missing-candidate queries whose target center was outside the
native crop, the top-probability motion crop covered0; all three motion crops
covered21, with four regions including the original. Validation found25 missing
candidates,11 outside the crop and4 covered by any proposal. This is geometric
center coverage without visual inference, not tracking accuracy or a gain.
[Corrected native-crop audit](refine-logs/runs/abc_train_search_audit_v2.json)
uses the original factor4/minsize10 provider; the earlier inline rectangular
approximation is obsolete. No checkpoint or live tracker was changed.

The complete-core merger now waits for all four full evaluation receipts and
both state-control receipts, then combines all five acceptance metrics, every
attribute and1,437 sequence rows (479 sequences x baseline/ABC/box-only), all
native/branch/calibration/forecast costs and paired intervals. It also retains
the complete TRAIN audits,31epochs and earlier baseline/C1 mechanism reports.
Its [CPU schema review](refine-logs/runs/abc_complete_merge_review.json) passed
with negative outcomes retained; this is fixture evidence, not benchmark results.
The merger is waiting and the formal ABC results are still pending.

- [Detailed experiment handoff](docs/HANDOFF_20261002.md)
- Sanity: `bash scripts/run_c1_sanity.sh`
- Initial training after sanity passes: `bash scripts/run_c1_initial.sh`
- Strict online inference: `bash scripts/run_online.sh --dataset lasher --root /path/to/testingset --variant c1 --output /path/to/results`.
- Initial training results, current jobs, known issues and next steps are all recorded in the single handoff document above.
- Current datasets: LasHeR and RGBT234. VTUAV is deferred.

The original paper/code description below is retained for attribution. Its
reported results are the authors' baseline results, not results of our additions.

# [AAAI2026] Group Orthogonal Low-Rank Adaptation for RGB-T Tracking

Official implementation of [Group Orthogonal Low-Rank Adaptation for RGB-T Tracking](https://arxiv.org/abs/2512.05359). (AAAI 2026)

[[Pretrained Models](https://drive.google.com/drive/folders/1p9DDPcP251-_mvLVUQ8BXbdEjDA4sttM?usp=sharing)] [[Weights](https://drive.google.com/drive/folders/1WQ7bZ9ZmXkKTryShMsTdmG5h1yLwGGxz?usp=sharing)] [[Raw Results](https://drive.google.com/drive/folders/1tScEdLUNNtSaGqXuchCM4I5LC8K6S7jX?usp=sharing)] [[Training Logs](https://drive.google.com/drive/folders/1gaN1ck4n75goq8WRIPWr3ECra0cbjw8i?usp=sharing)]

<img src="./assets/framework.jpg"/>

>**Abstract:** Parameter-efficient fine-tuning has emerged as a promising paradigm in RGB-T tracking, enabling downstream task adaptation by freezing pretrained parameters and fine-tuning only a small set of parameters. This set forms a rank space made up of multiple individual ranks, whose expressiveness directly shapes the model's adaptability. However, quantitative analysis reveals low-rank adaptation exhibits significant redundancy in the rank space, with many ranks contributing almost no practical information. This hinders the model's ability to learn more diverse knowledge to address the various challenges in RGB-T tracking. To address this issue, we propose the Group Orthogonal Low-Rank Adaptation (GOLA) framework for RGB-T tracking, which effectively leverages the rank space through structured parameter learning. Specifically, we adopt a rank decomposition partitioning strategy utilizing singular value decomposition to quantify rank importance, freeze crucial ranks to preserve the pretrained priors, and cluster the redundant ranks into groups to prepare for subsequent orthogonal constraints. We further design an inter-group orthogonal constraint strategy. This constraint enforces orthogonality between rank groups, compelling them to learn complementary features that target diverse challenges, thereby alleviating information redundancy. Experimental results demonstrate that GOLA effectively reduces parameter redundancy and enhances feature representation capabilities, significantly outperforming state-of-the-art methods across four benchmark datasets and validating its effectiveness in RGB-T tracking tasks.

## News
**[Dec. 24, 2025]**
* We’re thrilled to release [MMLoRAT](https://github.com/MelanTech/MMLoRAT), a multimodal extension of LoRAT with improved conciseness and overall refinement.

**[Dec. 11, 2025]**
* We have released the code, weights, raw results and training logs.

**[Nov. 08, 2025]**
* Our GOLA has been accepted by AAAI 2026.

## Upstream reported performance (not this run)
<table style="text-align: center;">
  <tr>
    <th rowspan="2">Variant</th>
    <th colspan="3">LasHeR</th>
    <th colspan="2">RGBT234</th>
    <th colspan="2">RGBT210</th>
    <th colspan="2">GTOT</th>
    <th rowspan="2">FPS</th>
  </tr>
  <tr>
    <th>PR(%)</th>
    <th>NPR(%)</th>
    <th>SR(%)</th>
    <th>MPR(%)</th>
    <th>MSR(%)</th>
    <th>PR(%)</th>
    <th>SR(%)</th>
    <th>MPR(%)</th>
    <th>MSR(%)</th>
  </tr>
  <tr>
    <td>GOLA-B</td>
    <td>77.5</td>
    <td>73.9</td>
    <td>61.6</td>
    <td>92.2</td>
    <td>69.5</td>
    <td>90.9</td>
    <td>67.0</td>
    <td>92.8</td>
    <td>78.5</td>
    <td>125</td>
  </tr>
  <tr>
    <td>GOLA-L</td>
    <td>78.1</td>
    <td>74.5</td>
    <td>61.9</td>
    <td>92.8</td>
    <td>71.3</td>
    <td>92.0</td>
    <td>68.7</td>
    <td>95.3</td>
    <td>80.9</td>
    <td>64</td>
  </tr>
</table>

## Prerequisites
### Environment
Assuming you have a `Python 3.10.15` environment with pip installed.

#### system packages (ubuntu)
```shell
apt update
apt install -y libturbojpeg
```
#### install pytorch
```shell
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

#### extra python packages
```shell
pip install -r requirements.txt
```
This codebase should also work on Windows and macOS for debugging purposes.

### Dataset
The paths should be organized as follows:
```
-- LasHeR/trainingset
    |-- 1boygo
    |-- 1handsth
    ...
```

### Prepare ```consts.yaml```
Fill in the paths.
```yaml
LasHeR_PATH: '/path/to/LasHeR0428/'
RGBT234_PATH: '/path/to/RGBT234/'
RGBT210_PATH: '/path/to/RGBT210/'
GTOT_PATH: '/path/to/GTOT/'
```

## Quick Start
* Our code performs evaluation automatically when model training is complete. 
  * **Model weight** is saved in ```/path/to/output/run_id/checkpoint/epoch_{last}/model.bin```.
  * **Performance metrics** can be found on terminal output.
  * **Tracking results** are saved in ```/path/to/output/run_id/eval/epoch_{last}/```.

* The performance metrics obtained from automatic evaluation **differ from** official evaluation tools, so we recommend using official tools for re-evaluation.

### Preparation for pretrained models
* Download [base.bin & large.bin](https://drive.google.com/drive/folders/1p9DDPcP251-_mvLVUQ8BXbdEjDA4sttM?usp=sharing) and put them in the `pretrained_models` folder.
* Execute `pretrained_models/params_svd_grouping.py` to group the ranks of LoRA parameters.
```shell
python params_svd_grouping.py --weight ./base.bin --n_retain 16 --n_groups 8
python params_svd_grouping.py --weight ./large.bin --n_retain 16 --n_groups 8
```
* New weights will be saved in the same directory as the original weights.
* You can skip this step and directly download `base_gola_r16_g8.bin` and `large_gola_r16_g8.bin`.

### Training
* Using `sh/train.sh` for training (Linux with NVIDIA GPU only)
```shell
export CUDA_VISIBLE_DEVICES=0,1  # Specify your GPU IDs

output_dir="/path/to/output/directory"  # Specify your output directory
weight_path="/path/to/pretrained/weight"  # Specify your pretrained model path

# Dry run
python ../main.py GOLA dinov2 --dry_run --distributed_nproc_per_node "${GPU_NUM}" --distributed_do_spawn_workers --disable_wandb --weight_path $weight_path

# GOLA-B
python ../main.py GOLA dinov2 --distributed_nproc_per_node "${GPU_NUM}" --distributed_do_spawn_workers --disable_wandb --weight_path $weight_path --output_dir="$output_dir" |& tee -a "$output_dir/train_stdout-$timestamp.log"

# GOLA-L
python ../main.py GOLA dinov2 --mixin_config large --distributed_nproc_per_node "${GPU_NUM}" --distributed_do_spawn_workers --disable_wandb --weight_path $weight_path --output_dir="$output_dir" |& tee -a "$output_dir/train_stdout-$timestamp.log"
```

### Evaluation
* We use [rgbt](https://github.com/opacity-black/RGBT_toolkit) library to evaluate the performance on RGB-T datasets.
* You can skip steps 1 and 2 and simply use the zip file saved during the training phase.

You can run evaluation with the following procedure to reproduce the results reported in the paper:

1. Edit ```config/_dataset/test-mm.yaml``` to specify the dataset to be evaluated.
```yaml
datasets:
  - name: "LasHeR"
    type: "MMOT"
    splits:
      - "test"

#  - name: "RGBT234"  # Uncomment to evaluate RGBT234
#    type: "MMOT"

#  - name: "RGBT210"  # Uncomment to evaluate RGBT210
#    type: "MMOT"

#  - name: "GTOT"  # Uncomment to evaluate GTOT
#    type: "MMOT"
```
* Note: Uncomment one dataset at a time to evaluate.

2. Edit ```sh/test.sh``` to specify the weight path and output directory.
```shell
output_dir="/path/to/output"
weight_path="path/to/weight.bin"
timestamp=$(date +"%Y.%m.%d-%H.%M.%S")

mkdir -p $output_dir

# Evaluation with GOLA-B
python ../main.py GOLA dinov2 --eval --mixin_config evaluation --distributed_nproc_per_node "${GPU_NUM}" --distributed_do_spawn_workers --weight_path $weight_path --device cuda --disable_wandb --output_dir=$output_dir |& tee -a "$output_dir/eval_stdout-$timestamp.log"

# Evaluation with GOLA-L
python ../main.py GOLA dinov2 --eval --mixin_config evaluation --mixin_config large --distributed_nproc_per_node "${GPU_NUM}" --distributed_do_spawn_workers --weight_path $weight_path --device cuda --disable_wandb --output_dir=$output_dir |& tee -a "$output_dir/eval_stdout-$timestamp.log"
```

3. Unzip the tracking results to a folder of your choice.

4. Edit and run the evaluation script in ```sh/evaluation.sh```.
```shell
# Evaluate on LasHeR
python ../evaluation.py lasher --tracker_names GOLA --result_paths /path/to/tracking/results

# Evaluate on RGBT234
python ../evaluation.py rgbt234 --tracker_names GOLA --result_paths /path/to/tracking/results

# Evaluate on RGBT210
python ../evaluation.py rgbt210 --tracker_names GOLA --result_paths /path/to/tracking/results

# Evaluate on GTOT
python ../evaluation.py gtot --tracker_names GOLA --result_paths /path/to/tracking/results
```

### Profile Model
* Using `profile_model.py` for model profiling.
```shell
python ../profile_model.py GOLA dinov2 --device cuda  # GOLA-B
python ../profile_model.py GOLA dinov2 --mixin_config large --device cuda  # GOLA-L
```

## Acknowledgements
- This repo is based on [LoRAT](https://github.com/LitingLin/LoRAT), we thank for it's `trackit` framework, which helps us to quickly implement our ideas.
- We thank the [rgbt](https://github.com/opacity-black/RGBT_toolkit) library for facilitating evaluation in a Python environment.

## Citation
```bibtex
@inproceedings{gola,
  title={Group Orthogonal Low-Rank Adaptation for RGB-T Tracking},
  author={Shao, Zekai and Hu, Yufan and Liu, jingyuan and Fan, Bin and Liu, Hongmin},
  booktitle={AAAI},
  year={2026}
} 
```
