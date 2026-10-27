# Soft-prompt functional recovery: experiment code and saved outputs

This branch contains the experiment implementation for recovering executable
text prompts from the behavior of a soft-prompt-obfuscated system. It includes
structured recovery training utilities, frozen-checkpoint inference, baselines,
task checks, text-extraction evaluation, and paired statistical analysis.

Qwen3.6-27B generates recovered prompts. Llama-3.1-8B-Instruct executes the original,
protected, and recovered prompts. The main evaluation reuses six existing
recovery adapters and existing soft prompts; its runner does not train models.

The `functional-recovery-code` branch has independent Git history and contains
experiment source, configuration examples, technical instructions, CPU tests,
and the saved functional-evaluation artifacts described below. Model weights,
training datasets, presentation materials, manuscript files, document builders,
and plotting code are not included.

## Get the code

```bash
git clone --branch functional-recovery-code --single-branch \
  https://github.com/HanzhiYan1121/prompt_obfuscation.git
cd prompt_obfuscation
```

## Published evaluation artifacts

The following 332 JSON files (6,828,379 bytes) are included, byte-for-byte as
saved in the original experiment:

```text
results/workshop_functional_recovery_v1/
  manifest.json
  benchmark.json
  collected/        # 30 JSON files
  evaluated/        # 300 JSON files, 30 targets for each of 10 methods
```

- `manifest.json`: the 83 audited targets, including 30 admitted functional
  targets, and the historical benchmark-source and protocol hashes.
- `benchmark.json`: the 30 functional tasks, each with 4 calibration, 4 evidence,
  2 selection, and 20 final evaluation cases, including queries and evaluator
  reference information.
- `collected/`: 3,600 records across original, protected, generic, and null
  conditions and all four partitions; 2,400 are final evaluation responses.
- `evaluated/`: 6,000 final evaluation responses and the corresponding prompts
  for learned recovery, zero1, zero6, ICL, T-GPS, mismatch, and four access
  ablations. Each method has 600 responses.

Together these contain the 8,400 final functional-evaluation responses across
14 conditions. The saved `scores` fields use the original v1 scorer; final
reported results use `analysis_tools/workshop_functional_scoring_v2.py`.
Recompute v2 scores from the saved outputs and benchmark cases instead of
averaging the historical `scores` fields. The released files retain their
original hashes and values.

These artifacts support inspection and functional rescoring without model
inference. They are a selected output release, not a complete archived run:
the full report command additionally requires protocol and provenance files,
attack-input and candidate-selection records, extraction outputs, and frozen
source assets described below. Model weights and training data remain external.
Benchmark references and final evaluation responses are evaluator-only data;
they must not be supplied to a recovery model during attack construction.

Keep this published snapshot unchanged. The example runtime configuration writes
new experiments to `runs/functional_recovery_new`.

## CPU checks

Python 3.10 or later is required. The synthetic benchmark and statistical tests
run without model weights, CUDA, or network access:

```bash
python -m pip install -r requirements-cpu.txt
python -m unittest discover -s tests -v
python recovery_tools/run_workshop_functional_suite.py --gpu 0 --dry-run
```

The checks exercise every admitted target template, task/contract separation,
the scoring revisions, bounded AST parsing, paired statistics, and separation
between attacker evidence and evaluator-only inputs.

## Required external assets

Running the full experiment also requires the original authorized assets in an
external `data_root`. The implementation does not silently replace missing
assets or select a different target collection.

```text
data_root/
  results/
    workshop_qwen36_reuse_v1/strict_targets.json
    sixcat200_<category>_structured_recovery/recovery_inputs.json
    sixcat200_<category>_structured_qwen36_27b_lora/
      split.json
      adapter/
        adapter_config.json
        adapter_model.safetensors
  <each target's result_dir>/
    original_prompt.txt
    params.json
    best_candidate.pt
    prepared_data/train_data.json
    prepared_data/test_data.json
```

The six categories are `role_persona`, `task_instruction`, `format_spec`,
`constraints`, `context_knowledge`, and `tool_protocol`.

- `strict_targets.json` contains a `sample` list with the audited 83 target IDs
  and their metadata.
- Each `recovery_inputs.json` is a list of records with `id`,
  `target_original_prompt`, `result_dir`, and `behavior_examples`. Metadata and
  `obfuscation` projections are needed for the explicitly stronger-access
  ablations. Paired examples contain user inputs and original/protected outputs.
- `split.json` supplies the historical training IDs. Preparation excludes IDs
  and normalized source strings in the union of all six training sets.
- `best_candidate.pt` contains the previously optimized soft vectors.
  `params.json` supplies at least `system_prompt` and `obf_sys_prompt_len`.
- Historical optimization/test inputs are used to check exact query separation.
  Model and dataset access must be obtained separately under their own terms.

## Configure and run inference

Install `requirements.txt` in an environment with a CUDA-compatible PyTorch
build. The version ranges are a dependency specification, not an exact lock of
the historical run. In particular, use a Transformers build that supports the
Qwen3.6-27B checkpoint; the old project's Transformers 4.x lock is not used here.

Edit a copy of `configs/runtime.example.json` to point at the external assets
and cached model checkpoints. Paths in the JSON are relative to the JSON file.
The model identifiers must refer to the intended Qwen3.6-27B recovery backbone
and Llama-3.1-8B-Instruct executor. Offline loading is the default.

```bash
python recovery_tools/run_workshop_functional_suite.py \
  --config configs/runtime.example.json \
  --gpu 0
```

This runs the following stages serially on the explicitly chosen GPU:

1. `prepare`: validate target exclusion, freeze task partitions and source hashes.
2. `collect`: execute original, protected, generic, and empty-prompt conditions.
3. `extract`: apply six fixed extraction queries to all 83 targets.
4. `recover`: generate learned, zero-shot, and ablation candidates.
5. `tgps`: run the bounded local T-GPS adaptation.
6. `select`: rank candidates using two protected selection responses.
7. `evaluate`: execute selected prompts, mismatch, and direct imitation on final inputs.
8. `score`: recompute v1/v2 scores, candidate selections, and paired statistics.

The runner uses the current Python interpreter. It does not discover idle GPUs,
wait for pre-existing jobs, or launch training. Existing manifests are validated
instead of overwritten. Inference caches and completion files support resuming
the same frozen configuration. Use a new run directory when changing inputs,
checkpoints, inference settings, or code.

Configuration can also be supplied through `RECOVERY_CONFIG`,
`RECOVERY_DATA_ROOT`, and `RECOVERY_RUN_DIR`. Optional JSON keys `adapter_root`
or `adapters` override checkpoint locations; `hf_cache` overrides the default
Hugging Face cache. Setting `local_files_only` to `false` explicitly allows model
loading to use the network.

## Evaluation interface

Each of the 30 admitted functional targets has 4 calibration inputs, 4 recovery
evidence inputs, 2 selection inputs, and 20 final inputs. Six trained adapters
produce six candidates, with no target-category oracle. The primary generator
receives only protected behavior examples. Source instructions, metadata,
original-service answers, gold labels, calibration inputs, and final inputs are
excluded from that primary generation interface.

The 14 conditions include original/protected/learned, seven baselines, and four
access ablations. The primary metric is the target-mean joint task/contract pass
rate. The reporter uses target bootstrap intervals, paired sign-flip tests,
separate Holm families for baselines and ablations, and lexical source-overlap
metrics. It writes JSON/CSV and validation records only.

To analyze a completed run without writing into it:

```bash
RECOVERY_CONFIG=configs/runtime.example.json \
python analysis_tools/report_workshop_functional_suite.py \
  --output-dir runs/recomputed-analysis
```

Scoring a historical run requires its complete saved outputs and frozen assets.
If its benchmark-source hash predates the portability edits, the reporter
requires the matching archived source under `data_root` and records that
verification mode separately from the code executed for reanalysis. Newly
prepared runs bind directly to this release's source hashes.

## Recovery training utilities

The evaluation itself reuses adapters. For inspecting or reproducing their
training procedure, this branch also includes:

- `recovery_tools/prepare_structured_6d_recovery_targets.py`: deterministic
  six-field target construction from authorized source records.
- `recovery_tools/train_acl100_recovery_lora.py`: the general QLoRA training
  entry point used by the six-category recipe.
- `recovery_tools/eval_acl100_recovery_lora.py`: structured prediction evaluation.
- `configs/training_recipe.json`: the recorded 160-train/40-test, five-epoch
  recipe with six paired behavior examples and the six output fields.

These utility names predate the six-category experiment. Their defaults are
general historical defaults, so use the recipe's explicit arguments and the
original split file to reproduce the intended model. They are not invoked by
the frozen-model evaluation runner.

```bash
python recovery_tools/train_acl100_recovery_lora.py --help
python recovery_tools/prepare_structured_6d_recovery_targets.py --help
```

`SOURCE_PROVENANCE.json` records the pre-export hashes of the original selected
source files. Portability edits are confined to this export. See
`THIRD_PARTY_NOTICES` for retained attribution; this export adds no new
project-wide license.
