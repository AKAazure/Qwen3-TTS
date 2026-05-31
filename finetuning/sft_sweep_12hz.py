# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
import argparse
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ALLOWED_SWEEP_PARAMS = {
    "lr",
    "weight_decay",
    "grad_clip_max_norm",
    "batch_size",
    "gradient_accumulation_steps",
    "lr_scheduler",
    "lr_warmup_steps",
    "lr_warmup_ratio",
    "lr_min_ratio",
    "num_epochs",
    "eval_batch_size",
    "max_eval_batches",
}
ALLOWED_LR_SCHEDULERS = {"constant", "linear", "cosine"}
EVAL_JSON_PREFIX = "SFT_EVAL_JSON "
TRAIN_JSON_PREFIX = "SFT_TRAIN_JSON "
EPOCH_JSON_PREFIX = "SFT_EPOCH_JSON "
SWEEP_TRIAL_JSON_PREFIX = "SFT_SWEEP_TRIAL_JSON "
SWEEP_OVERFIT_JSON_PREFIX = "SFT_SWEEP_OVERFIT_JSON "
BEST_CHECKPOINT_NAME = "checkpoint-best"
RETAINED_BEST_CHECKPOINT_DIRNAME = "_retained_best_checkpoint"
DEFAULT_OVERFIT_STOP_CONFIG = {
    "enabled": False,
    "min_epochs": 10,
    "patience": 3,
    "min_eval_regression_ratio": 0.03,
    "min_train_improvement_ratio": 0.02,
}
EVAL_TEXT_PATTERN = re.compile(r"^Epoch\s+(?P<epoch>\d+)\s+\|\s+Eval Loss:\s+(?P<loss>[0-9.eE+-]+)")
TRAIN_TEXT_PATTERN = re.compile(r"^Epoch\s+(?P<epoch>\d+)\s+\|\s+Train Loss:\s+(?P<loss>[0-9.eE+-]+)")
SMOKE_SAMPLE_SCRIPT = r"""
import argparse

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def torch_dtype(name):
    if name == "auto":
        return None
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", required=True)
parser.add_argument("--output_path", required=True)
parser.add_argument("--text", required=True)
parser.add_argument("--speaker", required=True)
parser.add_argument("--language", required=True)
parser.add_argument("--instruct", default="")
parser.add_argument("--dtype", default="float16", choices=["auto", "bfloat16", "float16", "float32"])
parser.add_argument("--attn_implementation", default="sdpa", choices=["none", "eager", "sdpa"])
args = parser.parse_args()

load_kwargs = {"device_map": "auto"}
dtype = torch_dtype(args.dtype)
if dtype is not None:
    load_kwargs["dtype"] = dtype
if args.attn_implementation != "none":
    load_kwargs["attn_implementation"] = args.attn_implementation

model = Qwen3TTSModel.from_pretrained(args.checkpoint_path, **load_kwargs)
generation_kwargs = {
    "text": args.text,
    "speaker": args.speaker,
    "language": args.language,
}
if args.instruct.strip():
    generation_kwargs["instruct"] = args.instruct.strip()
wavs, sample_rate = model.generate_custom_voice(**generation_kwargs)
sf.write(args.output_path, wavs[0], sample_rate)
print(f"smoke_wav={args.output_path}")
print(f"sample_rate={int(sample_rate)}")
"""


def main():
    args = parse_args()
    config = load_sweep_config(args)
    validate_sweep_config(config)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "sweep_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    trials_jsonl = output_root / "trials.jsonl"
    samples_jsonl = output_root / "samples.jsonl"
    trials_jsonl.write_text("", encoding="utf-8")
    samples_jsonl.write_text("", encoding="utf-8")

    method = config.get("method", "grid")
    if method == "grid":
        records = run_grid(args, config, trials_jsonl)
    elif method == "optuna_tpe":
        records = run_optuna_tpe(args, config, trials_jsonl)
    else:
        raise ValueError(f"unsupported sweep method: {method}")

    best = select_best_trial(records)
    if best is None:
        print("best_trial=none")
        return 0 if args.dry_run else 1

    promoted = None
    if args.promote_best_to and not args.dry_run:
        promoted = promote_best_checkpoint(best, Path(args.promote_best_to))
        best["promoted_checkpoint_path"] = promoted
        print(f"promoted_checkpoint={promoted}")
    if not args.dry_run:
        deleted_checkpoints = delete_trial_checkpoints(output_root)
        if deleted_checkpoints:
            mark_deleted_checkpoints(records, deleted_checkpoints)
        deleted_retained = delete_retained_best_checkpoint(output_root)
        if deleted_retained:
            clear_retained_checkpoint_path(records, deleted_retained)
            if best.get("retained_best_checkpoint_path") == deleted_retained:
                best["retained_best_checkpoint_path"] = None
        rewrite_trial_records(records, trials_jsonl)
        best["trial_checkpoints_deleted"] = best.get("trial_checkpoints_deleted", 0)
        print(f"trial_checkpoints_deleted={best['trial_checkpoints_deleted']}")

    best_path = output_root / "best_trial.json"
    best_path.write_text(json.dumps(best, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"best_trial_json={best_path}")
    print(f"best_trial_id={best['trial_id']}")
    print(f"best_eval_loss={best['best_eval_loss']}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen3-TTS SFT hyperparameter sweep trials.")
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--eval_jsonl", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant", choices=sorted(ALLOWED_LR_SCHEDULERS))
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.0)
    parser.add_argument("--lr_min_ratio", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip_max_norm", type=float, default=2.0)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--checkpoint_interval_epochs", type=int, default=1)
    parser.add_argument("--keep_last_checkpoints", type=int, default=3)
    parser.add_argument("--eval_interval_epochs", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--smoke_text", type=str, default=None)
    parser.add_argument("--smoke_language", type=str, default="Japanese")
    parser.add_argument("--smoke_instruct", type=str, default="")
    parser.add_argument("--smoke_output_dir", type=str, default=None)
    parser.add_argument("--smoke_dtype", type=str, default="float16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--smoke_attn_implementation", type=str, default="sdpa", choices=["none", "eager", "sdpa"])
    parser.add_argument("--sweep_config_json", type=str, default=None)
    parser.add_argument("--sweep_config_path", type=str, default=None)
    parser.add_argument("--promote_best_to", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if bool(args.sweep_config_json) == bool(args.sweep_config_path):
        raise ValueError("pass exactly one of --sweep_config_json or --sweep_config_path")
    if not args.dry_run and not args.eval_jsonl:
        raise ValueError("eval_jsonl is required unless --dry_run is set")
    if args.smoke_text is not None and not args.smoke_text.strip():
        raise ValueError("smoke_text must not be empty when provided")
    validate_training_args(vars(args))
    return args


def load_sweep_config(args):
    if args.sweep_config_json:
        raw = args.sweep_config_json
        source = "--sweep_config_json"
    else:
        source_path = Path(args.sweep_config_path)
        raw = source_path.read_text(encoding="utf-8")
        source = str(source_path)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid sweep config JSON from {source}: {error}") from error


def validate_sweep_config(config):
    if not isinstance(config, dict):
        raise ValueError("sweep config must be a JSON object")
    method = config.get("method", "grid")
    if method not in {"grid", "optuna_tpe"}:
        raise ValueError("sweep method must be grid or optuna_tpe")
    if config.get("metric", "eval_loss") != "eval_loss":
        raise ValueError("only metric=eval_loss is supported")
    if config.get("direction", "minimize") != "minimize":
        raise ValueError("only direction=minimize is supported")
    params = normalized_sweep_params(config)
    if not params:
        raise ValueError("sweep config requires at least one parameter in params, optimizer.params, or scheduler.params")
    unknown = sorted(set(params) - ALLOWED_SWEEP_PARAMS)
    if unknown:
        raise ValueError(f"unsupported sweep params: {unknown}")
    if "max_trials" in config and (not isinstance(config["max_trials"], int) or config["max_trials"] <= 0):
        raise ValueError("max_trials must be a positive integer")
    validate_overfit_stop_config(config.get("overfit_stop", {}))
    if method == "grid":
        for name, values in params.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"grid param {name} must be a non-empty list")
            for value in values:
                validate_single_param(name, value)
    else:
        for name, spec in params.items():
            validate_optuna_param(name, spec)


def normalized_sweep_params(config):
    merged = {}
    for path in (("params",), ("optimizer", "params"), ("scheduler", "params")):
        source = config
        for key in path:
            if not isinstance(source, dict) or key not in source:
                source = None
                break
            source = source[key]
        if source is None:
            continue
        if not isinstance(source, dict):
            raise ValueError(".".join(path) + " must be an object")
        for name, spec in source.items():
            if name in merged:
                raise ValueError(f"duplicate sweep param {name!r} in config")
            merged[name] = spec
    return merged


def validate_overfit_stop_config(raw_config):
    if raw_config is None:
        return
    if not isinstance(raw_config, dict):
        raise ValueError("overfit_stop must be an object")
    known = set(DEFAULT_OVERFIT_STOP_CONFIG)
    unknown = sorted(set(raw_config) - known)
    if unknown:
        raise ValueError(f"unsupported overfit_stop fields: {unknown}")
    if "enabled" in raw_config and not isinstance(raw_config["enabled"], bool):
        raise ValueError("overfit_stop.enabled must be a boolean")
    for name in ("min_epochs", "patience"):
        if name in raw_config and (not isinstance(raw_config[name], int) or raw_config[name] <= 0):
            raise ValueError(f"overfit_stop.{name} must be a positive integer")
    for name in ("min_eval_regression_ratio", "min_train_improvement_ratio"):
        if name in raw_config and (not isinstance(raw_config[name], (int, float)) or float(raw_config[name]) < 0):
            raise ValueError(f"overfit_stop.{name} must be a non-negative number")


def normalized_overfit_stop_config(config):
    raw_config = config.get("overfit_stop", {}) or {}
    policy = dict(DEFAULT_OVERFIT_STOP_CONFIG)
    policy.update(raw_config)
    policy["min_eval_regression_ratio"] = float(policy["min_eval_regression_ratio"])
    policy["min_train_improvement_ratio"] = float(policy["min_train_improvement_ratio"])
    return policy


def validate_optuna_param(name, spec):
    if not isinstance(spec, dict):
        raise ValueError(f"optuna param {name} must be an object")
    value_type = spec.get("type")
    if value_type == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"categorical param {name} requires non-empty choices")
        for choice in choices:
            validate_single_param(name, choice)
        return
    if value_type not in {"float", "int"}:
        raise ValueError(f"optuna param {name} type must be float, int, or categorical")
    if "low" not in spec or "high" not in spec:
        raise ValueError(f"optuna param {name} requires low and high")
    low = spec["low"]
    high = spec["high"]
    if value_type == "int":
        if not isinstance(low, int) or not isinstance(high, int) or low > high:
            raise ValueError(f"int param {name} requires integer low <= high")
    else:
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)) or low > high:
            raise ValueError(f"float param {name} requires numeric low <= high")
    if spec.get("log", False) and (low <= 0 or high <= 0):
        raise ValueError(f"log param {name} requires positive low/high")


def validate_training_args(values):
    int_positive = [
        "batch_size",
        "gradient_accumulation_steps",
        "num_epochs",
        "checkpoint_interval_epochs",
        "keep_last_checkpoints",
    ]
    int_nonnegative = ["eval_batch_size", "max_eval_batches", "lr_warmup_steps"]
    float_positive = ["lr"]
    float_nonnegative = ["weight_decay", "grad_clip_max_norm", "lr_warmup_ratio", "lr_min_ratio"]
    if values["lr_scheduler"] not in ALLOWED_LR_SCHEDULERS:
        raise ValueError(f"lr_scheduler must be one of {sorted(ALLOWED_LR_SCHEDULERS)}")
    for name in int_positive:
        if int(values[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in int_nonnegative:
        if int(values[name]) < 0:
            raise ValueError(f"{name} must be non-negative")
    for name in float_positive:
        if float(values[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in float_nonnegative:
        if float(values[name]) < 0:
            raise ValueError(f"{name} must be non-negative")
    if int(values["eval_interval_epochs"]) < 0:
        raise ValueError("eval_interval_epochs must be non-negative")
    if not 0 <= float(values["lr_warmup_ratio"]) < 1:
        raise ValueError("lr_warmup_ratio must be greater than or equal to 0 and less than 1")
    if not 0 <= float(values["lr_min_ratio"]) <= 1:
        raise ValueError("lr_min_ratio must be between 0 and 1")


def validate_single_param(name, value):
    if name == "lr_scheduler":
        if value not in ALLOWED_LR_SCHEDULERS:
            raise ValueError(f"lr_scheduler values must be one of {sorted(ALLOWED_LR_SCHEDULERS)}")
        validate_training_args({**base_param_defaults(), name: value})
        return
    if name in {
        "batch_size",
        "gradient_accumulation_steps",
        "num_epochs",
        "eval_batch_size",
        "max_eval_batches",
        "lr_warmup_steps",
    }:
        if not isinstance(value, int):
            raise ValueError(f"{name} values must be integers")
    else:
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} values must be numeric")
    validate_training_args({**base_param_defaults(), name: value})


def base_param_defaults():
    return {
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "lr": 2e-5,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 0,
        "lr_warmup_ratio": 0.0,
        "lr_min_ratio": 0.0,
        "weight_decay": 0.01,
        "grad_clip_max_norm": 2.0,
        "num_epochs": 3,
        "checkpoint_interval_epochs": 1,
        "keep_last_checkpoints": 3,
        "eval_interval_epochs": 1,
        "eval_batch_size": 0,
        "max_eval_batches": 0,
    }


def run_grid(args, config, trials_jsonl):
    records = []
    best_record = None
    trial_params = list(iter_grid_params(normalized_sweep_params(config)))
    max_trials = config.get("max_trials")
    overfit_stop_config = normalized_overfit_stop_config(config)
    if max_trials is not None:
        trial_params = trial_params[:max_trials]
    for index, params in enumerate(trial_params):
        record = run_trial(args, index, params, overfit_stop_config=overfit_stop_config, dry_run=args.dry_run)
        best_record = retain_best_checkpoint_candidate(record, best_record, Path(args.output_root), dry_run=args.dry_run)
        if not args.dry_run:
            deleted = delete_trial_checkpoints_in_trial_dir(Path(record["output_model_path"]))
            record["trial_checkpoints_deleted"] = len(deleted)
            if deleted:
                mark_deleted_checkpoints([record], deleted)
        records.append(record)
        append_jsonl(trials_jsonl, record)
    return records


def iter_grid_params(params):
    keys = list(params)
    for values in itertools.product(*(params[key] for key in keys)):
        yield dict(zip(keys, values))


def run_optuna_tpe(args, config, trials_jsonl):
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError("optuna_tpe sweep requires `python3 -m pip install optuna`") from error

    records = []
    best_record = None
    seed = int(config.get("seed", 42))
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner_name = config.get("pruner", "successive_halving")
    if pruner_name == "successive_halving":
        pruner = optuna.pruners.SuccessiveHalvingPruner()
    elif pruner_name == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    elif pruner_name == "none":
        pruner = optuna.pruners.NopPruner()
    else:
        raise ValueError("pruner must be successive_halving, hyperband, or none")

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    max_trials = int(config.get("max_trials", 10))
    overfit_stop_config = normalized_overfit_stop_config(config)

    def objective(trial):
        nonlocal best_record
        params = suggest_trial_params(trial, normalized_sweep_params(config))
        record = run_trial(
            args,
            trial.number,
            params,
            optuna_trial=trial,
            overfit_stop_config=overfit_stop_config,
            dry_run=args.dry_run,
        )
        best_record = retain_best_checkpoint_candidate(record, best_record, Path(args.output_root), dry_run=args.dry_run)
        if not args.dry_run:
            deleted = delete_trial_checkpoints_in_trial_dir(Path(record["output_model_path"]))
            record["trial_checkpoints_deleted"] = len(deleted)
            if deleted:
                mark_deleted_checkpoints([record], deleted)
        records.append(record)
        append_jsonl(trials_jsonl, record)
        if record["status"] == "pruned":
            raise optuna.TrialPruned()
        if args.dry_run:
            return 0.0
        if record["best_eval_loss"] is None or not math.isfinite(record["best_eval_loss"]):
            raise RuntimeError(f"trial {record['trial_id']} did not produce eval_loss")
        return record["best_eval_loss"]

    study.optimize(objective, n_trials=max_trials)
    return records


def suggest_trial_params(trial, params):
    suggested = {}
    for name, spec in params.items():
        value_type = spec["type"]
        if value_type == "categorical":
            suggested[name] = trial.suggest_categorical(name, spec["choices"])
        elif value_type == "int":
            suggested[name] = trial.suggest_int(name, spec["low"], spec["high"], log=bool(spec.get("log", False)))
        else:
            suggested[name] = trial.suggest_float(name, spec["low"], spec["high"], log=bool(spec.get("log", False)))
    return suggested


def run_trial(args, index, params, *, optuna_trial=None, overfit_stop_config=None, dry_run=False):
    trial_id = f"trial-{index:04d}"
    output_root = Path(args.output_root)
    trial_dir = output_root / "trials" / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_path = trial_dir / "sft_12hz.log"
    metrics_path = trial_dir / "metrics.jsonl"
    merged_params = build_trial_params(args, params)
    validate_training_args({**base_param_defaults(), **merged_params})
    command = build_sft_command(args, trial_dir, merged_params)
    overfit_stop_policy = overfit_stop_config or dict(DEFAULT_OVERFIT_STOP_CONFIG)
    best_checkpoint_path = str(trial_dir / BEST_CHECKPOINT_NAME) if args.eval_jsonl else None
    metrics_path.write_text("", encoding="utf-8")
    record = {
        "trial_id": trial_id,
        "status": "dry_run" if dry_run else "running",
        "params": merged_params,
        "checkpoint_policy": {
            "checkpoint_interval_epochs": args.checkpoint_interval_epochs,
            "keep_last_checkpoints": 1,
            "retain_trial_checkpoints": False,
            "save_best_eval_checkpoint": bool(args.eval_jsonl),
            "best_checkpoint_name": BEST_CHECKPOINT_NAME if args.eval_jsonl else None,
        },
        "command": command,
        "output_model_path": str(trial_dir),
        "log_path": str(log_path),
        "metrics_path": str(metrics_path),
        "metrics_history": [],
        "train_history": [],
        "eval_history": [],
        "final_train_loss": None,
        "final_eval_loss": None,
        "best_eval_loss": None,
        "best_eval_epoch": None,
        "checkpoint_path": None,
        "checkpoint_epoch": None,
        "best_checkpoint_path": best_checkpoint_path,
        "retained_best_checkpoint_path": None,
        "promoted_checkpoint_path": None,
        "checkpoint_deleted": False,
        "best_checkpoint_deleted": False,
        "trial_checkpoints_deleted": 0,
        "overfit_stop_policy": overfit_stop_policy,
        "overfit_stop_reason": None,
        "overfit_stop_epoch": None,
        "overfit_stop_metrics": None,
        "sample": None,
        "returncode": None,
        "duration_sec": 0.0,
    }
    (trial_dir / "trial.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"sweep_trial={trial_id}")
    print(SWEEP_TRIAL_JSON_PREFIX + json.dumps({"trial_id": trial_id, "params": merged_params}, sort_keys=True))
    print(f"sweep_trial_hyperparams={trial_id} {format_trial_params(merged_params)}")
    print("sweep_command=" + " ".join(command))
    if dry_run:
        return record

    start_time = time.monotonic()
    pruned = False
    overfit_stop = None
    script_dir = Path(__file__).resolve().parent
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(script_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
            for event in parse_metric_events(line):
                updated = update_metric_record(record, event)
                if updated["any"]:
                    write_metrics_jsonl(metrics_path, record)
                if updated["eval"] and optuna_trial is not None:
                    optuna_trial.report(event["eval_loss"], step=event["epoch"])
                if updated["eval"]:
                    overfit_stop = detect_overfit_stop(record, event, overfit_stop_policy)
                    if overfit_stop is not None:
                        apply_overfit_stop_record(record, overfit_stop)
                        print(format_overfit_stop_line(trial_id, overfit_stop))
                        print(SWEEP_OVERFIT_JSON_PREFIX + json.dumps(overfit_stop, sort_keys=True))
                        process.terminate()
                        break
                    if optuna_trial is not None and optuna_trial.should_prune():
                        pruned = True
                        print(f"sweep_trial_pruned={trial_id} epoch={event['epoch']} eval_loss={event['eval_loss']}")
                        process.terminate()
                        break
            if pruned or overfit_stop is not None:
                break
        if pruned or overfit_stop is not None:
            try:
                returncode = process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
        else:
            returncode = process.wait()

    record["returncode"] = returncode
    record["duration_sec"] = round(time.monotonic() - start_time, 3)
    if pruned:
        record["status"] = "pruned"
    elif overfit_stop is not None:
        record["status"] = "overfit_stopped"
        checkpoint = require_best_eval_checkpoint(record, trial_dir)
        attach_checkpoint_and_sample(args, record, checkpoint)
    elif returncode == 0:
        record["status"] = "completed"
        checkpoint = require_best_eval_checkpoint(record, trial_dir)
        attach_checkpoint_and_sample(args, record, checkpoint)
    else:
        record["status"] = "failed"
    (trial_dir / "trial.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if record["status"] == "failed":
        deleted = delete_trial_checkpoints_in_trial_dir(trial_dir)
        record["trial_checkpoints_deleted"] = len(deleted)
        if deleted:
            mark_deleted_checkpoints([record], deleted)
            (trial_dir / "trial.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise RuntimeError(f"{trial_id} failed with exit code {returncode}; see {log_path}")
    return record


def build_trial_params(args, overrides):
    params = {
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lr": args.lr,
        "lr_scheduler": args.lr_scheduler,
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_warmup_ratio": args.lr_warmup_ratio,
        "lr_min_ratio": args.lr_min_ratio,
        "weight_decay": args.weight_decay,
        "grad_clip_max_norm": args.grad_clip_max_norm,
        "num_epochs": args.num_epochs,
        "eval_batch_size": args.eval_batch_size,
        "max_eval_batches": args.max_eval_batches,
    }
    params.update(overrides)
    return params


def build_sft_command(args, trial_dir, params):
    command = [
        sys.executable,
        "-u",
        "sft_12hz.py",
        "--init_model_path",
        str(args.init_model_path),
        "--output_model_path",
        str(trial_dir),
        "--train_jsonl",
        str(args.train_jsonl),
        "--batch_size",
        str(params["batch_size"]),
        "--gradient_accumulation_steps",
        str(params["gradient_accumulation_steps"]),
        "--lr",
        str(params["lr"]),
        "--lr_scheduler",
        str(params["lr_scheduler"]),
        "--lr_warmup_steps",
        str(params["lr_warmup_steps"]),
        "--lr_warmup_ratio",
        str(params["lr_warmup_ratio"]),
        "--lr_min_ratio",
        str(params["lr_min_ratio"]),
        "--weight_decay",
        str(params["weight_decay"]),
        "--grad_clip_max_norm",
        str(params["grad_clip_max_norm"]),
        "--num_epochs",
        str(params["num_epochs"]),
        "--checkpoint_interval_epochs",
        str(args.checkpoint_interval_epochs),
        "--keep_last_checkpoints",
        "1",
        "--speaker_name",
        str(args.speaker_name),
    ]
    if args.eval_jsonl:
        command.extend([
            "--eval_jsonl",
            str(args.eval_jsonl),
            "--eval_interval_epochs",
            str(args.eval_interval_epochs),
            "--eval_batch_size",
            str(params["eval_batch_size"]),
            "--max_eval_batches",
            str(params["max_eval_batches"]),
            "--save_best_eval_checkpoint",
            "--best_checkpoint_name",
            BEST_CHECKPOINT_NAME,
        ])
    return command


def parse_metric_events(line):
    if line.startswith(EVAL_JSON_PREFIX):
        try:
            payload = json.loads(line[len(EVAL_JSON_PREFIX):])
        except json.JSONDecodeError:
            return []
        return [{"epoch": int(payload["epoch"]), "eval_loss": float(payload["eval_loss"])}]
    if line.startswith(TRAIN_JSON_PREFIX):
        try:
            payload = json.loads(line[len(TRAIN_JSON_PREFIX):])
        except json.JSONDecodeError:
            return []
        return [{"epoch": int(payload["epoch"]), "train_loss": float(payload["train_loss"])}]
    if line.startswith(EPOCH_JSON_PREFIX):
        try:
            payload = json.loads(line[len(EPOCH_JSON_PREFIX):])
        except json.JSONDecodeError:
            return []
        event = {"epoch": int(payload["epoch"])}
        if payload.get("train_loss") is not None:
            event["train_loss"] = float(payload["train_loss"])
        if payload.get("eval_loss") is not None:
            event["eval_loss"] = float(payload["eval_loss"])
        return [event]
    stripped = line.strip()
    match = EVAL_TEXT_PATTERN.match(stripped)
    if match:
        return [{"epoch": int(match.group("epoch")), "eval_loss": float(match.group("loss"))}]
    match = TRAIN_TEXT_PATTERN.match(stripped)
    if match:
        return [{"epoch": int(match.group("epoch")), "train_loss": float(match.group("loss"))}]
    return []


def update_metric_record(record, event):
    changed = {"any": False, "train": False, "eval": False, "best_eval": False}
    epoch = int(event["epoch"])
    metrics = None
    for existing in record["metrics_history"]:
        if existing["epoch"] == epoch:
            metrics = existing
            break
    if metrics is None:
        metrics = {"epoch": epoch}
        record["metrics_history"].append(metrics)

    if "train_loss" in event:
        train_loss = float(event["train_loss"])
        if metrics.get("train_loss") != train_loss:
            metrics["train_loss"] = train_loss
            upsert_epoch_history(record["train_history"], epoch, "train_loss", train_loss)
            record["final_train_loss"] = train_loss
            changed["any"] = True
            changed["train"] = True
    if "eval_loss" in event:
        eval_loss = float(event["eval_loss"])
        if metrics.get("eval_loss") != eval_loss:
            metrics["eval_loss"] = eval_loss
            upsert_epoch_history(record["eval_history"], epoch, "eval_loss", eval_loss)
            record["final_eval_loss"] = eval_loss
            if record["best_eval_loss"] is None or eval_loss < record["best_eval_loss"]:
                record["best_eval_loss"] = eval_loss
                record["best_eval_epoch"] = epoch
                changed["best_eval"] = True
            changed["any"] = True
            changed["eval"] = True

    record["metrics_history"].sort(key=lambda item: item["epoch"])
    record["train_history"].sort(key=lambda item: item["epoch"])
    record["eval_history"].sort(key=lambda item: item["epoch"])
    return changed


def detect_overfit_stop(record, event, policy):
    if not policy.get("enabled", False):
        return None
    epoch = int(event["epoch"])
    if epoch < int(policy["min_epochs"]):
        return None
    best_eval_loss = record.get("best_eval_loss")
    best_eval_epoch = record.get("best_eval_epoch")
    if best_eval_loss is None or best_eval_epoch is None or epoch <= int(best_eval_epoch):
        return None
    current_eval_loss = float(event["eval_loss"])
    current_train_loss = metric_value_at_epoch(record, epoch, "train_loss")
    best_train_loss = metric_value_at_epoch(record, int(best_eval_epoch), "train_loss")
    if current_train_loss is None or best_train_loss is None:
        return None

    evals_since_best = sum(1 for item in record["eval_history"] if item["epoch"] > int(best_eval_epoch))
    if evals_since_best < int(policy["patience"]):
        return None
    eval_threshold = float(best_eval_loss) * (1.0 + float(policy["min_eval_regression_ratio"]))
    train_threshold = float(best_train_loss) * (1.0 - float(policy["min_train_improvement_ratio"]))
    if current_eval_loss < eval_threshold:
        return None
    if float(current_train_loss) > train_threshold:
        return None

    return {
        "trial_id": record["trial_id"],
        "reason": "eval_regressed_while_train_improved",
        "epoch": epoch,
        "eval_loss": current_eval_loss,
        "train_loss": float(current_train_loss),
        "best_eval_loss": float(best_eval_loss),
        "best_eval_epoch": int(best_eval_epoch),
        "best_train_loss": float(best_train_loss),
        "evals_since_best": int(evals_since_best),
        "min_epochs": int(policy["min_epochs"]),
        "patience": int(policy["patience"]),
        "min_eval_regression_ratio": float(policy["min_eval_regression_ratio"]),
        "min_train_improvement_ratio": float(policy["min_train_improvement_ratio"]),
    }


def metric_value_at_epoch(record, epoch, key):
    for metrics in record["metrics_history"]:
        if int(metrics["epoch"]) == int(epoch) and key in metrics:
            return float(metrics[key])
    return None


def apply_overfit_stop_record(record, overfit_stop):
    record["overfit_stop_reason"] = overfit_stop["reason"]
    record["overfit_stop_epoch"] = overfit_stop["epoch"]
    record["overfit_stop_metrics"] = overfit_stop


def format_overfit_stop_line(trial_id, overfit_stop):
    return (
        f"sweep_trial_overfit_stop={trial_id} "
        f"epoch={overfit_stop['epoch']} "
        f"eval_loss={overfit_stop['eval_loss']} "
        f"best_eval_loss={overfit_stop['best_eval_loss']} "
        f"best_eval_epoch={overfit_stop['best_eval_epoch']} "
        f"train_loss={overfit_stop['train_loss']} "
        f"best_train_loss={overfit_stop['best_train_loss']} "
        f"reason={overfit_stop['reason']}"
    )


def upsert_epoch_history(history, epoch, key, value):
    for existing in history:
        if existing["epoch"] == epoch:
            existing[key] = value
            return
    history.append({"epoch": epoch, key: value})


def write_metrics_jsonl(path, record):
    with Path(path).open("w", encoding="utf-8") as output_file:
        for metrics in record["metrics_history"]:
            output_file.write(json.dumps(metrics, sort_keys=True) + "\n")


def select_best_trial(records):
    completed = [
        record
        for record in records
        if record["status"] in {"completed", "overfit_stopped"} and record["best_eval_loss"] is not None
    ]
    if not completed:
        return None
    return min(completed, key=lambda record: record["best_eval_loss"])


def find_trial_checkpoint(trial_dir):
    candidates = [
        path
        for path in trial_dir.glob("checkpoint-epoch-*")
        if path.is_dir()
    ]
    if not candidates:
        return None
    return max(candidates, key=checkpoint_epoch_number)


def require_best_eval_checkpoint(record, trial_dir):
    checkpoint = Path(record["best_checkpoint_path"] or Path(trial_dir) / BEST_CHECKPOINT_NAME)
    if record.get("best_eval_loss") is None:
        raise RuntimeError(f"{record['trial_id']} did not produce eval_loss")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"{record['trial_id']} best eval checkpoint is missing: {checkpoint}")
    return checkpoint


def attach_checkpoint_and_sample(args, record, checkpoint):
    record["checkpoint_path"] = str(checkpoint)
    record["checkpoint_epoch"] = record["best_eval_epoch"]
    record["best_checkpoint_path"] = str(checkpoint)
    sample = generate_smoke_sample(args, record, Path(checkpoint))
    if sample is not None:
        record["sample"] = sample
        append_jsonl(Path(args.output_root) / "samples.jsonl", sample)


def checkpoint_epoch_number(path):
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def promote_best_checkpoint(best, target):
    source = Path(best.get("retained_best_checkpoint_path") or best["best_checkpoint_path"])
    if not source.exists():
        raise FileNotFoundError(f"best checkpoint does not exist: {source}")
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return str(target)


def delete_trial_checkpoints(output_root):
    deleted = []
    for trial_dir in sorted(Path(output_root).glob("trials/trial-*")):
        checkpoints = []
        best_checkpoint = trial_dir / BEST_CHECKPOINT_NAME
        if best_checkpoint.is_dir():
            checkpoints.append(best_checkpoint)
        checkpoints.extend(path for path in trial_dir.glob("checkpoint-epoch-*") if path.is_dir())
        for checkpoint in sorted(checkpoints):
            shutil.rmtree(checkpoint)
            deleted.append(str(checkpoint))
            print(f"deleted_trial_checkpoint={checkpoint}")
    return deleted


def delete_trial_checkpoints_in_trial_dir(trial_dir):
    deleted = []
    checkpoints = []
    trial_dir = Path(trial_dir)
    best_checkpoint = trial_dir / BEST_CHECKPOINT_NAME
    if best_checkpoint.is_dir():
        checkpoints.append(best_checkpoint)
    checkpoints.extend(path for path in trial_dir.glob("checkpoint-epoch-*") if path.is_dir())
    for checkpoint in sorted(checkpoints):
        shutil.rmtree(checkpoint)
        deleted.append(str(checkpoint))
        print(f"deleted_trial_checkpoint={checkpoint}")
    return deleted


def mark_deleted_checkpoints(records, deleted_checkpoints):
    deleted = set(deleted_checkpoints)
    for record in records:
        checkpoint_path = record.get("checkpoint_path")
        if checkpoint_path in deleted:
            record["checkpoint_path"] = None
            record["checkpoint_deleted"] = True
        best_checkpoint_path = record.get("best_checkpoint_path")
        if best_checkpoint_path in deleted:
            record["best_checkpoint_path"] = None
            record["best_checkpoint_deleted"] = True


def rewrite_trial_records(records, trials_jsonl):
    trials_jsonl = Path(trials_jsonl)
    trials_jsonl.write_text("", encoding="utf-8")
    for record in records:
        append_jsonl(trials_jsonl, record)
        trial_json = Path(record["output_model_path"]) / "trial.json"
        trial_json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def retain_best_checkpoint_candidate(record, current_best_record, output_root, *, dry_run=False):
    if dry_run:
        return current_best_record
    if record["status"] not in {"completed", "overfit_stopped"} or record["best_eval_loss"] is None:
        return current_best_record
    if current_best_record is not None and current_best_record["best_eval_loss"] <= record["best_eval_loss"]:
        return current_best_record

    source = Path(record["best_checkpoint_path"])
    if not source.is_dir():
        raise FileNotFoundError(f"{record['trial_id']} best checkpoint is missing before retention: {source}")

    staging_dir = Path(output_root) / RETAINED_BEST_CHECKPOINT_DIRNAME
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    shutil.copytree(source, staging_dir)
    record["retained_best_checkpoint_path"] = str(staging_dir)
    print(f"retained_best_checkpoint={staging_dir}")
    if current_best_record is not None:
        current_best_record["retained_best_checkpoint_path"] = None
    return record


def delete_retained_best_checkpoint(output_root):
    retained = Path(output_root) / RETAINED_BEST_CHECKPOINT_DIRNAME
    if retained.is_dir():
        shutil.rmtree(retained)
        print(f"deleted_retained_best_checkpoint={retained}")
        return str(retained)
    return None


def clear_retained_checkpoint_path(records, deleted_retained):
    for record in records:
        if record.get("retained_best_checkpoint_path") == deleted_retained:
            record["retained_best_checkpoint_path"] = None


def generate_smoke_sample(args, record, checkpoint):
    if not args.smoke_text:
        return None

    sample_dir = Path(args.smoke_output_dir) if args.smoke_output_dir else Path(args.output_root) / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / sample_filename(record)
    sample_log_path = sample_path.with_suffix(".log")
    metadata_path = sample_path.with_suffix(".json")
    command = build_smoke_command(args, checkpoint, sample_path)

    start_time = time.monotonic()
    with sample_log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parent),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"smoke sample failed for {record['trial_id']} with exit code {returncode}; see {sample_log_path}")
    if not sample_path.exists() or sample_path.stat().st_size == 0:
        raise RuntimeError(f"smoke sample is missing or empty: {sample_path}")

    metadata = {
        "trial_id": record["trial_id"],
        "epoch": record["checkpoint_epoch"],
        "checkpoint_path": str(checkpoint),
        "sample_path": str(sample_path),
        "sample_log_path": str(sample_log_path),
        "sample_metadata_path": str(metadata_path),
        "text": args.smoke_text,
        "language": args.smoke_language,
        "instruct": args.smoke_instruct,
        "params": record["params"],
        "best_eval_loss": record["best_eval_loss"],
        "best_eval_epoch": record["best_eval_epoch"],
        "final_train_loss": record["final_train_loss"],
        "final_eval_loss": record["final_eval_loss"],
        "duration_sec": round(time.monotonic() - start_time, 3),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"smoke_sample={sample_path}")
    print(f"smoke_sample_metadata={metadata_path}")
    return metadata


def sample_filename(record):
    params = record["params"]
    parts = [
        f"b{params['batch_size']}",
        f"lr{format_sample_param(params['lr'])}",
        f"norm{format_sample_param(params['grad_clip_max_norm'])}",
        f"l{format_sample_param(params['weight_decay'])}",
        record["trial_id"].replace("-", ""),
        f"epoch{record['checkpoint_epoch']}",
    ]
    return "_".join(parts) + ".wav"


def format_trial_params(params):
    ordered = [
        ("batch_size", params["batch_size"]),
        ("gradient_accumulation_steps", params["gradient_accumulation_steps"]),
        ("lr", params["lr"]),
        ("lr_scheduler", params["lr_scheduler"]),
        ("lr_warmup_steps", params["lr_warmup_steps"]),
        ("lr_warmup_ratio", params["lr_warmup_ratio"]),
        ("lr_min_ratio", params["lr_min_ratio"]),
        ("weight_decay", params["weight_decay"]),
        ("grad_clip_max_norm", params["grad_clip_max_norm"]),
        ("num_epochs", params["num_epochs"]),
        ("eval_batch_size", params["eval_batch_size"]),
        ("max_eval_batches", params["max_eval_batches"]),
    ]
    return " ".join(f"{name}={format_sample_param(value)}" for name, value in ordered)


def format_sample_param(value):
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    value = float(value)
    if value.is_integer():
        return str(int(value))
    text = str(value)
    if "e" in text:
        return text
    if 0 < abs(value) < 0.1:
        mantissa, exponent = f"{value:.12e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        return f"{mantissa}e{int(exponent)}"
    return text


def build_smoke_command(args, checkpoint, sample_path):
    command = [
        sys.executable,
        "-u",
        "-c",
        SMOKE_SAMPLE_SCRIPT,
        "--checkpoint_path",
        str(checkpoint),
        "--output_path",
        str(sample_path),
        "--text",
        args.smoke_text,
        "--speaker",
        str(args.speaker_name),
        "--language",
        str(args.smoke_language),
        "--dtype",
        str(args.smoke_dtype),
        "--attn_implementation",
        str(args.smoke_attn_implementation),
    ]
    if args.smoke_instruct:
        command.extend(["--instruct", args.smoke_instruct])
    return command


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
