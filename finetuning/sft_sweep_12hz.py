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
    "num_epochs",
    "eval_batch_size",
    "max_eval_batches",
}
EVAL_JSON_PREFIX = "SFT_EVAL_JSON "
EVAL_TEXT_PATTERN = re.compile(r"^Epoch\s+(?P<epoch>\d+)\s+\|\s+Eval Loss:\s+(?P<loss>[0-9.eE+-]+)")


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
    trials_jsonl.write_text("", encoding="utf-8")

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

    best_path = output_root / "best_trial.json"
    best_path.write_text(json.dumps(best, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"best_trial_json={best_path}")
    print(f"best_trial_id={best['trial_id']}")
    print(f"best_eval_loss={best['best_eval_loss']}")
    if args.promote_best_to and not args.dry_run:
        promoted = promote_best_checkpoint(best, Path(args.promote_best_to))
        print(f"promoted_checkpoint={promoted}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen3-TTS SFT hyperparameter sweep trials.")
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--eval_jsonl", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip_max_norm", type=float, default=2.0)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--checkpoint_interval_epochs", type=int, default=1)
    parser.add_argument("--keep_last_checkpoints", type=int, default=3)
    parser.add_argument("--eval_interval_epochs", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--sweep_config_json", type=str, default=None)
    parser.add_argument("--sweep_config_path", type=str, default=None)
    parser.add_argument("--promote_best_to", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if bool(args.sweep_config_json) == bool(args.sweep_config_path):
        raise ValueError("pass exactly one of --sweep_config_json or --sweep_config_path")
    if not args.dry_run and not args.eval_jsonl:
        raise ValueError("eval_jsonl is required unless --dry_run is set")
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
    params = config.get("params")
    if not isinstance(params, dict) or not params:
        raise ValueError("sweep config requires a non-empty params object")
    unknown = sorted(set(params) - ALLOWED_SWEEP_PARAMS)
    if unknown:
        raise ValueError(f"unsupported sweep params: {unknown}")
    if "max_trials" in config and (not isinstance(config["max_trials"], int) or config["max_trials"] <= 0):
        raise ValueError("max_trials must be a positive integer")
    if method == "grid":
        for name, values in params.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"grid param {name} must be a non-empty list")
            for value in values:
                validate_single_param(name, value)
    else:
        for name, spec in params.items():
            validate_optuna_param(name, spec)


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
    int_positive = ["batch_size", "num_epochs", "checkpoint_interval_epochs", "keep_last_checkpoints"]
    int_nonnegative = ["eval_batch_size", "max_eval_batches"]
    float_positive = ["lr"]
    float_nonnegative = ["weight_decay", "grad_clip_max_norm"]
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


def validate_single_param(name, value):
    if name in {"batch_size", "num_epochs", "eval_batch_size", "max_eval_batches"}:
        if not isinstance(value, int):
            raise ValueError(f"{name} values must be integers")
    else:
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} values must be numeric")
    validate_training_args({**base_param_defaults(), name: value})


def base_param_defaults():
    return {
        "batch_size": 2,
        "lr": 2e-5,
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
    trial_params = list(iter_grid_params(config["params"]))
    max_trials = config.get("max_trials")
    if max_trials is not None:
        trial_params = trial_params[:max_trials]
    for index, params in enumerate(trial_params):
        record = run_trial(args, index, params, dry_run=args.dry_run)
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

    def objective(trial):
        params = suggest_trial_params(trial, config["params"])
        record = run_trial(args, trial.number, params, optuna_trial=trial, dry_run=args.dry_run)
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


def run_trial(args, index, params, *, optuna_trial=None, dry_run=False):
    trial_id = f"trial-{index:04d}"
    output_root = Path(args.output_root)
    trial_dir = output_root / "trials" / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_path = trial_dir / "sft_12hz.log"
    merged_params = build_trial_params(args, params)
    validate_training_args({**base_param_defaults(), **merged_params})
    command = build_sft_command(args, trial_dir, merged_params)
    record = {
        "trial_id": trial_id,
        "status": "dry_run" if dry_run else "running",
        "params": merged_params,
        "command": command,
        "output_model_path": str(trial_dir),
        "log_path": str(log_path),
        "eval_history": [],
        "best_eval_loss": None,
        "best_eval_epoch": None,
        "best_checkpoint_path": str(trial_dir / "checkpoint-best"),
        "returncode": None,
        "duration_sec": 0.0,
    }
    (trial_dir / "trial.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"sweep_trial={trial_id}")
    print("sweep_command=" + " ".join(command))
    if dry_run:
        return record

    start_time = time.monotonic()
    pruned = False
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
            event = parse_eval_event(line)
            if event:
                updated = update_eval_record(record, event)
                if updated and optuna_trial is not None:
                    optuna_trial.report(event["eval_loss"], step=event["epoch"])
                    if optuna_trial.should_prune():
                        pruned = True
                        print(f"sweep_trial_pruned={trial_id} epoch={event['epoch']} eval_loss={event['eval_loss']}")
                        process.terminate()
                        break
        if pruned:
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
    elif returncode == 0:
        record["status"] = "completed"
        checkpoint = find_trial_checkpoint(trial_dir)
        if checkpoint:
            record["best_checkpoint_path"] = str(checkpoint)
    else:
        record["status"] = "failed"
    (trial_dir / "trial.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if record["status"] == "failed":
        raise RuntimeError(f"{trial_id} failed with exit code {returncode}; see {log_path}")
    return record


def build_trial_params(args, overrides):
    params = {
        "batch_size": args.batch_size,
        "lr": args.lr,
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
        "--lr",
        str(params["lr"]),
        "--weight_decay",
        str(params["weight_decay"]),
        "--grad_clip_max_norm",
        str(params["grad_clip_max_norm"]),
        "--num_epochs",
        str(params["num_epochs"]),
        "--checkpoint_interval_epochs",
        str(args.checkpoint_interval_epochs),
        "--keep_last_checkpoints",
        str(args.keep_last_checkpoints),
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
            "checkpoint-best",
        ])
    return command


def parse_eval_event(line):
    if line.startswith(EVAL_JSON_PREFIX):
        try:
            payload = json.loads(line[len(EVAL_JSON_PREFIX):])
        except json.JSONDecodeError:
            return None
        return {"epoch": int(payload["epoch"]), "eval_loss": float(payload["eval_loss"])}
    match = EVAL_TEXT_PATTERN.match(line.strip())
    if not match:
        return None
    return {"epoch": int(match.group("epoch")), "eval_loss": float(match.group("loss"))}


def update_eval_record(record, event):
    for existing in record["eval_history"]:
        if existing["epoch"] == event["epoch"] and existing["eval_loss"] == event["eval_loss"]:
            return False
    record["eval_history"].append(event)
    if record["best_eval_loss"] is None or event["eval_loss"] < record["best_eval_loss"]:
        record["best_eval_loss"] = event["eval_loss"]
        record["best_eval_epoch"] = event["epoch"]
    return True


def select_best_trial(records):
    completed = [
        record
        for record in records
        if record["status"] == "completed" and record["best_eval_loss"] is not None
    ]
    if not completed:
        return None
    return min(completed, key=lambda record: record["best_eval_loss"])


def find_trial_checkpoint(trial_dir):
    best_checkpoint = trial_dir / "checkpoint-best"
    if best_checkpoint.exists():
        return best_checkpoint
    candidates = [
        path
        for path in trial_dir.glob("checkpoint-epoch-*")
        if path.is_dir()
    ]
    if not candidates:
        return None
    return max(candidates, key=checkpoint_epoch_number)


def checkpoint_epoch_number(path):
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def promote_best_checkpoint(best, target):
    source = Path(best["best_checkpoint_path"])
    if not source.exists():
        raise FileNotFoundError(f"best checkpoint does not exist: {source}")
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return str(target)


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
