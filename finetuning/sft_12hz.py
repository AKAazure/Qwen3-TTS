# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

target_speaker_embedding = None


def load_jsonl(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def compute_finetune_loss(model, batch, *, capture_target_speaker=False):
    global target_speaker_embedding

    input_ids = batch['input_ids']
    codec_ids = batch['codec_ids']
    ref_mels = batch['ref_mels']
    text_embedding_mask = batch['text_embedding_mask']
    codec_embedding_mask = batch['codec_embedding_mask']
    attention_mask = batch['attention_mask']
    codec_0_labels = batch['codec_0_labels']
    codec_mask = batch['codec_mask']
    speaker_embedding_mask = batch.get('speaker_embedding_mask')

    speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
    if capture_target_speaker and target_speaker_embedding is None:
        target_speaker_embedding = speaker_embedding

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]

    input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
    input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    _place_speaker_embedding(input_codec_embedding, speaker_embedding, speaker_embedding_mask)

    input_embeddings = input_text_embedding + input_codec_embedding

    for i in range(1, 16):
        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
        input_embeddings = input_embeddings + codec_i_embedding

    outputs = model.talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=codec_0_labels[:, 1:],
        output_hidden_states=True
    )

    hidden_states = outputs.hidden_states[0][-1]
    talker_hidden_states = hidden_states[codec_mask[:, :-1]]
    talker_codec_ids = codec_ids[codec_mask]

    _, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
    return outputs.loss + 0.3 * sub_talker_loss


def _place_speaker_embedding(input_codec_embedding, speaker_embedding, speaker_embedding_mask=None):
    """Place each sample's speaker embedding at its dataset-defined speaker slot."""
    if speaker_embedding_mask is None:
        input_codec_embedding[:, 6, :] = speaker_embedding
        return

    for row_index in range(input_codec_embedding.shape[0]):
        positions = torch.nonzero(speaker_embedding_mask[row_index], as_tuple=False).flatten()
        if positions.numel() != 1:
            raise ValueError("speaker_embedding_mask must contain exactly one True value per sample")
        input_codec_embedding[row_index, positions.item(), :] = speaker_embedding[row_index]


def evaluate(model, eval_dataloader, accelerator, *, max_eval_batches=0):
    model.eval()
    eval_losses = []
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            if max_eval_batches > 0 and step >= max_eval_batches:
                break
            loss = compute_finetune_loss(model, batch)
            sample_count = batch['input_ids'].shape[0]
            gathered_losses = accelerator.gather_for_metrics(loss.detach().float().repeat(sample_count))
            eval_losses.append(gathered_losses.cpu())
    model.train()
    if not eval_losses:
        return None
    return torch.cat(eval_losses).mean().item()


def save_checkpoint(model, accelerator, *, model_path, output_dir, speaker_name, overwrite=False):
    if overwrite and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(model_path, output_dir, dirs_exist_ok=True)

    input_config_file = os.path.join(model_path, "config.json")
    output_config_file = os.path.join(output_dir, "config.json")
    with open(input_config_file, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    config_dict["tts_model_type"] = "custom_voice"
    talker_config = config_dict.get("talker_config", {})
    talker_config["spk_id"] = {
        speaker_name: 3000
    }
    talker_config["spk_is_dialect"] = {
        speaker_name: False
    }
    config_dict["talker_config"] = talker_config

    with open(output_config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

    drop_prefix = "speaker_encoder"
    keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
    for k in keys_to_drop:
        del state_dict[k]

    weight = state_dict['talker.model.codec_embedding.weight']
    state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
    save_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, save_path)


def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip_max_norm", type=float, default=2.0)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--checkpoint_interval_epochs", type=int, default=1)
    parser.add_argument("--keep_last_checkpoints", type=int, default=3)
    parser.add_argument("--eval_jsonl", type=str, default=None)
    parser.add_argument("--eval_interval_epochs", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--save_best_eval_checkpoint", action="store_true")
    parser.add_argument("--best_checkpoint_name", type=str, default="checkpoint-best")
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    args = parser.parse_args()
    if args.checkpoint_interval_epochs <= 0:
        raise ValueError("checkpoint_interval_epochs must be positive")
    if args.keep_last_checkpoints <= 0:
        raise ValueError("keep_last_checkpoints must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if args.grad_clip_max_norm < 0:
        raise ValueError("grad_clip_max_norm must be non-negative")
    if args.eval_interval_epochs < 0:
        raise ValueError("eval_interval_epochs must be non-negative")
    if args.eval_batch_size < 0:
        raise ValueError("eval_batch_size must be non-negative")
    if args.max_eval_batches < 0:
        raise ValueError("max_eval_batches must be non-negative")
    if args.save_best_eval_checkpoint and not args.eval_jsonl:
        raise ValueError("save_best_eval_checkpoint requires eval_jsonl")
    if not args.best_checkpoint_name.strip():
        raise ValueError("best_checkpoint_name must not be empty")
    if os.path.basename(args.best_checkpoint_name) != args.best_checkpoint_name:
        raise ValueError("best_checkpoint_name must be a single directory name")

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16")

    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = load_jsonl(args.train_jsonl)
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    eval_dataloader = None
    if args.eval_jsonl:
        eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else args.batch_size
        eval_data = load_jsonl(args.eval_jsonl)
        if not eval_data:
            raise ValueError(f"eval_jsonl has no rows: {args.eval_jsonl}")
        eval_dataset = TTSDataset(eval_data, qwen3tts.processor, config)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=eval_dataset.collate_fn,
        )

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if eval_dataloader is None:
        model, optimizer, train_dataloader = accelerator.prepare(
            qwen3tts.model, optimizer, train_dataloader
        )
    else:
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            qwen3tts.model, optimizer, train_dataloader, eval_dataloader
        )

    num_epochs = args.num_epochs
    best_eval_loss = None
    model.train()

    for epoch in range(num_epochs):
        epoch_losses = []
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                loss = compute_finetune_loss(model, batch, capture_target_speaker=True)
                epoch_losses.append(float(loss.detach().float().item()))

                accelerator.backward(loss)

                if accelerator.sync_gradients and args.grad_clip_max_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip_max_norm)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else None
        if train_loss is not None:
            accelerator.print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f}")
            accelerator.print("SFT_TRAIN_JSON " + json.dumps(
                {"epoch": epoch + 1, "train_loss": train_loss},
                sort_keys=True,
            ))

        eval_loss = None
        should_evaluate = (
            eval_dataloader is not None
            and args.eval_interval_epochs > 0
            and (((epoch + 1) % args.eval_interval_epochs == 0) or (epoch + 1 == num_epochs))
        )
        if should_evaluate:
            eval_loss = evaluate(model, eval_dataloader, accelerator, max_eval_batches=args.max_eval_batches)
            if eval_loss is not None:
                accelerator.print(f"Epoch {epoch + 1} | Eval Loss: {eval_loss:.4f}")
                accelerator.print("SFT_EVAL_JSON " + json.dumps(
                    {"epoch": epoch + 1, "eval_loss": eval_loss},
                    sort_keys=True,
                ))
                if best_eval_loss is None or eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    if accelerator.is_main_process and args.save_best_eval_checkpoint:
                        best_output_dir = os.path.join(args.output_model_path, args.best_checkpoint_name)
                        save_checkpoint(
                            model,
                            accelerator,
                            model_path=MODEL_PATH,
                            output_dir=best_output_dir,
                            speaker_name=args.speaker_name,
                            overwrite=True,
                        )
                        print(f"Saved best checkpoint: {best_output_dir} eval_loss={eval_loss:.4f} epoch={epoch + 1}")

        if train_loss is not None:
            accelerator.print("SFT_EPOCH_JSON " + json.dumps(
                {"epoch": epoch + 1, "train_loss": train_loss, "eval_loss": eval_loss},
                sort_keys=True,
            ))

        should_save_checkpoint = ((epoch + 1) % args.checkpoint_interval_epochs == 0) or (epoch + 1 == num_epochs)
        if accelerator.is_main_process and should_save_checkpoint:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch + 1}")
            save_checkpoint(
                model,
                accelerator,
                model_path=MODEL_PATH,
                output_dir=output_dir,
                speaker_name=args.speaker_name,
            )
            print(f"Saved checkpoint: {output_dir}")
            checkpoint_dirs = [
                os.path.join(args.output_model_path, name)
                for name in os.listdir(args.output_model_path)
                if name.startswith("checkpoint-epoch-")
                and os.path.isdir(os.path.join(args.output_model_path, name))
            ]
            checkpoint_dirs.sort(key=lambda path: os.path.getmtime(path))
            for checkpoint_to_delete in checkpoint_dirs[:-args.keep_last_checkpoints]:
                shutil.rmtree(checkpoint_to_delete)
                print(f"Deleted old checkpoint: {checkpoint_to_delete}")

if __name__ == "__main__":
    train()
