"""Train the small BooleLOT period checkpoint from scratch on CPU or CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory

from gpt_oss.vintage.period import (CHECKPOINT_FILES, CODE_TAG, EOS, PAD, PASSAGE_TAG,
                                    SOURCE_PATH, check_program, load_catalogue, program_prefix, sha256)
from training.period_curriculum import PROGRAMS


def curriculum(catalogue):
    if set(catalogue["programs"]) != set(PROGRAMS):
        raise ValueError("Every program needs a reviewed source and an implementation.")
    code_examples = []
    for key, source in PROGRAMS.items():
        check_program(source, key)
        code_examples.append((program_prefix(key), source))
    passages = []
    for row in catalogue["passages"]:
        if row["split"] == "train":
            passages.append((f"{PASSAGE_TAG}{row['id']}\n", row["text"]))
    return code_examples, passages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=650)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1:
        parser.error("--steps and --threads must be positive")
    if args.output.exists():
        parser.error("--output already exists")

    import torch
    from safetensors.torch import save_model
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from gpt_oss.vintage.period_model import PeriodModel

    torch.set_num_threads(args.threads)
    torch.manual_seed(1898)
    sampler = random.Random(1898)
    catalogue = load_catalogue()
    code_examples, passages = curriculum(catalogue)
    examples = code_examples + passages

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        (text for pair in examples for text in pair),
        trainers.BpeTrainer(vocab_size=768, min_frequency=2,
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                            special_tokens=[PAD, EOS, CODE_TAG, PASSAGE_TAG]),
    )
    config = dict(vocab_size=tokenizer.get_vocab_size(), context=256, width=96, heads=4, layers=3)
    model = PeriodModel(**config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    pad, eos = tokenizer.token_to_id(PAD), tokenizer.token_to_id(EOS)

    def encode(example):
        prompt, answer = example
        prefix = tokenizer.encode(prompt).ids
        output = tokenizer.encode(answer).ids + [eos]
        inputs = prefix + output[:-1]
        labels = [-100] * (len(prefix) - 1) + output
        if len(inputs) > model.context:
            raise ValueError(f"Training example exceeds context: {prompt}")
        return inputs, labels

    code_rows, passage_rows = ([encode(pair) for pair in rows] for rows in (code_examples, passages))

    def metrics(rows, *, optimize=False):
        width = max(len(inputs) for inputs, _ in rows)
        inputs = torch.tensor([x + [pad] * (width - len(x)) for x, _ in rows], device=args.device)
        answers = torch.tensor([y + [-100] * (width - len(y)) for _, y in rows], device=args.device)
        logits = model(inputs)
        # Average within each sample so UTF-8 passages do not outweigh programs.
        token_losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), answers, reduction="none")
        loss = (token_losses.sum(dim=1) / (answers != -100).sum(dim=1)).mean()
        if optimize:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        return round(float(loss.detach()), 6)

    with torch.no_grad():
        start_loss = {"code": metrics(code_rows), "period_prose": metrics(passage_rows)}
    for step in range(1, args.steps + 1):
        loss = metrics(code_rows + sampler.sample(passage_rows, 2), optimize=True)
        if step == 1 or step % 50 == 0 or step == args.steps:
            print(f"step {step:4d}/{args.steps}: training loss {loss:.5f}", flush=True)
    model.eval()
    with torch.no_grad():
        final_loss = {"code": metrics(code_rows), "period_prose": metrics(passage_rows)}
    print(json.dumps({"initial_loss": start_loss, "training_loss": final_loss}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        checkpoint = Path(temp) / "checkpoint"
        checkpoint.mkdir()
        save_model(model.to("cpu"), checkpoint / "model.safetensors")
        tokenizer.save(str(checkpoint / "tokenizer.json"))
        (checkpoint / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "cutoff_year": catalogue["cutoff_year"],
            "catalogue_sha256": sha256(SOURCE_PATH),
            "initialized_from": "random",
            "seed": 1898,
            "steps": args.steps,
            "device": args.device,
            "pytorch_version": torch.__version__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "code_examples": len(code_examples),
            "code_training_sha256": {key: hashlib.sha256(code.encode()).hexdigest() for key, code in PROGRAMS.items()},
            "prose_examples": len(passages),
            "prose_training_ids": [row["id"] for row in catalogue["passages"] if row["split"] == "train"],
            "initial_loss": start_loss,
            "training_loss": final_loss,
            "artifacts": {filename: sha256(checkpoint / filename) for filename in CHECKPOINT_FILES},
        }
        (checkpoint / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        checkpoint.rename(args.output)


if __name__ == "__main__":
    main()
