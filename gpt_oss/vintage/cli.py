"""Terminal client for BooleLOT and its vintage model profiles."""
import argparse
from pathlib import Path

from .engine import VintageEngine
from .period import PeriodAssistant
from .registry import MODEL_REGISTRY


def main():
    choices = sorted(key for key, value in MODEL_REGISTRY.items() if value.task == "generation")
    parser = argparse.ArgumentParser(description="BooleLOT vintage model playground")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--model", choices=choices, default="vintage-llm-340m")
    backend.add_argument("--period-checkpoint", type=Path, help="Directory of the trained pre-1899 checkpoint")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--ask", help="Answer one question and exit")
    args = parser.parse_args()
    if args.period_checkpoint:
        assistant = PeriodAssistant(args.period_checkpoint)
        answer = assistant.answer
        name = "BooleLOT, sources through 1898"
    else:
        engine = VintageEngine(args.model, max_new_tokens=args.max_tokens, temperature=args.temperature)
        system = (
            "You are BooleLOT, a historically grounded assistant. Use period-appropriate diction "
            "without parody. Distinguish documented facts from conjecture. Do not invent quotations "
            "or sources. Never claim that prompting removes knowledge from model weights."
        )

        def answer(prompt):
            return engine.generate(prompt, system=system).text

        name = engine.profile.model_id
    if args.ask:
        print(answer(args.ask))
        return
    print(f"BooleLOT Vintage — {name} (type /quit to exit)")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"/quit", "/exit"}:
            break
        if not prompt:
            continue
        try:
            print("\nBooleLOT:", answer(prompt))
        except Exception as exc:
            print(f"\nInference error: {exc}")


if __name__ == "__main__":
    main()
