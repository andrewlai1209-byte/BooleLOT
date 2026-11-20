"""Closed-corpus BooleLOT profile for sources published before 1899."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from gpt_oss.chronology import ChronologyGuard

CUTOFF_YEAR = 1898
SOURCE_PATH = Path(__file__).resolve().parents[1] / "data" / "period_sources.json"
CODE_TAG = "<|python|>"
PASSAGE_TAG = "<|passage|>"
EOS = "<|end|>"
PAD = "<|pad|>"
CHECKPOINT_FILES = ("model.safetensors", "tokenizer.json", "config.json")
LANGUAGE_MARKER = re.compile(r"\b(?:python|code|program|function|implement)\b|程式|函式|函數|實作|寫出", re.I)
OTHER_LANGUAGE = re.compile(r"\b(?:javascript|typescript|rust|java|sql|html|bash|ruby|golang)\b|c\+\+|c#", re.I)
CHINESE = re.compile(r"[\u3400-\u9fff]")
PROGRAM_BUILTINS = frozenset({"abs", "any", "len", "min", "range", "set", "sorted", "sum", "ValueError"})
PROGRAM_ERRORS = frozenset({
    "n must be nonnegative", "n must be positive", "exponent must be nonnegative",
    "at least one value is required", "incompatible matrices", "expected a two-by-two matrix",
})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_catalogue(path: Path = SOURCE_PATH) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["cutoff_year"] != CUTOFF_YEAR:
        raise ValueError("Expected a source cutoff of 1898.")
    sources = data["sources"]
    for source_id, source in sources.items():
        year = source["year"]
        if type(year) is not int or not 1 <= year <= CUTOFF_YEAR:
            raise ValueError(f"Source {source_id} is outside the period.")
        url = urlsplit(source["url"])
        if url.scheme != "https" or not url.hostname or not all(source.get(k) for k in ("title", "location")):
            raise ValueError(f"Source {source_id} has incomplete citation metadata.")
    record_ids = set()
    source_splits = {}
    for passage in data["passages"]:
        source_id, split = passage["source"], passage["split"]
        if source_id not in sources or split not in ("train", "eval"):
            raise ValueError("Unknown passage source or split.")
        if source_splits.setdefault(source_id, split) != split:
            raise ValueError(f"Source {source_id} crosses the train/eval boundary.")
        key = passage["id"]
        text = passage["text"]
        if key in record_ids or not text.strip() or not passage["locator"] or not passage["aliases"]:
            raise ValueError(f"Passage {key} is duplicate or incomplete.")
        record_ids.add(key)
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != passage["sha256"]:
            raise ValueError(f"Passage hash mismatch for {key}.")
    for key, program in data["programs"].items():
        if not key.isidentifier() or program["source"] not in sources or not program["aliases"]:
            raise ValueError(f"Program {key} has incomplete source metadata.")
    return data


def program_prefix(key: str) -> str:
    return f"{CODE_TAG}{key}\n"


def has_alias(question: str, alias: str) -> bool:
    if CHINESE.search(alias):
        return alias in question
    return bool(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", question))


def check_program(code: str, name: str) -> None:
    """Validate the restricted subset used by the arithmetic examples."""
    allowed = (
        ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Assign, ast.AugAssign,
        ast.While, ast.For, ast.If, ast.Return, ast.Raise, ast.BoolOp, ast.BinOp,
        ast.UnaryOp, ast.Compare, ast.Call, ast.List, ast.Tuple, ast.Subscript,
        ast.ListComp, ast.GeneratorExp, ast.comprehension, ast.Name, ast.Load, ast.Store, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.BitAnd, ast.BitOr,
        ast.Or, ast.And, ast.Not, ast.USub, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    )
    tree = ast.parse(code, mode="exec")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Expected one function definition.")
    function = tree.body[0]
    arguments = function.args
    if (function.name != name or function.decorator_list or function.returns or function.type_params
            or arguments.defaults or arguments.kwonlyargs or arguments.vararg or arguments.kwarg
            or any(arg.annotation or arg.arg in PROGRAM_BUILTINS for arg in arguments.args + arguments.posonlyargs)):
        raise ValueError("Unexpected function signature.")
    for node in ast.walk(tree):
        if not isinstance(node, allowed) or (isinstance(node, ast.FunctionDef) and node is not function):
            raise ValueError(f"Unexpected Python syntax: {type(node).__name__}.")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in PROGRAM_BUILTINS):
            raise ValueError("Unexpected function call.")
        if isinstance(node, ast.Name) and (node.id.startswith("__")
                or (isinstance(node.ctx, ast.Store) and node.id in PROGRAM_BUILTINS)):
            raise ValueError("Unexpected identifier.")
        if isinstance(node, ast.Constant) and (type(node.value) not in (str, int, bool)
                or (isinstance(node.value, str) and node.value not in PROGRAM_ERRORS)):
            raise ValueError("Unexpected constant.")
    if "#" in code or "`" in code:
        raise ValueError("Unexpected prose or formatting in generated code.")
    compile(tree, "<period-program>", "exec")


class PeriodAssistant:
    def __init__(self, checkpoint: Path):
        from safetensors.torch import load_model
        from tokenizers import Tokenizer

        from .period_model import PeriodModel

        self.catalogue = load_catalogue()
        checkpoint = Path(checkpoint)
        manifest = json.loads((checkpoint / "training_manifest.json").read_text(encoding="utf-8"))
        if (manifest["cutoff_year"] != CUTOFF_YEAR or manifest["catalogue_sha256"] != sha256(SOURCE_PATH)
                or manifest["initialized_from"] != "random"):
            raise ValueError("The checkpoint does not match this period catalogue.")
        for filename in CHECKPOINT_FILES:
            if manifest["artifacts"][filename] != sha256(checkpoint / filename):
                raise ValueError(f"Checkpoint hash mismatch for {filename}.")
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        self.tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
        self.model = PeriodModel(**config).eval()
        load_model(self.model, checkpoint / "model.safetensors")
        self.guard = ChronologyGuard(CUTOFF_YEAR)

    def generate_program(self, key: str) -> str | None:
        if key not in self.catalogue["programs"]:
            raise ValueError(f"Unknown program: {key}.")
        prefix = self.tokenizer.encode(program_prefix(key)).ids
        tokens = self.model.complete(prefix, self.tokenizer.token_to_id(EOS))
        if not tokens:
            return None
        code = self.tokenizer.decode(tokens, skip_special_tokens=False)
        try:
            check_program(code, key)
        except (SyntaxError, ValueError):
            return None
        return code.rstrip()

    def answer(self, question: str) -> str:
        zh = bool(CHINESE.search(question))
        uncertainty = "此事未有書證，姑俟博考。" if zh else "On that point I have no authority at hand and must reserve judgment."
        programming = bool(LANGUAGE_MARKER.search(question))
        if self.guard.reasons(question, programming=programming):
            return "此事出所考年界，未敢妄言。" if zh else "That matter lies beyond the date of my authorities. I cannot speak to it."

        normalized = re.sub(r"\s+", " ", question.casefold()).strip()
        if programming:
            matches = [key for key, entry in self.catalogue["programs"].items()
                       if any(has_alias(normalized, alias) for alias in entry["aliases"])]
            if OTHER_LANGUAGE.search(question) or len(matches) != 1:
                return ("今所習者，乃算術與集合之數例，止以 Python 行之。" if zh else
                        "My present exercises are confined to certain operations in arithmetic and on sets, expressed in Python.")
            key = matches[0]
            code = self.generate_program(key)
            if code is None:
                return "此式未成，請容再考。" if zh else "The computation is wanting in form, so I must withhold it."
            program = self.catalogue["programs"][key]
            preface = f"其法如下。{program['zh']}。" if zh else f"The operation may be set down thus. {program['en']}."
            introduction = "算理所據" if zh else "Mathematical authority"
            return f"{preface}\n\n```python\n{code}\n```\n\n{introduction} · {self.cite(program['source'])}"

        candidates = []
        for passage in self.catalogue["passages"]:
            lengths = [len(alias) for alias in passage["aliases"] if has_alias(normalized, alias)]
            if lengths:
                candidates.append((max(lengths), passage))
        if not candidates:
            return uncertainty
        score = max(value for value, _ in candidates)
        best = [passage for value, passage in candidates if value == score]
        if len(best) != 1:
            return uncertainty
        passage = best[0]
        preface = "謹按原文，有言如下。" if zh else "I beg leave to adduce the passage itself."
        citation = self.cite(passage["source"], passage["locator"])
        return f"{preface}\n\n> {passage['text']}\n\n{citation}"

    def cite(self, source_id: str, locator: str = "") -> str:
        source = self.catalogue["sources"][source_id]
        date = source["year"]
        location = " · ".join(filter(None, (source["location"], locator)))
        return f"[{source['title']}]({source['url']}) · {location} · {date}"
