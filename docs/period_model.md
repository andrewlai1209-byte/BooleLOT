# BooleLOT period checkpoint

This experimental profile uses a small decoder and tokenizer initialized and trained locally. The historical cutoff is **December 31, 1898**. Its only factual references are the dated works and passages in the [source catalogue](../gpt_oss/data/period_sources.json). The archive links point to digital copies or transcriptions of those works. The training run uses the selected transcriptions, with their modern introductions and annotations excluded. Ten passages supply prose from four authors. Two other authors provide passages reserved from training.

Historical answers quote the reviewed passage and give its location. Replies use Traditional Chinese literary phrasing or English phrasing suitable to the period. The historical lookup uses a fixed local catalogue and abstains when it finds no matching source. The code model receives the selected exercise name. It has no live web access or user-provided retrieval sources.

The coding training set contains sixteen newly written Python examples expressing period mathematics. The source citations identify the mathematical ideas. Python is the output notation. The model supports these exercises in particular.

- Greatest common divisor, least common multiple, positive divisors, prime sieve, and perfect numbers
- Intersection, union, and complement relative to a given set
- Fibonacci numbers, binomial coefficients, rows of Pascal’s triangle, triangular numbers, and nonnegative integer powers
- Median, matrix multiplication, and the determinant of a two-by-two matrix

Evaluation asks for the supported exercises with new wording and runs the emitted functions on held-out inputs. It also checks attributed quotations, including from authors reserved from training, and abstentions on unsupported questions. These checks measure the listed repertoire. Broader historical coverage, free-form historical prose, and new programming tasks remain unverified.

## Train and run

Use Python 3.12 or newer in the repository checkout. Install the optional dependencies, or install a compatible CPU build of PyTorch first to keep the install small on machines without a GPU.

```bash
python -m pip install -e '.[period]'
python -m training.train_period --output data/classical/boolelot-period
python -m training.evaluate_period data/classical/boolelot-period \
  --report data/classical/boolelot-period/evaluation.json
```

Unpack a supplied checkpoint or point to the directory produced by training. The loader verifies the catalogue, tokenizer, configuration and weights against the training manifest.

```bash
python -m gpt_oss.vintage.cli --period-checkpoint data/classical/boolelot-period \
  --ask '請以 Python 求二數之最大公因數。'
python -m gpt_oss.vintage.cli --period-checkpoint data/classical/boolelot-period \
  --ask '嚴復所言譯事三難，何所指？'
```

To open the same profile in the local web chat, run the command below and open [the local chat page](http://127.0.0.1:8000/). The profile uses the newest question in each request and serves the trained weights locally.

```bash
python -m gpt_oss.responses_api.openai_chat --host 127.0.0.1 \
  --period-checkpoint data/classical/boolelot-period
```
