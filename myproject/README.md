# myapp-template

A sample Python project using:

- `src/` layout
- dataclass-based config
- `.env` support
- `logging`
- split `train / infer / eval` entrypoints
- `pytest`

## Install

```bash
python -m pip install -e .
```

## Run without installation shortcuts

```bash
python -m myapp.train --config configs/train.yaml
python -m myapp.infer --config configs/infer.yaml
python -m myapp.eval --config configs/eval.yaml
```

## Run with console scripts

```bash
myapp-train --config configs/train.yaml
myapp-infer --config configs/infer.yaml
myapp-eval --config configs/eval.yaml
```

## Tests

```bash
pytest
```
