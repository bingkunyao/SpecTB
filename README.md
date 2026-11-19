# SpecTB

SpecTB is a framework that internalizes verification knowledge (from RTL code and test-point IR) into model parameters, so that at inference the model generates a testbench end-to-end from the design specification only (no RTL, no IR).

This folder contains the training data samples, the merged inference model (hosted on Hugging Face), and an inference script.

## Data

`data/` contains four aligned JSONL files. Each line has exactly two fields:

```json
{"number": "<design id>", "content": "..."}
```

| File | `content` |
|---|---|
| `spec.jsonl` | module specification text |
| `code.jsonl` | golden RTL implementation |
| `ir.jsonl` | test-point IR, serialized as a JSON string |
| `tb.jsonl` | testbench |
| `test.jsonl` | test dataset |

## Model

The model is available on Hugging Face:

```
zwhc/SpecTB
```

## Inference

Install dependencies and run:

```bash
pip install -r inference/requirements.txt

python inference/infer.py \
  --model zwhc/SpecTB \
  --prompt "Your specification prompt here" \
  --tensor-parallel-size 1
```


## Notes

- The data is derived from multiple open RTL/testbench sources; please follow the corresponding upstream licenses.
- The release is only for research use.
