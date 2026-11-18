
import argparse
import json
import sys

from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local model path")
    ap.add_argument("--prompt", default="", help="a single ready-to-use prompt")
    ap.add_argument("--prompt-file", default="", help="JSONL file with a 'prompt' field per line")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    ap.add_argument("--max-model-len", type=int, default=49152)
    ap.add_argument("--max-tokens", type=int, default=24576)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--num-generations", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.prompt and not args.prompt_file:
        sys.exit("provide --prompt or --prompt-file")

    if args.prompt_file:
        prompts = [json.loads(line)["prompt"] for line in open(args.prompt_file) if line.strip()]
    else:
        prompts = [args.prompt]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    params = SamplingParams(
        n=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    outputs = llm.generate(prompts, params)
    for i, out in enumerate(outputs):
        for j, gen in enumerate(out.outputs):
            print(json.dumps(
                {"prompt_index": i, "generation": j, "text": gen.text},
                ensure_ascii=False,
            ))


if __name__ == "__main__":
    main()
