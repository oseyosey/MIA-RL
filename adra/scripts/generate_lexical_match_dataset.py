import argparse
import json
import os
from typing import List

from datasets import Dataset, Features, Sequence, Value


def read_ground_truths(path: str) -> List[str]:
    """Read ground-truth strings from txt/json file.

    If *path* ends with .json or .jsonl we expect a list or json-lines with a
    key "text". Otherwise treat the file as plain text with one string per line.
    """
    if path.endswith(('.json', '.jsonl')):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x) for x in data]
        raise ValueError('JSON ground-truth file must contain a list of strings.')
    # txt
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f if line.strip()]


def build_dataset(ground_truths: List[str], num_prompts: int) -> Dataset:
    """Create HF Dataset with blank prompts and shared *ground_truths* list."""
    records = []
    for idx in range(num_prompts):
        records.append({
            'data_source': 'lexical_match_custom',
            'prompt': "", # * blank prompt: we want the model to generate freely!
            'ground_truths': ground_truths,  # list
            'idx': idx,
        })
    features = Features({
        'data_source': Value('string'),
        'prompt': Value('string'),
        'ground_truths': Sequence(Value('string')),
        'idx': Value('int32'),
    })
    return Dataset.from_list(records, features=features)


def build_dataset_from_ground_truth_prompts(ground_truths: List[str], num_prompt_words: int = 1) -> Dataset:
    """Create HF Dataset where *prompt* is built from the first *num_prompt_words* of each ground-truth.

    Each ground-truth string will yield exactly one record. The *ground_truths* list is
    attached unchanged to each record so that evaluation logic that expects the full
    set of acceptable answers continues to work.

    Parameters
    ----------
    ground_truths: List[str]
        All target strings that should be matched.
    num_prompt_words: int, default 1
        How many words to copy from the beginning of each ground-truth into the
        prompt. If a ground-truth is shorter than *num_prompt_words*, the entire
        string is used.
    """

    records = []
    for idx, gt in enumerate(ground_truths):
        # Split on whitespace – this is a simple heuristic that works for most languages.
        prompt_tokens = gt.split()
        prompt = " ".join(prompt_tokens[:num_prompt_words]) if prompt_tokens else ""

        records.append({
            "data_source": "lexical_match_custom_gt_prompt",
            "prompt": prompt,
            "ground_truths": ground_truths,  # keep full list for evaluation
            "idx": idx,
        })

    features = Features({
        "data_source": Value("string"),
        "prompt": Value("string"),
        "ground_truths": Sequence(Value("string")),
        "idx": Value("int32"),
    })

    return Dataset.from_list(records, features=features)


def build_dataset_with_prefix_decoding(ground_truths: List[str], num_prompt_words: int = 1) -> Dataset:
    """Create HF Dataset where `assistant_prefix` is built from the first `num_prompt_words` of each ground-truth.

    Each ground-truth string will yield exactly one record. The `ground_truths` list is
    attached unchanged to each record so that evaluation logic that expects the full
    set of acceptable answers continues to work. The `prompt` is always blank.

    Parameters
    ----------
    ground_truths: List[str]
        All target strings that should be matched.
    num_prompt_words: int, default 1
        How many words to copy from the beginning of each ground-truth into the
        assistant prefix. If a ground-truth is shorter than *num_prompt_words*, the entire
        string is used.
    """

    records = []
    for idx, gt in enumerate(ground_truths):
        # Split on whitespace – this is a simple heuristic that works for most languages.
        prompt_tokens = gt.split()
        prefix = " ".join(prompt_tokens[:num_prompt_words]) if prompt_tokens else ""

        records.append({
            "data_source": "lexical_match_custom_prefix_decoding",
            "prompt": "",  # Blank prompt
            "assistant_prefix": prefix,
            "ground_truths": ground_truths,  # keep full list for evaluation
            "idx": idx,
        })

    features = Features({
        "data_source": Value("string"),
        "prompt": Value("string"),
        "assistant_prefix": Value("string"),
        "ground_truths": Sequence(Value("string")),
        "idx": Value("int32"),
    })

    return Dataset.from_list(records, features=features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a minimal string-matching HF dataset.')
    parser.add_argument('--ground_truth_file', required=True, help='Path to file containing ground-truth strings.')
    parser.add_argument('--output_dir', default='./string_match_hf', help='Directory to save the HF dataset')
    parser.add_argument('--num_prompts', type=int, default=1, help='Number of blank prompts to create when not using ground-truth-based prompts (default: 1)')
    parser.add_argument('--num_prompt_words', type=int, default=1, help='Number of words from each ground truth to use as the prompt when --use_gt_prompts is set (default: 1)')
    parser.add_argument('--use_gt_prompts', action='store_true', help='If set, build prompts from ground truths instead of using blank prompts.')
    parser.add_argument('--use_prefix_decoding', action='store_true', help='If set, build prompts with blank prompts and assistant_prefix from ground truths.')
    args = parser.parse_args()

    if args.use_gt_prompts and args.use_prefix_decoding:
        raise ValueError("--use_gt_prompts and --use_prefix_decoding cannot be used at the same time.")

    gts = read_ground_truths(args.ground_truth_file)

    if args.use_prefix_decoding:
        print("Using build_dataset_with_prefix_decoding")
        ds = build_dataset_with_prefix_decoding(gts, args.num_prompt_words)
    elif args.use_gt_prompts:
        print("Using build_dataset_from_ground_truth_prompts")
        ds = build_dataset_from_ground_truth_prompts(gts, args.num_prompt_words)
    else:
        print("Using build_dataset")
        ds = build_dataset(gts, args.num_prompts)

    os.makedirs(args.output_dir, exist_ok=True)
    ds.save_to_disk(args.output_dir)
    print(f"Dataset with {len(ds)} entries saved to {args.output_dir}") 