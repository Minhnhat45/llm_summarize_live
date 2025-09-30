from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

from transformers import AutoTokenizer

DEFAULT_DATA_PATH = Path("result_task_2/fo_t2_fewshot_1509_qwen2.5-14b-instruct_0309_final.json")
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-1.5-pro-latest"


def load_dataset(path: Path | str = DEFAULT_DATA_PATH) -> List[Dict]:
    """Load the evaluation samples from the provided JSON file."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("Dataset JSON must contain a list of samples")

    return data


def _iter_prompt_segments(dataset: Iterable[Dict]) -> Iterator[Tuple[int, str, str, str]]:
    """Yield (index, system_text, user_text, output_text) tuples from the dataset."""
    for index, sample in enumerate(dataset):
        input_section = sample.get("input", {})
        system_text = input_section.get("system", "") or ""
        user_text = input_section.get("user", "") or ""
        output_text = sample.get("output", "") or ""
        yield index, system_text, user_text, output_text


def count_qwen_tokens(
    dataset: Iterable[Dict],
    model_id: str = DEFAULT_QWEN_MODEL,
    add_special_tokens: bool = False,
) -> Dict[str, object]:
    """Count tokens for each sample using a Qwen tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    records: List[Dict[str, int]] = []
    total_input_tokens = 0
    total_output_tokens = 0

    for index, system_text, user_text, output_text in _iter_prompt_segments(dataset):
        system_tokens = len(tokenizer.encode(system_text, add_special_tokens=add_special_tokens)) if system_text else 0
        user_tokens = len(tokenizer.encode(user_text, add_special_tokens=add_special_tokens)) if user_text else 0
        output_tokens = len(tokenizer.encode(output_text, add_special_tokens=add_special_tokens)) if output_text else 0
        input_tokens = system_tokens + user_tokens

        records.append(
            {
                "index": index,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "system_tokens": system_tokens,
                "user_tokens": user_tokens,
            }
        )

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

    return {
        "records": records,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }


def _get_openai_encoding(model_name: str):
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "tiktoken is required for OpenAI token counting. Install it with `pip install tiktoken`."
        ) from exc

    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_openai_tokens(
    dataset: Iterable[Dict],
    model_name: str = DEFAULT_OPENAI_MODEL,
) -> Dict[str, object]:
    """Count tokens for each sample using OpenAI's tiktoken encoders."""
    encoding = _get_openai_encoding(model_name)

    records: List[Dict[str, int]] = []
    total_input_tokens = 0
    total_output_tokens = 0

    for index, system_text, user_text, output_text in _iter_prompt_segments(dataset):
        system_tokens = len(encoding.encode(system_text)) if system_text else 0
        user_tokens = len(encoding.encode(user_text)) if user_text else 0
        output_tokens = len(encoding.encode(output_text)) if output_text else 0
        input_tokens = system_tokens + user_tokens

        records.append(
            {
                "index": index,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "system_tokens": system_tokens,
                "user_tokens": user_tokens,
            }
        )

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

    return {
        "records": records,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }


def _get_gemini_model(model_name: str):
    try:
        import google.generativeai as genai
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "google-generativeai is required for Gemini token counting. Install it with `pip install google-generativeai`."
        ) from exc

    return genai.GenerativeModel(model_name)


def count_gemini_tokens(
    dataset: Iterable[Dict],
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> Dict[str, object]:
    """Count tokens for each sample by delegating to the Gemini API."""
    # model = _get_gemini_model(model_name)
    from google.generativeai import genai
    client = genai.Client()
    # model = client.get_model(model_name)
    

    records: List[Dict[str, int]] = []
    total_input_tokens = 0
    total_output_tokens = 0

    for index, system_text, user_text, output_text in _iter_prompt_segments(dataset):
        input_text = "\n\n".join(part for part in (system_text, user_text) if part)
        input_tokens = client.models.count_tokens(input_text, model="genimi-2.0-flash").total_tokens if input_text else 0
        output_tokens = client.models.count_tokens(output_text, model="genimi-2.0-flash").total_tokens if output_text else 0

        records.append(
            {
                "index": index,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "system_tokens": input_tokens,  # Gemini does not expose per-part counts
                "user_tokens": 0,
            }
        )

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

    return {
        "records": records,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }


def _format_summary(provider: str, summary: Dict[str, object]) -> str:
    records = summary.get("records", [])
    total_input = summary.get("total_input_tokens", 0)
    total_output = summary.get("total_output_tokens", 0)
    return (
        f"provider={provider} samples={len(records)} "
        f"input_tokens={total_input} output_tokens={total_output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Token counting utilities")
    parser.add_argument(
        "--provider",
        action="append",
        choices=["qwen", "openai", "gemini"],
        help="Run token counting for the selected provider(s). Default: qwen",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to the dataset JSON file",
    )

    args = parser.parse_args()
    providers = args.provider or ["qwen"]

    dataset = load_dataset(args.data)

    for provider in providers:
        try:
            if provider == "qwen":
                summary = count_qwen_tokens(dataset)
            elif provider == "openai":
                summary = count_openai_tokens(dataset)
            elif provider == "gemini":
                summary = count_gemini_tokens(dataset)
            else:  # pragma: no cover - defensive fallback
                raise ValueError(f"Unknown provider: {provider}")
        except ImportError as exc:
            print(f"{provider}: skipped ({exc})")
            continue

        print(_format_summary(provider, summary))


if __name__ == "__main__":
    main()
