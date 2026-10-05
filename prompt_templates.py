"""Fixed sentiment prompts for reproducible Qwen evaluation.

Choose a dataset and template ID before an evaluation run. Keep that choice
constant for every example and model in the run.
"""

from types import MappingProxyType


# Post-audit revision for Qwen2.5-3B-Instruct on SST-2 only. The fixed
# T1-T4 prompts below remain unchanged for the earlier Qwen models.
QWEN3B_SST2_T4_PROMPT = (
    "Classify the sentiment as exactly one label: positive or negative. "
    "Return only positive or negative label. No explanation.\n\n"
    "Text: {text}"
)


PROMPT_TEMPLATES = MappingProxyType({
    "sst2": MappingProxyType({
        "template_1": (
            "Classify the sentiment of this text as positive or negative.\n"
            "Reply with only one label: positive or negative.\n\n"
            "Text: {text}"
        ),
        "template_2": (
            "Determine whether the following text expresses positive or negative sentiment.\n"
            "Answer only: positive or negative.\n\n"
            "Text: {text}"
        ),
        "template_3": (
            "Read the text and identify its sentiment.\n"
            "Output exactly one word: positive or negative.\n\n"
            "Text: {text}"
        ),
        "template_4": (
            "Is the sentiment in this text positive or negative?\n"
            "Return only the label, with no explanation.\n\n"
            "Text: {text}"
        ),
    }),
    "tweet_eval_sentiment": MappingProxyType({
        "template_1": (
            "Classify the sentiment of this text as negative, neutral, or positive.\n"
            "Reply with only one label: negative, neutral, or positive.\n\n"
            "Text: {text}"
        ),
        "template_2": (
            "Determine whether the following text expresses negative, neutral, or positive sentiment.\n"
            "Answer only: negative, neutral, or positive.\n\n"
            "Text: {text}"
        ),
        "template_3": (
            "Read the text and identify its sentiment.\n"
            "Output exactly one word: negative, neutral, or positive.\n\n"
            "Text: {text}"
        ),
        "template_4": (
            "Is the sentiment in this text negative, neutral, or positive?\n"
            "Return only the label, with no explanation.\n\n"
            "Text: {text}"
        ),
    }),
})


def render_prompt(dataset: str, template_id: str, text: str) -> str:
    """Insert the example text without modifying the selected prompt wording."""
    return PROMPT_TEMPLATES[dataset][template_id].format(text=text)
