"""Load LeetCodeDataset and clean the problem text.

The scraped descriptions lost their superscripts: "10<sup>5</sup>" became "105" and "O(n<sup>2</sup>)" became
"O(n2)". Magnitudes are strong algorithm cues (n <= 20 hints bitmask, n <= 10^5 hints n log n), so we restore them
as explicit tokens like "pow5".
"""

import re
import unicodedata

import pandas as pd

from leetfly import paths

TEXT_VERSION = 2  # bump when cleaning changes; invalidates cached tasks
_POW = re.compile(r"(?<![\w.^])(-?)10([1-9])(?![\d.])")  # "104", "-109", "2 * 105" -> 10^d
_BIGO = re.compile(r"O\(([a-z])([2-9])\)")  # "O(n2)" -> "O(n^2)"
# Worked examples are mostly variable names and numbers; dropping them helped on dev CV (see README, pilot log).
_EXAMPLES = re.compile(r"Example\s*\d*:.*?(?=Example\s*\d*:|Constraints:|$)", re.S)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = _EXAMPLES.sub(" ", text)
    text = _BIGO.sub(r"O(\1^\2)", text)
    text = _POW.sub(r"\g<1>pow\2", text)
    return text


def problem_text(title: str, description: str) -> str:
    """The title is a strong, short cue, so it is counted twice."""
    return f"{title}. {title}. {clean_text(description)}".strip()


def title_from_slug(slug: str) -> str:
    return slug.replace("-", " ")


def load() -> pd.DataFrame:
    raw = pd.read_parquet(paths.LEETCODE)
    df = pd.DataFrame(
        {
            "slug": raw["task_id"],
            "qid": raw["question_id"].astype(int),
            "title": raw["task_id"].map(title_from_slug),
            "difficulty": raw["difficulty"],
            "tags": raw["tags"].map(list),
            "date": pd.to_datetime(raw["estimated_date"]),
            "official_split": raw["split"],
        }
    )
    df["text"] = [problem_text(t, d) for t, d in zip(df["title"], raw["problem_description"])]
    return df.sort_values(["date", "qid"]).reset_index(drop=True)
