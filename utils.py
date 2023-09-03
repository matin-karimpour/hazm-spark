import re
from os import path
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

data_path = Path(__file__).parent / "data"

default_words = Path(data_path) / "words.dat"
default_stopwords = Path(data_path) / "stopwords.dat"
default_verbs = Path(data_path) / "verbs.dat"
informal_words = Path(data_path) / "iwords.dat"
informal_verbs = Path(data_path) / "iverbs.dat"

def words_list(
    words_file: str = default_words,
) -> List[Tuple[str, int, Tuple[str]]]:
    """لیست کلمات را برمی‌گرداند.

    Examples:
        >>> from hazm.utils import words_list
        >>> words_list()[1]
        ('آب', 549005877, ('N', 'AJ'))

    Args:
        words_file: مسیر فایل حاوی کلمات.

    Returns:
        فهرست کلمات.

    """
    with Path.open(words_file, encoding="utf-8") as words_file:
        items = [line.strip().split("\t") for line in words_file]
        return [
            (item[0], int(item[1]), tuple(item[2].split(",")))
            for item in items
            if len(item) == 3
        ]
