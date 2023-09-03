import re
from pathlib import Path
from typing import List

from hazm import NUMBERS
from hazm import Lemmatizer
from hazm import Normalizer
from hazm import SentenceTokenizer
from hazm import Stemmer
from hazm import WordTokenizer
from hazm import informal_normalizer

import pyspark

from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col

from utils import default_verbs, informal_verbs, informal_words, default_words



verb_file: str = informal_verbs
word_file: str = informal_words
seperation_flag: bool = False

lemmatizer = Lemmatizer()
ilemmatizer = informal_normalizer.InformalLemmatizer()
stemmer = Stemmer()

sent_tokenizer = SentenceTokenizer()
word_tokenizer = WordTokenizer()


def informal_conjugations(verb: str) -> List[str]:
    """صورت‌های صرفی فعل را در شکل محاوره‌ای تولید می‌کند.

    Args:
        verb: فعلی که باید صرف شود.

    Returns:
        صورت‌های صرفی فعل.

    """
    ends = ["م", "ی", "", "یم", "ین", "ن"]
    present_simples = [verb + end for end in ends]
    if verb.endswith("ا"):
        present_simples[2] = verb + "د"
    else:
        present_simples[2] = verb + "ه"
    present_not_simples = ["ن" + item for item in present_simples]
    present_imperfects = ["می‌" + item for item in present_simples]
    present_not_imperfects = ["ن" + item for item in present_imperfects]
    present_subjunctives = [
        item if item.startswith("ب") else "ب" + item for item in present_simples
    ]
    present_not_subjunctives = ["ن" + item for item in present_simples]
    return (
        present_simples
        + present_not_simples
        + present_imperfects
        + present_not_imperfects
        + present_subjunctives
        + present_not_subjunctives
    )

def informal_to_formal_conjucation(i, f, flag):
    iv = informal_conjugations(i)
    fv = lemmatizer.conjugation.get_all(f)
    res = {}
    if flag:
        for i, j in zip(iv, fv[48:]):
            res[i] = j
            if "‌" in i:
                res[i.replace("‌", "")] = j
                res[i.replace("‌", " ")] = j
            if i.endswith("ین"):
                res[i[:-1] + "د"] = j
    else:
        for i, j in zip(iv[8:], fv[56:]):
            res[i] = j
            if "‌" in i:
                res[i.replace("‌", "")] = j
                res[i.replace("‌", " ")] = j
            if i.endswith("ین"):
                res[i[:-1] + "د"] = j

    return res

def informal_to_formal_conjucation(i, f, flag):
            iv = informal_conjugations(i)
            fv = lemmatizer.conjugation.get_all(f)
            res = {}
            if flag:
                for i, j in zip(iv, fv[48:]):
                    res[i] = j
                    if "‌" in i:
                        res[i.replace("‌", "")] = j
                        res[i.replace("‌", " ")] = j
                    if i.endswith("ین"):
                        res[i[:-1] + "د"] = j
            else:
                for i, j in zip(iv[8:], fv[56:]):
                    res[i] = j
                    if "‌" in i:
                        res[i.replace("‌", "")] = j
                        res[i.replace("‌", " ")] = j
                    if i.endswith("ین"):
                        res[i[:-1] + "د"] = j

            return res

with Path.open(verb_file, encoding="utf8") as vf:
            iverb_map = {}
            for f, i, flag in [x.strip().split(" ", 2) for x in vf]:
                iverb_map.update(informal_to_formal_conjucation(i, f, flag))

with Path.open(word_file, encoding="utf8") as wf:
    iword_map = dict([x.strip().split(" ", 1) for x in wf])


with Path.open(verb_file, encoding="utf8") as vf:
    pastVerbs = {}
    presentVerbs = {}
    for f, i, _flag in [x.strip().split(" ", 2) for x in vf]:
        splitedf = f.split("#")
        presentVerbs.update({i: splitedf[1]})
        pastVerbs.update({splitedf[0]: splitedf[0]})
with Path.open(default_verbs, encoding="utf8") as vf:
    for f, i in [x.strip().split("#", 2) for x in vf]:
        presentVerbs.update({i: i})
        pastVerbs.update({f: f})




@udf
def split_token_words(token: str) -> str:
    """هرجایی در متن فاصله نیاز بود قرار می‌دهد.

    متأسفانه در برخی از متن‌ها، به بهانهٔ صرفه‌جویی در زمان یا از سرِ تنبلی،
    فاصله‌گذاری‌ها درست رعایت نمی‌شود. مثلاً جملهٔ «تو را دوست دارم.» به این
    شکل نوشته می‌شود: «تورادوست دارم.» این تابع فواصل ضروری را در متن
    ایجاد می‌کند و آن را به شکل صحیح برمی‌گرداند.

    Args:
        token: توکنی که باید فاصله‌گذاری شود.

    Returns:
        توکنی با فاصله‌گذاری صحیح.

    """
    with Path.open(verb_file, encoding="utf8") as vf:
        iverb_map = {}
        for f, i, flag in [x.strip().split(" ", 2) for x in vf]:
            iverb_map.update(informal_to_formal_conjucation(i, f, flag))

    with Path.open(word_file, encoding="utf8") as wf:
        iword_map = dict([x.strip().split(" ", 1) for x in wf])

    # words = set()
    tokenizer = WordTokenizer(join_verb_parts=False)
    words = tokenizer.words
    if seperation_flag:
        words.update(list(iword_map.keys()))
        words.update(list(iword_map.values()))
        words.update(list(iverb_map.keys()))
        words.update(list(iverb_map.values()))
        words.update(lemmatizer.words)
        words.update(list(lemmatizer.verbs.keys()))
        words.update(list(lemmatizer.verbs.values()))

    def shekan(token):
        res = [""]
        for i in token:
            res[-1] += i
            if i in {"ا", "د", "ذ", "ر", "ز", "ژ", "و", *list(NUMBERS)}:
                res.append("")
        while "" in res:
            res.remove("")
        return res

    def perm(lst):
        if len(lst) > 1:
            up = perm(lst[1:])
        else:
            return [lst]
        res = []
        for i in up:
            res.append([lst[0], *i])
            res.append([lst[0] + i[0]] + i[1:])
        res.sort(key=len)
        return res

    token = re.sub(r"(.)\1{2,}", r"\1", token)
    ps = perm(shekan(token))
    #verbs, words = informalLemmatizer()
    for c in ps:
        if {lemmatizer.lemmatize(x) for x in c}.issubset(words):
            return " ".join(c)
    return token


def normalized_word(word: str) -> List[str]:
        """اشکال مختلف نرمالایزشدهٔ کلمه را برمی‌گرداند.

        Examples:
            >>> normalizer = InformalNormalizer()
            >>> normalizer.normalized_word('می‌رم')
            ['می‌روم', 'می‌رم']

        Args:
            word: کلمه‌ای که باید نرمال‌سازی شود.

        Returns:
            اشکال نرمالایزشدهٔ کلمه.

        """
        # >>> normalizer = InformalNormalizer(seperation_flag=True)
        # >>> normalizer.normalized_word('صداوسیماجمهوری')

        def analyze_word(word):
            end_words_list = [
                "هاست",
                "هایی",
                "هایم",
                "ترین",
                "ایی",
                "انی",
                "شان",
                "شون",
                "است",
                "تان",
                "تون",
                "مان",
                "مون",
                "هام",
                "هاش",
                "های",
                "طور",
                "ها",
                "تر",
                "ئی",
                "یی",
                "یم",
                "ام",
                "ای",
                "ان",
                "هم",
                "رو",
                "یت",
                "ه",
                "ی",
                "ش",
                "و",
                "ا",
                "ت",
                "م",
            ]

            return_list = []

            collection_of_word_and_suffix = []

            found_early = False

            mid_word_condidate = []

            if word.endswith(("\u200c", "\u200e")):
                word = word[:-1]

            if word in lemmatizer.words or word in iword_map:
                if word in lemmatizer.words:
                    collection_of_word_and_suffix.append({"word": word, "suffix": []})
                if word in iword_map:
                    collection_of_word_and_suffix.append(
                        {"word": iword_map[word], "suffix": []},
                    )
                found_early = True

            if not found_early:
                for endword in end_words_list:
                    if word.endswith(endword):
                        sliceword = word[: -1 * len(endword)]
                        if (
                            sliceword in lemmatizer.words
                            or sliceword in iword_map
                        ):
                            if sliceword in lemmatizer.words:
                                collection_of_word_and_suffix.append(
                                    {"word": sliceword, "suffix": [endword]},
                                )
                            if sliceword in iword_map:
                                collection_of_word_and_suffix.append(
                                    {
                                        "word": iword_map[sliceword],
                                        "suffix": [endword],
                                    },
                                )
                        else:
                            mid_word_condidate.append(sliceword)
                            mid_word_condidate.append([endword])

                for endword in end_words_list:
                    for i in range(len(mid_word_condidate) - 1):
                        if i % 2 == 1:
                            continue
                        midword = mid_word_condidate[i]
                        mid_word_end_word_list = mid_word_condidate[i + 1]
                        if midword.endswith(endword):
                            sliceword = midword[: -1 * len(endword)]
                            if (
                                sliceword in lemmatizer.words
                                or sliceword in iword_map
                            ):
                                if sliceword in lemmatizer.words:
                                    collection_of_word_and_suffix.append(
                                        {
                                            "word": sliceword,
                                            "suffix": [
                                                endword,
                                                *mid_word_end_word_list,
                                            ],
                                        },
                                    )
                                if sliceword in iword_map:
                                    collection_of_word_and_suffix.append(
                                        {
                                            "word": iword_map[sliceword],
                                            "suffix": [
                                                endword,
                                                *mid_word_end_word_list,
                                            ],
                                        },
                                    )

            for i in range(len(collection_of_word_and_suffix)):
                new_possible_word_list = append_suffix_to_word(
                    collection_of_word_and_suffix[i],
                )
                for j in range(len(new_possible_word_list)):
                    new_possible_word = new_possible_word_list[j]
                    if new_possible_word not in return_list:
                        return_list.append(new_possible_word)

            return return_list

        def analyze_verb_word(word):
            if word in pastVerbs:
                word = pastVerbs[word]
                return [word]

            if word in iword_map:
                return []

            if word in lemmatizer.words:
                if word[-1] == "ن":
                    None  # noqa: B018
                else:
                    return []

            return_list = []

            collection_of_verb_list = []

            end_verb_list = [
                "یم",
                "دم",
                "دیم",
                "ید",
                "دی",
                "دید",
                "ند",
                "دن",
                "دند",
                "ین",
                "دین",
                "ست",
                "ستم",
                "ستی",
                "ستیم",
                "ستید",
                "ستند",
                "ستن",
                "م",
                "ی",
                "ه",
                "د",
                "ن",
            ]

            for endverb in end_verb_list:
                if word.endswith(endverb):
                    if endverb == "ین":
                        collection_of_verb_list.append(
                            {"word": word[:-2], "suffix": "ید"},
                        )
                    elif endverb == "ستن":
                        collection_of_verb_list.append(
                            {"word": word[:-3], "suffix": "ستند"},
                        )
                    elif endverb == "ن":
                        collection_of_verb_list.append(
                            {"word": word[:-1], "suffix": "ن"},
                        )
                        collection_of_verb_list.append(
                            {"word": word[:-1], "suffix": "ند"},
                        )
                    elif endverb == "ه":
                        if len(word) > 1:
                            if word[-2] != "د":
                                collection_of_verb_list.append(
                                    {"word": word[:-1], "suffix": "د"},
                                )
                            collection_of_verb_list.append(
                                {"word": word[:-1], "suffix": "ه"},
                            )
                        else:
                            collection_of_verb_list.append(
                                {"word": word[:-1], "suffix": "ه"},
                            )
                    else:
                        collection_of_verb_list.append(
                            {
                                "word": word[: -1 * len(endverb)],
                                "suffix": endverb,
                            },
                        )
            collection_of_verb_list.append({"word": word, "suffix": ""})
            collection_of_verb_list_2 = []
            for i in range(len(collection_of_verb_list)):
                mainword = collection_of_verb_list[i]["word"]
                collection_of_verb_list[i]["preffix"] = ""
                if mainword.startswith("بر"):
                    modified_word = mainword[2:]
                    new_main_word = ""
                    if modified_word.startswith("نمی"):
                        collection_of_verb_list[i]["preffix"] = "برنمی"
                        new_main_word = modified_word[3:]
                    elif modified_word.startswith("می"):
                        collection_of_verb_list[i]["preffix"] = "برمی"
                        new_main_word = modified_word[2:]
                    elif modified_word.startswith("ن"):
                        collection_of_verb_list[i]["preffix"] = "برن"
                        new_main_word = modified_word[1:]
                    elif modified_word.startswith("بی"):
                        collection_of_verb_list[i]["preffix"] = "بربی"
                        new_main_word = modified_word[2:]
                    elif modified_word.startswith("ب"):
                        collection_of_verb_list[i]["preffix"] = "برب"
                        new_main_word = modified_word[1:]
                    else:
                        collection_of_verb_list[i]["preffix"] = "بر"
                        new_main_word = modified_word
                        collection_of_verb_list_2.append(
                            {
                                "word": mainword,
                                "preffix": "",
                                "suffix": collection_of_verb_list[i]["suffix"],
                            },
                        )

                    if new_main_word:
                        collection_of_verb_list[i]["word"] = new_main_word
                elif mainword.startswith("نمی"):
                    collection_of_verb_list[i]["preffix"] = "نمی"
                    collection_of_verb_list[i]["word"] = mainword[3:]
                elif mainword.startswith("می"):
                    collection_of_verb_list[i]["preffix"] = "می"
                    collection_of_verb_list[i]["word"] = mainword[2:]
                elif mainword.startswith("ن"):
                    collection_of_verb_list[i]["preffix"] = "ن"
                    collection_of_verb_list[i]["word"] = mainword[1:]
                    collection_of_verb_list_2.append(
                        {
                            "word": mainword,
                            "preffix": "",
                            "suffix": collection_of_verb_list[i]["suffix"],
                        },
                    )

                elif mainword.startswith("بی"):
                    collection_of_verb_list[i]["preffix"] = "بی"
                    collection_of_verb_list[i]["word"] = mainword[2:]
                elif mainword.startswith("ب"):
                    collection_of_verb_list[i]["preffix"] = "ب"
                    collection_of_verb_list[i]["word"] = mainword[1:]
                    collection_of_verb_list_2.append(
                        {
                            "word": mainword,
                            "preffix": "",
                            "suffix": collection_of_verb_list[i]["suffix"],
                        },
                    )

            for i in range(len(collection_of_verb_list_2)):
                collection_of_verb_list.append(collection_of_verb_list_2[i])

            collection_of_real_verb_list = []
            for i in range(len(collection_of_verb_list)):
                mainword = collection_of_verb_list[i]["word"]
                if mainword.startswith(("\u200c", "\u200e")):
                    mainword = mainword[1:]

                mainword2 = None
                if mainword.startswith("ا"):
                    mainword2 = "آ" + mainword[1:]
                if mainword in pastVerbs:
                    collection_of_verb_list[i]["word"] = pastVerbs[mainword]
                    collection_of_real_verb_list.append(collection_of_verb_list[i])
                if mainword in presentVerbs:
                    collection_of_verb_list[i]["word"] = presentVerbs[mainword]
                    collection_of_real_verb_list.append(collection_of_verb_list[i])
                if mainword2 is not None and not (
                    collection_of_verb_list[i]["preffix"] == "بربی"
                    or collection_of_verb_list[i]["preffix"] == "بی"
                ):
                    if mainword2 in pastVerbs:
                        collection_of_verb_list[i]["word"] = pastVerbs[mainword2]
                        collection_of_real_verb_list.append(collection_of_verb_list[i])
                    if mainword2 in presentVerbs:
                        collection_of_verb_list[i]["word"] = presentVerbs[
                            mainword2
                        ]
                        collection_of_real_verb_list.append(collection_of_verb_list[i])

            for i in range(len(collection_of_real_verb_list)):
                preffix = collection_of_real_verb_list[i]["preffix"]
                suffix = collection_of_real_verb_list[i]["suffix"]
                mainword = collection_of_real_verb_list[i]["word"]
                returnword = preffix
                if preffix.endswith("می"):
                    returnword += "‌"
                returnword += mainword
                returnword += suffix
                if mainword and returnword not in return_list:
                    return_list.append(returnword)

            return return_list

        def append_suffix_to_word(one_collection_of_word_and_suffix):
            mainword = one_collection_of_word_and_suffix["word"]
            suffix_list = one_collection_of_word_and_suffix["suffix"]
            adhesive_alphabet = {
                "ب": "ب",
                "پ": "پ",
                "ت": "ت",
                "ث": "ث",
                "ج": "ج",
                "چ": "چ",
                "ح": "ح",
                "خ": "خ",
                "س": "س",
                "ش": "ش",
                "ص": "ص",
                "ض": "ض",
                "ع": "ع",
                "غ": "غ",
                "ف": "ف",
                "ق": "ق",
                "ک": "ک",
                "گ": "گ",
                "ل": "ل",
                "م": "م",
                "ن": "ن",
                "ه": "ه",
                "ی": "ی",
            }
            return_list = []
            returnword = mainword
            return_word2 = None
            return_word3 = None
            if len(suffix_list) == 0:
                return [returnword]
            if len(suffix_list) > 1:
                if suffix_list[0] == "ه" and suffix_list[1] == "ا":
                    suffix_list[0] = "ها"
                    suffix_list.remove(suffix_list[1])
                if suffix_list[0] == "ه" and suffix_list[1] == "است":
                    suffix_list[0] = "هاست"
                    suffix_list.remove(suffix_list[1])
                if suffix_list[0] == "ت" and suffix_list[1] == "ا":
                    suffix_list[0] = "تا"
                    suffix_list.remove(suffix_list[1])
            for i in range(len(suffix_list)):
                if suffix_list[i] == "شون":
                    returnword += "شان"
                elif suffix_list[i] == "تون":
                    returnword += "تان"
                elif suffix_list[i] == "مون":
                    returnword += "مان"
                elif suffix_list[i] == "هام":
                    try:
                        adhesive_alphabet[returnword[-1]]
                        returnword += "‌"
                    except:
                        None  # noqa: B018
                    returnword += "هایم"
                elif suffix_list[i] == "ها":  # noqa: SIM114
                    try:
                        adhesive_alphabet[returnword[-1]]
                        returnword += "‌"
                    except:
                        None  # noqa: B018
                    returnword += "ها"
                elif (
                    suffix_list[i] == "ا"
                    and suffix_list[len(suffix_list) - 1] == "ا"
                    and not returnword.endswith("ه")
                ):
                    try:
                        adhesive_alphabet[returnword[-1]]
                        returnword += "‌"
                    except:
                        None  # noqa: B018
                    returnword += "ها"
                elif suffix_list[i] == "و" and suffix_list[len(suffix_list) - 1] == "و":
                    return_word2 = returnword
                    return_word2 += " و"
                    returnword += " را"

                elif (
                    suffix_list[i] == "رو" and suffix_list[len(suffix_list) - 1] == "رو"
                ):
                    returnword += " را"

                elif suffix_list[i] == "ه" and suffix_list[len(suffix_list) - 1] == "ه":
                    return_word2 = returnword
                    return_word2 += "ه"
                    return_word3 = returnword
                    return_word3 += " است"
                    returnword += "ه است"
                else:
                    returnword += suffix_list[i]
            return_list.append(returnword)
            if return_word2 is not None:
                return_list.append(return_word2)
            if return_word3 is not None:
                return_list.append(return_word3)
            return return_list

        def straight_forward_result(word):
            straight_forward_dic = {
                "ب": ["به"],
                "ک": ["که"],
                "آن": ["آن"],
                "می‌آید": ["می‌آید"],
                "میاید": ["می‌آید"],
                "می‌آیم": ["می‌آیم"],
                "میایم": ["می‌آیم"],
                "نمی‌آید": ["نمی‌آید"],
                "نمیاید": ["نمی‌آید"],
                "نمی‌آیم": ["نمی‌آیم"],
                "نمیایم": ["نمی‌آیم"],
                "برمی‌آید": ["برمی‌آید"],
                "برمیاید": ["برمی‌آید"],
                "برمی‌آیم": ["برمی‌آیم"],
                "برمیایم": ["برمی‌آیم"],
                "برنمی‌آید": ["برنمی‌آید"],
                "برنمیاید": ["برنمی‌آید"],
                "برنمی‌آیم": ["برنمی‌آیم"],
                "برنمیایم": ["برنمی‌آیم"],
                "منظوره": ["منظوره"],
                "بدن": ["بدن"],
                "میا": ["میا"],
                "نیس": ["نیست"],
                "فک": ["فکر"],
                "برام": ["برایم"],
                "آ": ["آ"],
                "آی": ["آی"],
                "این": ["این"],
                "است": ["است"],
                "ان": ["ان"],
                "اند": ["اند"],
                "میان": ["میان"],
                "گردن": ["گردن"],
                "اینهمه": ["اینهمه"],
                "آنهمه": ["آنهمه"],
                "الیه": ["الیه"],
                "غرغره": ["غرغره"],
                "لیله": ["لیله"],
                "بزرگانه": ["بزرگانه"],
                "پرستانه": ["پرستانه"],
                "ام": ["ام"],
                "بادی": ["بادی"],
                "نان": ["نان"],
                "باورم": ["باورم"],
                "اوه": ["اوه"],
                "چقد": ["چقدر"],
                "چو": ["چون"],
                "هس": ["هست"],
                "اومدند": ["آمدند"],
                "ش": ["اش"],
                "بش": ["بهش"],
                "ازت": ["از تو"],
                "رو": ["را", "رو"],
                "پایتون": ["پایتون"],
                "اردن": ["اردن"],
                "دست": ["دست"],
                "دستی": ["دستی"],
                "دستم": ["دستم"],
                "دین": ["دین"],
                "شین": ["شین"],
                "سراتو": ["سراتو"],
                "فالو": ["فالو"],
                "هرجا": ["هرجا"],
                "میدان": ["میدان"],
                "میدون": ["میدان"],
                "کفا": ["کفا"],
                "ویا": ["و یا"],
                "نشد": ["نشد"],
                "شو": ["شو"],
                "مشیا": ["مشیا"],
                "پلاسما": ["پلاسما"],
                "فیلیمو": ["فیلیمو"],
                "پاشو": ["پاشو"],
                "میر": ["میر"],
                "بارم": ["بار هم", "بارم"],
                "برند": ["برند"],
                "کنه": ["کند"],
                "بتونه": ["بتواند"],
                "باشه": ["باشد"],
                "بخوان": ["بخوان"],
                "بدم": ["بدم"],
                "برم": ["برم"],
                "بده": ["بده"],
                "نده": ["نده"],
                "شهرو": ["شهرو"],
                "شیرو": ["شیرو"],
                "نگذاشته": ["نگذاشته"],
                "نگرفته": ["نگرفته"],
                "نمیشناخته": ["نمی‌شناخته"],
                "نمی‌شناخته": ["نمی‌شناخته"],
                "بشین": ["بشین"],
                "هارو": ["ها را"],
                "مارو": ["ما را"],
                "میاومد": ["می‌آمد"],
                "می‌اومد": ["می‌آمد"],
                "میخواسته": ["می‌خواسته"],
                "می‌خواسته": ["می‌خواسته"],
                "نمیخواسته": ["نمی‌خواسته"],
                "نمی‌خواسته": ["نمی‌خواسته"],
                "میتوانسته": ["می‌توانسته"],
                "می‌توانسته": ["می‌توانسته"],
                "میرفته": ["می‌رفته"],
                "می‌رفته": ["می‌رفته"],
                "نشین": ["نشین"],
                "انا": ["انا"],
                "خونی": ["خونی"],
                "خون": ["خون"],
                "یالا": ["یالا"],
                "میخواند": ["می‌خواند"],
                "می‌خواند": ["می‌خواند"],
                "نمیخواند": ["نمی‌خواند"],
                "نمی‌خواند": ["نمی‌خواند"],
                "میده": ["می‌دهد"],
                "می‌ده": ["می‌دهد"],
                "میشه": ["می‌شود"],
                "می‌شه": ["می‌شود"],
                "میشد": ["می‌شد"],
                "می‌شد": ["می‌شد"],
                "میشدم": ["می‌شدم"],
                "می‌شدم": ["می‌شدم"],
                "نمیشد": ["نمی‌شد"],
                "نمی‌شد": ["نمی‌شد"],
                "بردم": ["بردم"],
                "بره": ["بره", "برود"],
                "شم": ["بشوم"],
                "اوست": ["اوست"],
                "بیا": ["بیا"],
                "نیا": ["نیا"],
                "میاد": ["می‌آید"],
                "نشدی": ["نشدی"],
                "بخواند": ["بخواند"],
                "سیا": ["سیا"],
                "میدید": ["می‌دید"],
                "می‌دید": ["می‌دید"],
                "وا": ["وا"],
                "برگشته": ["برگشته"],
                "میخواست": ["می‌خواست"],
                "می‌خواست": ["می‌خواست"],
            }
            try:
                return straight_forward_dic[word]
            except:
                return []

        straight_forward_words = straight_forward_result(word)
        if len(straight_forward_words) > 0:
            return straight_forward_words

        verb_words_list = analyze_verb_word(word)
        if len(verb_words_list) > 0:
            return verb_words_list
        possible_words = analyze_word(word)

        mainword = word
        if mainword in possible_words:
            possible_words.remove(mainword)
            possible_words.append(mainword)
        elif len(possible_words) == 0:
            possible_words.append(mainword)

        return possible_words



def normalize(text: str) -> List[List[List[str]]]:
        """متن محاوره‌ای را به متن فارسی معیار تبدیل می‌کند.

        Examples:
            >>> normalizer = InformalNormalizer()
            >>> normalizer.normalize('بابا یه شغل مناسب واسه بچه هام پیدا کردن که به جایی برنمیخوره !')
            [[['بابا'], ['یک'], ['شغل'], ['مناسب'], ['برای'], ['بچه'], ['هایم'], ['پیدا'], ['کردن', 'کردند'], ['که'], ['به'], ['جایی'], ['برنمی\u200cخورد', 'برنمی\u200cخوره'], ['!']]]
            >>> normalizer = InformalNormalizer()
            >>> normalizer.normalize('اجازه بدیم همسرمون در جمع خانواده‌اش احساس آزادی کنه و فکر نکنه که ما دائم هواسمون بهش هست .')
            [[['اجازه'], ['بدهیم'], ['همسرمان'], ['در'], ['جمع'], ['خانواده\u200cاش'], ['احساس'], ['آزادی'], ['کند'], ['و'], ['فکر'], ['نکند', 'نکنه'], ['که'], ['ما'], ['دائم'], ['حواسمان'], ['بهش'], ['هست'], ['.']]]

        Args:
            text: متن محاوره‌ای که باید تبدیل به متن فارسی معیار شود.

        Returns:
           متن فارسی معیار.

        """
        norm = Normalizer()
        text = norm.normalize(text)
        sents = [
            word_tokenizer.tokenize(sentence)
            for sentence in sent_tokenizer.tokenize(text)
        ]

        return [[normalized_word(word) for word in sent] for sent in sents]


def informal_normalize(
            df: pyspark.sql.dataframe.DataFrame, 
            colname: str,
            ) -> pyspark.sql.dataframe.DataFrame:

    new_col_name="normalize_"+colname
    df = df.withColumn(new_col_name, col(colname))

    df = df.select(*[normalized_word(new_col_name).alias(new_col_name) if column == new_col_name else column for column in df.columns])


    return df

if __name__ == "__main__":

    # spark_session = SparkSession.builder.appName(
    #     'Spark_Session').getOrCreate()

    # df = spark_session.createDataFrame([
    #                                     ('﷽',1),
    #                                     ("5 درصد حَذفِ اِعراب",2),
    #                                     ("بابا یه شغل مناسب واسه بچه هام پیدا کردن که به جایی برنمیخوره !",3),
    #                                     ("تورادوست دارم",4),
    #                                     ("سلام   دنیا",5)], ("text",'id'))
    
    # normalize_df = informal_normalize(df, 'text')
    
    # normalize_df.select("text").show()
    # normalize_df.select("normalize_text").show()

    print(normalized_word("می‌رم"))
    print(normalize("بابا یه شغل مناسب واسه بچه هام پیدا کردن که به جایی برنمیخوره !"))
    inf_norm = informal_normalizer.InformalNormalizer()
    print(inf_norm.split_token_words(token="تورادوست دارم"))