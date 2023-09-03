import re
import pyspark

from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from typing import List
from hazm import Lemmatizer
from hazm import WordTokenizer


def token_spacing(tokens: List[str]) -> List[str]:
        """توکن‌های ورودی را به فهرستی از توکن‌های نرمال‌سازی شده تبدیل می‌کند.
        در این فرایند ممکن است برخی از توکن‌ها به یکدیگر بچسبند؛
        برای مثال: `['زمین', 'لرزه', 'ای']` تبدیل می‌شود به: `['زمین‌لرزه‌ای']`.

        Examples:
            >>> normalizer = Normalizer()
            >>> normalizer.token_spacing(['کتاب', 'ها'])
            ['کتاب‌ها']
            >>> normalizer.token_spacing(['او', 'می', 'رود'])
            ['او', 'می‌رود']
            >>> normalizer.token_spacing(['ماه', 'می', 'سال', 'جدید'])
            ['ماه', 'می', 'سال', 'جدید']
            >>> normalizer.token_spacing(['اخلال', 'گر'])
            ['اخلال‌گر']
            >>> normalizer.token_spacing(['زمین', 'لرزه', 'ای'])
            ['زمین‌لرزه‌ای']
            >>> normalizer.token_spacing([])
            []

        Args:
            tokens: توکن‌هایی که باید نرمال‌سازی شود.

        Returns:
            لیستی از توکن‌های نرمال‌سازی شده به شکل `[token1, token2, ...]`.

        """
        # >>> normalizer.token_spacing(['پرداخت', 'شده', 'است'])
        suffixes = {
                "ی",
                "ای",
                "ها",
                "های",
                "تر",
                "تری",
                "ترین",
                "گر",
                "گری",
                "ام",
                "ات",
                "اش",
            }
        result = []
        verbs = Lemmatizer(joined_verb_parts=False).verbs
        #words = word_tokenizer.words
        tokenizer = WordTokenizer(join_verb_parts=False)
        words = tokenizer.words
        for t, token in enumerate(tokens):
            joined = False

            if result:
                token_pair = result[-1] + "‌" + token
                if (
                    token_pair in verbs
                    or token_pair in words
                    and words[token_pair][0] > 0
                ):
                    joined = True

                    if (
                        t < len(tokens) - 1
                        and token + "_" + tokens[t + 1] in verbs
                    ):
                        joined = False

                elif token in suffixes and result[-1] in words:
                    joined = True

            if joined:
                result.pop()
                result.append(token_pair)
            else:
                result.append(token)

        return result




@udf
def persian_style(text: str) -> str:
        persian_style_patterns = [
                ('"([^\n"]+)"', r"«\1»"),  # replace quotation with gyoome
                (r"([\d+])\.([\d+])", r"\1٫\2"),  # replace dot with momayez
                (r" ?\.\.\.", " …"),  # replace 3 dots
            ]
        """برخی از حروف و نشانه‌ها را با حروف و نشانه‌های فارسی جایگزین می‌کند.

        Examples:
            >>> normalizer = Normalizer()
            >>> normalizer.persian_style('"نرمال‌سازی"')
            '«نرمال‌سازی»'
            >>> normalizer.persian_style('و ...')
            'و …'
            >>> normalizer.persian_style('10.450')
            '10٫450'
            >>> normalizer.persian_style('')
            ''

        Args:
            text: متنی که باید حروف و نشانه‌های آن با حروف و نشانه‌های فارسی جایگزین شود.

        Returns:
            متنی با حروف و نشانه‌های فارسی‌سازی شده.

        """
        for pattern, repl in persian_style_patterns:
            text = re.sub(pattern, repl, text)
        return text
@udf
def remove_diacritics(text: str) -> str:
      diacritics_patterns = [
              # remove FATHATAN, DAMMATAN, KASRATAN, FATHA, DAMMA, KASRA, SHADDA, SUKUN
              ("[\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652]", ""),
          ]
      """اِعراب را از متن حذف می‌کند.

      Examples:
          >>> normalizer = Normalizer()
          >>> normalizer.remove_diacritics('حَذفِ اِعراب')
          'حذف اعراب'
          >>> normalizer.remove_diacritics('آمدند')
          'آمدند'
          >>> normalizer.remove_diacritics('متن بدون اعراب')
          'متن بدون اعراب'
          >>> normalizer.remove_diacritics('')
          ''

      Args:
          text: متنی که باید اعراب آن حذف شود.

      Returns:
          متنی بدون اعراب.

      """
      for pattern, repl in diacritics_patterns:
            text = re.sub(pattern, repl, text)
      return text
@udf
def remove_specials_chars( text: str) -> str:
        specials_chars_patterns = [
                # Remove almoast all arabic unicode superscript and subscript characters in the ranges of 00600-06FF, 08A0-08FF, FB50-FDFF, and FE70-FEFF
                (
                    "[\u0605\u0653\u0654\u0655\u0656\u0657\u0658\u0659\u065a\u065b\u065c\u065d\u065e\u065f\u0670\u0610\u0611\u0612\u0613\u0614\u0615\u0616\u0618\u0619\u061a\u061e\u06d4\u06d6\u06d7\u06d8\u06d9\u06da\u06db\u06dc\u06dd\u06de\u06df\u06e0\u06e1\u06e2\u06e3\u06e4\u06e5\u06e6\u06e7\u06e8\u06e9\u06ea\u06eb\u06ec\u06ed\u06fd\u06fe\u08ad\u08d4\u08d5\u08d6\u08d7\u08d8\u08d9\u08da\u08db\u08dc\u08dd\u08de\u08df\u08e0\u08e1\u08e2\u08e3\u08e4\u08e5\u08e6\u08e7\u08e8\u08e9\u08ea\u08eb\u08ec\u08ed\u08ee\u08ef\u08f0\u08f1\u08f2\u08f3\u08f4\u08f5\u08f6\u08f7\u08f8\u08f9\u08fa\u08fb\u08fc\u08fd\u08fe\u08ff\ufbb2\ufbb3\ufbb4\ufbb5\ufbb6\ufbb7\ufbb8\ufbb9\ufbba\ufbbb\ufbbc\ufbbd\ufbbe\ufbbf\ufbc0\ufbc1\ufc5e\ufc5f\ufc60\ufc61\ufc62\ufc63\ufcf2\ufcf3\ufcf4\ufd3e\ufd3f\ufe70\ufe71\ufe72\ufe76\ufe77\ufe78\ufe79\ufe7a\ufe7b\ufe7c\ufe7d\ufe7e\ufe7f\ufdfa\ufdfb]",
                    "",
                ),
            ]
        """برخی از کاراکترها و نشانه‌های خاص را که کاربردی در پردازش متن ندارند حذف
        می‌کند.

        Examples:
            >>> normalizer = Normalizer()
            >>> normalizer.remove_specials_chars('پیامبر اکرم ﷺ')
            'پیامبر اکرم '

        Args:
            text: متنی که باید کاراکترها و نشانه‌های اضافهٔ آن حذف شود.

        Returns:
            متنی بدون کاراکترها و نشانه‌های اضافه.

        """
        for pattern, repl in specials_chars_patterns:
            text = re.sub(pattern, repl, text)
        return text
@udf
def persian_number(text: str) -> str:
        a = "0123456789%٠١٢٣٤٥٦٧٨٩"
        b = "۰۱۲۳۴۵۶۷۸۹٪۰۱۲۳۴۵۶۷۸۹"
        """اعداد لاتین و علامت % را با معادل فارسی آن جایگزین می‌کند.

        Examples:
            >>> normalizer = Normalizer()
            >>> normalizer.persian_number('5 درصد')
            '۵ درصد'
            >>> normalizer.persian_number('۵ درصد')
            '۵ درصد'
            >>> normalizer.persian_number('')
            ''

        Args:
            text: متنی که باید اعداد لاتین و علامت % آن با معادل فارسی جایگزین شود.

        Returns:
            متنی با اعداد و علامت ٪ فارسی.

        """
        
        translations = {ord(a): b for a, b in zip(a, b)}
        return text.translate(translations)

@udf
def unicodes_replacement(text: str) -> str:
        replacements = [
                ("﷽", "بسم الله الرحمن الرحیم"),
                ("﷼", "ریال"),
                ("(ﷰ|ﷹ)", "صلی"),
                ("ﷲ", "الله"),
                ("ﷳ", "اکبر"),
                ("ﷴ", "محمد"),
                ("ﷵ", "صلعم"),
                ("ﷶ", "رسول"),
                ("ﷷ", "علیه"),
                ("ﷸ", "وسلم"),
                ("ﻵ|ﻶ|ﻷ|ﻸ|ﻹ|ﻺ|ﻻ|ﻼ", "لا"),
            ]
        """برخی از کاراکترهای خاص یونیکد را با معادلِ نرمال آن جایگزین می‌کند. غالباً
        این کار فقط در مواردی صورت می‌گیرد که یک کلمه در قالب یک کاراکتر یونیکد تعریف
        شده است.

        **فهرست این کاراکترها و نسخهٔ جایگزین آن:**

        |کاراکتر|نسخهٔ جایگزین|
        |--------|------------------|
        |﷽|بسم الله الرحمن الرحیم|
        |﷼|ریال|
        |ﷰ، ﷹ|صلی|
        |ﷲ|الله|
        |ﷳ|اکبر|
        |ﷴ|محمد|
        |ﷵ|صلعم|
        |ﷶ|رسول|
        |ﷷ|علیه|
        |ﷸ|وسلم|
        |ﻵ، ﻶ، ﻷ، ﻸ، ﻹ، ﻺ، ﻻ، ﻼ|لا|

        Examples:
            >>> normalizer = Normalizer()
            >>> normalizer.remove_specials_chars('پیامبر اکرم ﷺ')
            'پیامبر اکرم '
            >>> normalizer.remove_specials_chars('')
            ''

        Args:
            text: متنی که باید برخی از کاراکترهای یونیکد آن (جدول بالا)، با شکل استاندارد، جایگزین شود.

        Returns:
            متنی که برخی از کاراکترهای یونیکد آن با شکل استاندارد جایگزین شده است.

        """
        for old, new in replacements:
            text = re.sub(old, new, text)

        return text
@udf
def seperate_mi(text: str) -> str:
        """پیشوند «می» و «نمی» را در افعال جدا کرده و با نیم‌فاصله می‌چسباند.

        Examples:
            >>> normalizer = Normalizer()
            >>> normalizer.seperate_mi('نمیدانم چه میگفت')
            'نمی‌دانم چه می‌گفت'
            >>> normalizer.seperate_mi('میز')
            'میز'
            >>> normalizer.seperate_mi('')
            ''


        Args:
            text: متنی که باید پیشوند «می» و «نمی» در آن جدا شود.

        Returns:
            متنی با «می» و «نمی» جدا شده.

        """
        verbs = Lemmatizer(joined_verb_parts=False).verbs
        matches = re.findall(r"\bن?می[آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی]+", text)
        for m in matches:
            r = re.sub("^(ن?می)", r"\1‌", m)
            if r in verbs:
                text = text.replace(m, r)

        return text

@udf
def decrease_repeated_chars(text: str) -> str:
        """تکرارهای زائد حروف را در کلماتی مثل سلامممممم حذف می‌کند و در مواردی که
        نمی‌تواند تشخیص دهد دست کم به دو تکرار کاهش می‌دهد.

        Examples:
            >>> normalizer = Normalizer()
            >>> normalizer.decrease_repeated_chars('سلامممم به همه')
            'سلام به همه'
            >>> normalizer.decrease_repeated_chars('سلامم به همه')
            'سلامم به همه'
            >>> normalizer.decrease_repeated_chars('سلامم را برسان')
            'سلامم را برسان'
            >>> normalizer.decrease_repeated_chars('سلاممم را برسان')
            'سلام را برسان'
            >>> normalizer.decrease_repeated_chars('')
            ''

        Args:
            text: متنی که باید تکرارهای زائد آن حذف شود.

        Returns:
            متنی بدون کاراکترهای زائد یا حداقل با دو تکرار.

        """
        more_than_two_repeat_pattern = (
                r"([آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی])\1{2,}"
            )
        repeated_chars_pattern = (
            r"[آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی]*"
            + more_than_two_repeat_pattern
            + "[آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی]*"
        )
        
        tokenizer = WordTokenizer(join_verb_parts=False)
        words = tokenizer.words
        

        matches = re.finditer(repeated_chars_pattern, text)

        for m in matches:
            word = m.group()
            if word not in words:
                no_repeat = re.sub(more_than_two_repeat_pattern, r"\1", word)
                two_repeat = re.sub(more_than_two_repeat_pattern, r"\1\1", word)

                if (no_repeat in words) != (two_repeat in words):
                    r = no_repeat if no_repeat in words else two_repeat
                    text = text.replace(word, r)
                else:
                    text = text.replace(word, two_repeat)

        return text


@udf
def correct_spacing(text: str) -> str:
        """فاصله‌گذاری‌ها را در پیشوندها و پسوندها اصلاح می‌کند.

        Examples:
            >>> normalizer = Normalizer()
            >>> normalizer.correct_spacing("سلام   دنیا")
            'سلام دنیا'
            >>> normalizer.correct_spacing("به طول ۹متر و عرض۶")
            'به طول ۹ متر و عرض ۶'
            >>> normalizer.correct_spacing("کاروان‌‌سرا")
            'کاروان‌سرا'
            >>> normalizer.correct_spacing("‌سلام‌ به ‌همه‌")
            'سلام به همه'
            >>> normalizer.correct_spacing("سلام دنیـــا")
            'سلام دنیا'
            >>> normalizer.correct_spacing("جمعهها که کار نمی کنم مطالعه می کنم")
            'جمعه‌ها که کار نمی‌کنم مطالعه می‌کنم'
            >>> normalizer.correct_spacing(' "سلام به همه"   ')
            '"سلام به همه"'
            >>> normalizer.correct_spacing('')
            ''

        Args:
            text (str): متنی که باید فاصله‌گذاری‌های آن اصلاح شود.

        Returns:
            (str): متنی با فاصله‌گذاری‌های اصلاح‌شده.


        """

        extra_space_patterns = [
                (r" {2,}", " "),  # remove extra spaces
                (r"\n{3,}", "\n\n"),  # remove extra newlines
                (r"\u200c{2,}", "\u200c"),  # remove extra ZWNJs
                (r"\u200c{1,} ", " "),  # remove unneded ZWNJs before space
                (r" \u200c{1,}", " "),  # remove unneded ZWNJs after space
                (r"\b\u200c*\B", ""),  # remove unneded ZWNJs at the beginning of words
                (r"\B\u200c*\b", ""),  # remove unneded ZWNJs at the end of words
                (r"[ـ\r]", ""),  # remove keshide, carriage returns
            ]
        
        punc_after, punc_before = r"\.:!،؛؟»\]\)\}", r"«\[\(\{"

        affix_spacing_patterns = [
                (r"([^ ]ه) ی ", r"\1‌ی "),  # fix ی space
                (r"(^| )(ن?می) ", r"\1\2‌"),  # put zwnj after می, نمی
                # put zwnj before تر, تری, ترین, گر, گری, ها, های
                (
                    r"(?<=[^\n\d "
                    + punc_after
                    + punc_before
                    + "]{2}) (تر(ین?)?|گری?|های?)(?=[ \n"
                    + punc_after
                    + punc_before
                    + "]|$)",
                    r"‌\1",
                ),
                # join ام, ایم, اش, اند, ای, اید, ات
                (
                    r"([^ ]ه) (ا(م|یم|ش|ند|ی|ید|ت))(?=[ \n" + punc_after + "]|$)",
                    r"\1‌\2",
                ),
                # شنبهها => شنبه‌ها
                ("(ه)(ها)", r"\1‌\2"),
            ]
        
        punctuation_spacing_patterns = [
                # remove space before and after quotation
                ('" ([^\n"]+) "', r'"\1"'),
                (" ([" + punc_after + "])", r"\1"),  # remove space before
                ("([" + punc_before + "]) ", r"\1"),  # remove space after
                # put space after . and :
                (
                    "([" + punc_after[:3] + "])([^ " + punc_after + r"\d۰۱۲۳۴۵۶۷۸۹])",
                    r"\1 \2",
                ),
                (
                    "([" + punc_after[3:] + "])([^ " + punc_after + "])",
                    r"\1 \2",
                ),  # put space after
                (
                    "([^ " + punc_before + "])([" + punc_before + "])",
                    r"\1 \2",
                ),  # put space before
                # put space after number; e.g., به طول ۹متر -> به طول ۹ متر
                (r"(\d)([آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی])", r"\1 \2"),
                # put space after number; e.g., به طول۹ -> به طول ۹
                (r"([آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی])(\d)", r"\1 \2"),
            ]
        for pattern, repl in extra_space_patterns:
            text = re.sub(pattern, repl, text)
            
        tokenizer = WordTokenizer(join_verb_parts=False)

        lines = text.split("\n")
        result = []
        for line in lines:
            tokens = tokenizer.tokenize(line)
            spaced_tokens = token_spacing(tokens)
            line = " ".join(spaced_tokens)
            result.append(line)

        text = "\n".join(result)

        for pattern, repl in affix_spacing_patterns:
            text = re.sub(pattern, repl, text)

        for pattern, repl in punctuation_spacing_patterns:
            text = re.sub(pattern, repl, text)

        return text



def normalize(
            df: pyspark.sql.dataframe.DataFrame, 
            colname: str,
            correct_spacing_en: bool = True,
            remove_diacritics_en: bool = True,
            remove_specials_chars_en: bool = True,
            decrease_repeated_chars_en: bool = True,
            persian_style_en: bool = True,
            persian_number_en: bool = True,
            unicodes_replacement_en: bool = True,
            seperate_mi_en: bool = True,) -> pyspark.sql.dataframe.DataFrame:

    new_col_name="normalize_"+colname
    df = df.withColumn(new_col_name, col(colname))

    if persian_style_en:
      df = df.select(*[persian_style(new_col_name).alias(new_col_name) if column == new_col_name else column for column in df.columns])

    if persian_number_en:
        df = df.select(*[persian_number(new_col_name).alias(new_col_name) if column == new_col_name else column for column in df.columns])

    if unicodes_replacement_en:
        df = df.select(*[unicodes_replacement(new_col_name).alias(new_col_name) if column == new_col_name else column for column in df.columns])

    if remove_specials_chars_en:
        df = df.select(*[persian_style(new_col_name).alias(new_col_name) if column == new_col_name else column for column in df.columns])

    if remove_diacritics_en:
        df = df.select(*[remove_diacritics(new_col_name).alias(new_col_name) if column == new_col_name else column for column in df.columns])

    if seperate_mi_en:
        df = df.select(*[seperate_mi(new_col_name).alias(new_col_name) if column == new_col_name else column for column in df.columns])

    if correct_spacing_en:
        df = df.select(*[correct_spacing(new_col_name).alias(new_col_name) if column == new_col_name else column for column in df.columns])

    if decrease_repeated_chars_en:
        df = df.select(*[decrease_repeated_chars(new_col_name).alias(new_col_name) if column == new_col_name else column for column in df.columns])

    return df

if __name__ == "__main__":
    # Creating a spark session
    spark_session = SparkSession.builder.appName(
        'Spark_Session').getOrCreate()

    df = spark_session.createDataFrame([
                                        ('﷽',1),
                                        ("5 درصد حَذفِ اِعراب",2),
                                        ("نمیکنم",3),
                                        ("سلامممممم",4),
                                        ("سلام   دنیا",5)], ("text",'id'))
    normalize_df = normalize(df, 'text')

    normalize_df.select("text").show()
    normalize_df.select("normalize_text").show()