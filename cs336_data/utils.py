from abc import ABC, abstractmethod

from fastwarc.warc import ArchiveIterator, WarcRecordType

from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
import fasttext
import re
import nltk


def extract_text_from_byte_string(input: bytes) -> str | None:
    if not input:
        return ""
    output = None
    # Try UTF-8 first (most common encoding)
    try:
        output = input.decode("utf-8")
    except UnicodeDecodeError:
        pass

    # Detect encoding and try that
    encoding = detect_encoding(input)
    if encoding:
        try:
            output = input.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            pass
    if output is None:
        return None
    # now output is some string, we start the extraction

    return extract_plain_text(output)


def detect_main_language(text: str) -> tuple[str, float]:
    model = fasttext.load_model("lid.176.bin")
    # fasttext predict requires single line input - replace newlines with spaces
    text_clean = text.replace("\n", " ").strip()
    predictions = model.predict(text_clean)
    # process '__label__zh' -> 'zh' with regex
    language = predictions[0][0].replace("__label__", "")
    confidence = predictions[1][0]
    return (language, confidence)


def mask_email_address(text: str) -> tuple[str, int]:
    """
    Replace email addresses with a placeholder |||EMAIL_ADDRESS|||
    Return the masked string and the number of email addresses masked
    """
    email_regex = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    masked_text = re.sub(email_regex, "|||EMAIL_ADDRESS|||", text)
    return masked_text, len(re.findall(email_regex, text))


def mask_phone_number(text: str) -> tuple[str, int]:
    """
    Replace phone numbers with a placeholder |||PHONE_NUMBER|||
    Return the masked string and the number of email addresses masked
    """
    phone_regex = r"(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"
    masked_text = re.sub(phone_regex, "|||PHONE_NUMBER|||", text)
    return masked_text, len(re.findall(phone_regex, text))


def mask_ip_address(text: str) -> tuple[str, int]:
    """
    Replace phone numbers with a placeholder |||IP_ADDRESS|||
    Return the masked string and the number of ip addresses masked
    """
    # IPv4 address pattern; 4 numbers up to 255 separated by points
    ipv4_pattern = r"\b((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.?\b){4}\b"
    masked_text = re.sub(ipv4_pattern, "|||IP_ADDRESS|||", text)
    return masked_text, len(re.findall(ipv4_pattern, text))


def mask_pii(text: str) -> str:
    """
    Mask PII in a string
    """
    masked_text = text
    masked_text, num_emails = mask_email_address(masked_text)
    masked_text, num_phones = mask_phone_number(masked_text)
    masked_text, num_ips = mask_ip_address(masked_text)
    return masked_text


def detect_nsfw(text: str):
    """
    Detect if a text contains NSFW content
    """
    model = fasttext.load_model("jigsaw_fasttext_bigrams_nsfw_final.bin")
    # fasttext predict requires single line input - replace newlines with spaces
    text_clean = text.replace("\n", " ").strip()
    predictions = model.predict(text_clean)
    nsfw = predictions[0][0].replace("__label__", "")
    confidence = predictions[1][0]
    return (nsfw, confidence)


def detect_toxic(text: str):
    """
    Detect if a text contains NSFW content
    """
    model = fasttext.load_model("jigsaw_fasttext_bigrams_hatespeech_final.bin")
    # fasttext predict requires single line input - replace newlines with spaces
    text_clean = text.replace("\n", " ").strip()
    predictions = model.predict(text_clean)
    toxic = predictions[0][0].replace("__label__", "")
    confidence = predictions[1][0]
    return (toxic, confidence)


class QualityFilter(ABC):
    """Abstract base class for quality filters using Chain of Responsibility pattern."""

    def __init__(self):
        self._next_filter: QualityFilter | None = None

    def set_next(self, filter: "QualityFilter") -> "QualityFilter":
        """Set the next filter in the chain and return it for chaining."""
        self._next_filter = filter
        return filter

    def handle(self, text: str) -> bool:
        """
        Process the text through this filter and the chain.
        Returns True if text should be FILTERED OUT (fails quality check).
        """
        if self.should_filter(text):
            return True
        if self._next_filter:
            return self._next_filter.handle(text)
        return False

    @abstractmethod
    def should_filter(self, text: str) -> bool:
        """Return True if text fails this quality check and should be filtered out."""
        pass


class WordCountFilter(QualityFilter):
    """Filter texts with less than 50 or more than 100,000 words."""

    def should_filter(self, text: str) -> bool:
        words = text.split()
        word_count = len(words)
        return word_count < 50 or word_count > 100_000


class MeanWordLengthFilter(QualityFilter):
    """Filter texts with mean word length outside the range of 3 to 10 characters."""

    def should_filter(self, text: str) -> bool:
        words = text.split()
        if not words:
            return True
        mean_length = sum(len(word) for word in words) / len(words)
        return mean_length < 3 or mean_length > 10


class EllipsisFilter(QualityFilter):
    """Filter texts with more than 30% of lines ending with an ellipsis."""

    def should_filter(self, text: str) -> bool:
        lines = text.split("\n")
        if not lines:
            return False
        ellipsis_count = sum(1 for line in lines if line.rstrip().endswith("..."))
        return (ellipsis_count / len(lines)) > 0.3


class AlphabeticWordFilter(QualityFilter):
    """Filter texts with less than 80% of words containing at least one alphabetic character."""

    def should_filter(self, text: str) -> bool:
        words = text.split()
        if not words:
            return True
        alpha_words = sum(1 for word in words if any(c.isalpha() for c in word))
        return (alpha_words / len(words)) < 0.8


def gopher_quality_filters(text: str) -> bool:
    """
    Apply filters to a text to determine if it is suitable for Gopher.
    Uses Chain of Responsibility pattern.

    Return True if any of these conditions are met (text should be filtered out):
    • Contain less than 50 or more than 100,000 words.
    • Have a mean word length outside the range of 3 to 10 characters.
    • Have more than 30% of lines ending with an ellipsis ("...").
    • Contain less than 80% of words with at least one alphabetic character.
    """
    # Build the chain
    word_count_filter = WordCountFilter()
    mean_word_length_filter = MeanWordLengthFilter()
    ellipsis_filter = EllipsisFilter()
    alphabetic_filter = AlphabeticWordFilter()

    word_count_filter.set_next(mean_word_length_filter).set_next(ellipsis_filter).set_next(alphabetic_filter)

    # Process through the chain
    return word_count_filter.handle(text)
