from fastwarc.warc import ArchiveIterator, WarcRecordType

from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
import fasttext
import re


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
    nsfw = predictions[0][0].replace("__label__", "")
    confidence = predictions[1][0]
    return (nsfw, confidence)
