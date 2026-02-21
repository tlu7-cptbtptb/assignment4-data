from fastwarc.warc import ArchiveIterator, WarcRecordType

from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text


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
