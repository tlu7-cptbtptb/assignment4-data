import argparse
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType

from utils import extract_text_from_byte_string

"""
Tester for the functions in utils.py
"""


def process_warc_file(warc_path: str, max_records: int = 10):
    """Process a WARC file and extract text from HTML responses."""
    warc_path = Path(warc_path)
    if not warc_path.exists():
        print(f"Error: File not found: {warc_path}")
        return

    print(f"Processing WARC file: {warc_path}")
    record_count = 0

    with open(warc_path, "rb") as f:
        for record in ArchiveIterator(f):
            if record.record_type == WarcRecordType.response:
                content = record.reader.read()
                text = extract_text_from_byte_string(content)

                print(f"\n{'=' * 60}")
                print(f"Record {record_count + 1}")
                print(f"URL: {record.headers.get('WARC-Target-URI', 'N/A')}")
                print(f"Content length: {len(content)} bytes")
                print("Extracted text preview (first 500 chars):")
                print(text[:500] if text else "(empty)")

                record_count += 1
                if record_count >= max_records:
                    print(f"\n...stopped after {max_records} records")
                    break

    print(f"\nTotal records processed: {record_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process WARC files and extract text")
    parser.add_argument("warc_file", nargs="?", help="Path to WARC file (.warc.gz)")
    parser.add_argument(
        "--max-records", "-n", type=int, default=10, help="Maximum number of records to process (default: 10)"
    )

    args = parser.parse_args()

    if args.warc_file:
        process_warc_file(args.warc_file, args.max_records)
    else:
        # Fallback to simple test
        html_bytes = b"<html><body><p>hello world</p></body></html>"
        text = extract_text_from_byte_string(html_bytes)
        print(f"Extracted text from HTML bytes: {text}")
