"""
Quality Classifier Data Pipeline

Build training data for a fastText quality classifier using:
- Positive examples: Wikipedia pages
- Negative examples: Common Crawl pages
"""

import gzip
import random
import subprocess
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm import tqdm

from cs336_data.utils import extract_text_from_byte_string, detect_main_language, gopher_quality_filters


def truncate_to_words(text: str, max_words: int = 100) -> str:
    """Truncate text to the first max_words words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def sample_positive_urls(input_gz_path: str, num_samples: int, output_path: str) -> list[str]:
    """
    Take the first N URLs from the gzipped Wikipedia URL list.

    Args:
        input_gz_path: Path to enwiki-20240420-extracted_urls.txt.gz
        num_samples: Number of URLs to take
        output_path: Path to output sampled URLs

    Returns:
        List of sampled URLs
    """
    urls: list[str] = []

    with gzip.open(input_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if not url:
                continue

            urls.append(url)
            if len(urls) >= num_samples:
                break

    with open(output_path, "w", encoding="utf-8") as f:
        for url in urls:
            f.write(url + "\n")

    print(f"Collected {len(urls)} URLs to {output_path}")
    return urls


def scrape_urls_to_warc(urls_file: str, output_warc_prefix: str) -> str:
    """
    Use wget to scrape URLs and save as WARC file.

    Args:
        urls_file: Path to file containing URLs (one per line)
        output_warc_prefix: Prefix for output WARC file

    Returns:
        Path to the generated WARC file
    """
    with open(urls_file, encoding="utf-8") as f:
        num_urls = sum(1 for line in f if line.strip())

    print(f"Scraping {num_urls} URLs (this may take a while)...")

    cmd = [
        "wget",
        "--timeout=5",
        "-i",
        urls_file,
        f"--warc-file={output_warc_prefix}",
        "-O",
        "/dev/null",
    ]

    try:
        subprocess.run(cmd, check=False, capture_output=True)
    except subprocess.SubprocessError as e:
        print(f"Warning: wget encountered issues: {e}")

    warc_path = f"{output_warc_prefix}.warc.gz"
    print(f"Generated WARC file: {warc_path}")
    return warc_path


def extract_texts_from_warc(warc_path: str, num_samples: int) -> list[str]:
    """
    Use fastwarc to iterate through WARC records and extract plain text.
    Filters documents using gopher_quality_filters and stops when num_samples is reached.

    Args:
        warc_path: Path to WARC file (gzipped)
        num_samples: Number of valid samples to collect

    Returns:
        List of extracted text documents that pass quality filters
    """
    texts: list[str] = []
    min_text_length = 100

    with open(warc_path, "rb") as f:
        pbar = tqdm(total=num_samples, desc="Extracting positive texts")
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            if len(texts) >= num_samples:
                break

            try:
                content = record.reader.read()
                text = extract_text_from_byte_string(content)

                if not text or len(text.strip()) < min_text_length:
                    continue

                text = text.strip()

                if gopher_quality_filters(text):
                    continue

                text = truncate_to_words(text, 100)
                texts.append(text)
                pbar.update(1)
            except Exception as e:
                print(f"Error extracting text from record: {e}")
                continue
        pbar.close()

    print(f"Extracted {len(texts)} text documents from {warc_path}")
    return texts


def sample_negative_from_wet(wet_gz_path: str, num_samples: int) -> list[str]:
    """
    Parse the Common Crawl WET file and take the first N English examples.

    WET files contain already extracted plain text (no HTML extraction needed).

    Args:
        wet_gz_path: Path to Common Crawl WET file (gzipped)
        num_samples: Number of samples to collect

    Returns:
        List of text documents (filtered for English)
    """
    texts: list[str] = []
    min_text_length = 100

    with open(wet_gz_path, "rb") as f:
        pbar = tqdm(total=num_samples, desc="Sampling negative examples")
        for record in ArchiveIterator(f, record_types=WarcRecordType.conversion):
            if len(texts) >= num_samples:
                break

            try:
                content = record.reader.read()
                if not content:
                    continue

                try:
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    continue

                text = text.strip()
                if len(text) < min_text_length:
                    continue

                lang_header = record.headers.get("WARC-Identified-Content-Language", "")
                is_english = lang_header.startswith("eng") if lang_header else False

                if not is_english:
                    try:
                        lang, confidence = detect_main_language(text[:1000])
                        is_english = lang == "en" and confidence > 0.8
                    except Exception:
                        is_english = False

                if not is_english:
                    continue

                text = truncate_to_words(text, 100)
                texts.append(text)
                pbar.update(1)

            except Exception as e:
                print(f"Error processing WET record: {e}")
                continue
        pbar.close()

    print(f"Collected {len(texts)} negative examples from {wet_gz_path}")
    return texts


def build_fasttext_training_val_file(
    positive_texts: list[str],
    negative_texts: list[str],
    train_output_path: str,
    val_output_path: str,
    train_ratio: float = 0.9,
) -> None:
    """
    Create fastText training and validation files.

    Format: __label__positive <text> or __label__negative <text>
    Each document on a single line with newlines replaced by spaces.
    Splits data into 90% train, 10% validation.

    Args:
        positive_texts: List of positive (high quality) text documents
        negative_texts: List of negative (low quality) text documents
        train_output_path: Path to output training file
        val_output_path: Path to output validation file
        train_ratio: Ratio of data to use for training (default 0.9)
    """
    lines: list[str] = []

    for text in positive_texts:
        cleaned = text.replace("\n", " ").replace("\r", " ")
        cleaned = " ".join(cleaned.split())
        if cleaned:
            lines.append(f"__label__positive {cleaned}")

    for text in negative_texts:
        cleaned = text.replace("\n", " ").replace("\r", " ")
        cleaned = " ".join(cleaned.split())
        if cleaned:
            lines.append(f"__label__negative {cleaned}")

    random.shuffle(lines)

    split_idx = int(len(lines) * train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    with open(train_output_path, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")

    with open(val_output_path, "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line + "\n")

    print(f"Created training file with {len(train_lines)} examples at {train_output_path}")
    print(f"Created validation file with {len(val_lines)} examples at {val_output_path}")
    print(f"  - Total positive examples: {len(positive_texts)}")
    print(f"  - Total negative examples: {len(negative_texts)}")


def main():
    """
    Main pipeline to build quality classifier training data.
    """
    random.seed(42)

    base_dir = Path(__file__).parent.parent

    wiki_urls_gz = base_dir / "enwiki-20240420-extracted_urls.txt.gz"
    wet_file = base_dir / "CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz"
    sampled_urls_file = base_dir / "sampled_wiki_urls.txt"
    warc_prefix = base_dir / "wiki_positive"
    training_file = base_dir / "quality_classifier_train.txt"
    validation_file = base_dir / "quality_classifier_validation.txt"

    num_positive_samples = 500
    num_negative_samples = 500
    num_urls_to_scrape = 1000

    print("=" * 60)
    print("Quality Classifier Data Pipeline")
    print("=" * 60)

    print("\n[Step 1] Sampling Wikipedia URLs...")
    sample_positive_urls(str(wiki_urls_gz), num_urls_to_scrape, str(sampled_urls_file))

    print("\n[Step 2] Scraping Wikipedia content...")
    warc_path = scrape_urls_to_warc(str(sampled_urls_file), str(warc_prefix))

    print("\n[Step 3] Extracting positive texts from WARC (with quality filtering)...")
    positive_texts = extract_texts_from_warc(warc_path, num_positive_samples)

    print("\n[Step 4] Sampling negative examples from Common Crawl WET...")
    negative_texts = sample_negative_from_wet(str(wet_file), num_negative_samples)

    print("\n[Step 5] Building fastText training and validation files...")
    build_fasttext_training_val_file(positive_texts, negative_texts, str(training_file), str(validation_file))

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print("\nOutput files:")
    print(f"  - Sampled URLs: {sampled_urls_file}")
    print(f"  - WARC file: {warc_path}")
    print(f"  - data output file: train {training_file}, val {validation_file}")


if __name__ == "__main__":
    main()
