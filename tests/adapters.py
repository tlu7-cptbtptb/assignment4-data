from __future__ import annotations

import os
from typing import Any
from cs336_data.utils import (
    extract_text_from_byte_string,
    detect_main_language,
    mask_email_address,
    mask_phone_number,
    mask_ip_address,
    detect_nsfw,
    detect_toxic,
    gopher_quality_filters,
    quality_classify,
    exact_deduplication,
    minhash_deduplication,
)


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text_from_byte_string(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    lang, confidence = detect_main_language(text)
    return lang, confidence


def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_email_address(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone_number(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return mask_ip_address(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return detect_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return detect_toxic(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    return quality_classify(text)


def run_gopher_quality_filter(text: str) -> bool:
    return not gopher_quality_filters(text)


def run_exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike):
    exact_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    minhash_deduplication(
        input_file_paths=input_files,
        num_hash=num_hashes,
        num_band=num_bands,
        n_gram_len=ngrams,
        jaccard_threshold=jaccard_threshold,
        output_dir=output_directory,
    )
