from abc import ABC, abstractmethod

from fastwarc.warc import ArchiveIterator, WarcRecordType

from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
import fasttext
import re
import nltk
from collections import Counter
import os


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
        alpha_words: int = sum(1 for word in words if any(c.isalpha() for c in word))
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


def quality_classify(text: str) -> tuple[bool, float]:
    """
    Classify text quality using a trained fastText-style classifier.

    Args:
        text: Input text to classify
        model_path: Path to the trained model .pt file

    Returns:
        Probability that the text is high quality (positive class)
    """
    import torch
    from cs336_data.quality_classifier import FastTextClassifier, Vocabulary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "quality_classifier.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    config = checkpoint["config"]
    model = FastTextClassifier(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_dims=config["hidden_dims"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Reconstruct vocabulary from saved data
    vocab = Vocabulary(min_freq=checkpoint.get("vocab_min_freq", 1))
    vocab.word2idx = checkpoint["vocab_word2idx"]
    vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}

    token_ids = vocab.encode(text)
    if not token_ids:
        token_ids = [vocab.word2idx["<UNK>"]]

    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).to(device)
    offsets_tensor = torch.tensor([0], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(token_ids_tensor, offsets_tensor)
        prob = torch.sigmoid(logits).item()
    prediction = "cc" if prob < 0.5 else "wiki"
    # print(f"tlu7. ... prediction:', {prediction}, 'prob:', {prob}, 'text:', {text[:100]}")
    return (prediction, prob)


def _hash_line(line: str) -> int:
    """Compute a hash of a line for memory-efficient deduplication."""
    import hashlib

    return int(hashlib.sha256(line.encode("utf-8")).hexdigest(), 16)


def exact_deduplication(input_file_paths: list[str], output_dir: str) -> None:
    """
    if the input paths are a/1.txt and
    a/2.txt, and the output directory is b/, your function should write the files b/1.txt and b/2.txt.

    First, make one pass through the corpus to count how many occurrences of each line we observe.
    Then, in a second pass, we can rewrite each document by preserving only its unique lines.

    e.g. input
    1.txt: "line_a \n line_b"
    2.txt: "line_c \n line_b \n line_x"

    write to
    <output_dir>/1.txt: "line_a"
    <output_dir>/2.txt: "line_c \n line_x"

    Memory optimization: Uses hash of line as key instead of full line string.
    """
    # Pass 1: Count occurrences using hash as key
    line_hash_counter: Counter[int] = Counter()
    for path in input_file_paths:
        with open(path) as f:
            for line in f:
                line_hash_counter[_hash_line(line)] += 1

    # Pass 2: Write only lines that appear exactly once globally
    for path in input_file_paths:
        output_path = os.path.join(output_dir, os.path.basename(path))
        with open(path) as f, open(output_path, "w") as out_f:
            for line in f:
                if line_hash_counter[_hash_line(line)] == 1:
                    out_f.write(line)


def minhash_deduplication(
    input_file_paths: list[str],
    num_hash: int,
    num_band: int,
    n_gram_len: int,
    output_dir: str,
    jaccard_threshold: float = 0.8,
) -> None:
    """
    Steps:

    minhashing:

    - Represent each document as a set S of n-grams (e.g. 3-grams).
    - Run a hash function on each n-gram to get a hash value.
    - minhash(hi, S) = min{h(i, x) | x in S}
    - document representation: minhash values from each hash function; [minhash(h1, S), minhash(h2, S), ... , minhash(hk, S)]

    banding:

    - (Can assume num_hash is divisible by num_band (e.g. 50 and 10))
    - Split the minhash values into num_band bands of equal size.
    - build cluster: if 2 docs have the same band value for any band, they are in the same cluster.

    actual jaccard deduplication:
    - for documents in the same cluster, compute the jaccard similarity between them.
    - if jaccard similarity is above a threshold, keep only one
    """
    import hashlib
    from collections import defaultdict

    # Large prime for hash functions
    PRIME = 2**61 - 1
    MAX_HASH = 2**32

    # Generate random hash function parameters (a, b) for each hash function
    import random

    random.seed(42)  # For reproducibility
    hash_params = [(random.randint(1, PRIME - 1), random.randint(0, PRIME - 1)) for _ in range(num_hash)]

    def get_ngrams(text: str, n: int) -> set[str]:
        """Extract character n-grams from text."""
        text = text.lower().replace("\n", " ")
        if len(text) < n:
            return {text} if text else set()
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    def hash_ngram(ngram: str) -> int:
        """Hash an n-gram to an integer."""
        return int(hashlib.md5(ngram.encode("utf-8")).hexdigest(), 16) % MAX_HASH

    def compute_minhash(ngrams: set[str], hash_params: list[tuple[int, int]]) -> list[int]:
        """Compute minhash signature for a set of n-grams."""
        if not ngrams:
            return [MAX_HASH] * len(hash_params)

        # Hash all n-grams once
        ngram_hashes = [hash_ngram(ng) for ng in ngrams]

        signature = []
        for a, b in hash_params:
            # h(x) = (a * x + b) % PRIME
            min_hash = min((a * h + b) % PRIME for h in ngram_hashes)
            signature.append(min_hash)
        return signature

    def get_bands(signature: list[int], num_band: int) -> list[tuple[int, ...]]:
        """Split signature into bands."""
        band_size = len(signature) // num_band
        return [tuple(signature[i * band_size : (i + 1) * band_size]) for i in range(num_band)]

    def jaccard_similarity(set1: set[str], set2: set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union

    # Step 1: Read documents and compute minhash signatures
    doc_data: dict[str, tuple[str, set[str], list[int]]] = {}  # path -> (content, ngrams, signature)

    for path in input_file_paths:
        with open(path) as f:
            content = f.read()
        ngrams = get_ngrams(content, n_gram_len)
        signature = compute_minhash(ngrams, hash_params)
        doc_data[path] = (content, ngrams, signature)

    # Step 2: LSH banding - group documents by band values
    band_buckets: dict[tuple[int, tuple[int, ...]], set[str]] = defaultdict(
        set
    )  # (band_idx, band_value) -> set of paths

    for path, (content, ngrams, signature) in doc_data.items():
        bands = get_bands(signature, num_band)
        for band_idx, band_value in enumerate(bands):
            band_buckets[(band_idx, band_value)].add(path)

    # Step 3: Find candidate pairs (documents that share at least one band)
    candidate_pairs: set[tuple[str, str]] = set()
    for bucket_paths in band_buckets.values():
        if len(bucket_paths) > 1:
            paths_list = list(bucket_paths)
            for i in range(len(paths_list)):
                for j in range(i + 1, len(paths_list)):
                    pair = tuple(sorted([paths_list[i], paths_list[j]]))
                    candidate_pairs.add(pair)
    print("tlu7... candidate_pairs:", candidate_pairs)

    # Step 4: Compute actual Jaccard similarity for candidates and build clusters
    # Use Union-Find to group duplicates
    parent: dict[str, str] = {path: path for path in input_file_paths}

    def find(x: str) -> str:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for path1, path2 in candidate_pairs:
        ngrams1 = doc_data[path1][1]
        ngrams2 = doc_data[path2][1]
        similarity = jaccard_similarity(ngrams1, ngrams2)
        if similarity >= jaccard_threshold:
            union(path1, path2)

    # Step 5: Keep only one document per cluster (the representative)
    clusters: dict[str, list[str]] = defaultdict(list)
    for path in input_file_paths:
        root = find(path)
        clusters[root].append(path)

    # Keep the first document in each cluster (by sorted order for determinism)
    docs_to_keep: set[str] = set()
    for cluster_paths in clusters.values():
        cluster_paths.sort()
        docs_to_keep.add(cluster_paths[0])
    print("tlu7... docs_to_keep:", docs_to_keep)

    # Step 6: Write output files
    os.makedirs(output_dir, exist_ok=True)
    for path in input_file_paths:
        output_path = os.path.join(output_dir, os.path.basename(path))
        if path in docs_to_keep:
            # Keep the document
            with open(path) as f, open(output_path, "w") as out_f:
                out_f.write(f.read())
