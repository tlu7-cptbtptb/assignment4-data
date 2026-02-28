"""
Quality Classifier using FastText-style architecture.

Architecture: whitespace tokenization -> embedding lookup -> mean pooling -> MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter


class Vocabulary:
    """Simple vocabulary for whitespace tokenization."""

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.word2idx: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word: dict[int, str] = {0: "<PAD>", 1: "<UNK>"}

    def build_from_texts(self, texts: list[str]) -> None:
        """Build vocabulary from training texts using whitespace tokenization."""
        counter: Counter[str] = Counter()
        for text in texts:
            tokens = text.lower().split()
            counter.update(tokens)

        for word, freq in counter.items():
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text: str) -> list[int]:
        """Convert text to list of token indices."""
        tokens = text.lower().split()
        return [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]

    def __len__(self) -> int:
        return len(self.word2idx)


class FastTextClassifier(nn.Module):
    """
    Lightweight fastText-style classifier for binary quality classification.
    Architecture: token embeddings -> mean pooling -> MLP -> output
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dims: list[int] = [64, 32],
    ):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, token_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: Flattened token indices for batch
            offsets: Starting index of each sample in token_ids
        Returns:
            logits: (batch_size, 1) - raw logits for binary classification
        """
        embedded = self.embedding(token_ids, offsets)
        return self.fc(embedded)


class QualityDataset(Dataset):
    """Dataset for fastText-format quality classification data."""

    def __init__(self, file_path: str, vocab: Vocabulary):
        self.samples: list[tuple[str, int]] = []
        self.vocab = vocab

        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("__label__positive"):
                    label = 1
                    text = line[len("__label__positive") :].strip()
                elif line.startswith("__label__negative"):
                    label = 0
                    text = line[len("__label__negative") :].strip()
                else:
                    continue
                if text:
                    self.samples.append((text, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        text, label = self.samples[idx]
        token_ids = self.vocab.encode(text)
        if not token_ids:
            token_ids = [self.vocab.word2idx["<UNK>"]]
        return token_ids, label


def collate_fn(
    batch: list[tuple[list[int], int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for variable-length sequences with EmbeddingBag."""
    token_ids_list, labels = zip(*batch)

    offsets = [0]
    flat_ids = []
    for ids in token_ids_list:
        flat_ids.extend(ids)
        offsets.append(len(flat_ids))
    offsets = offsets[:-1]

    return (
        torch.tensor(flat_ids, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float),
    )


class QualityClassifierTrainer:
    """Trainer class for the quality classifier."""

    def __init__(
        self,
        train_path: str,
        val_path: str,
        embed_dim: int = 100,
        hidden_dims: int = [64, 32],
        batch_size: int = 32,
        min_word_freq: int = 2,
        pos_weight: float = 5.0,
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.min_word_freq = min_word_freq
        self.pos_weight = pos_weight

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab: Vocabulary | None = None
        self.model: FastTextClassifier | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None

    def _build_vocabulary(self) -> None:
        """Build vocabulary from training data."""
        self.vocab = Vocabulary(min_freq=self.min_word_freq)
        train_texts = []

        with open(self.train_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("__label__positive"):
                    text = line[len("__label__positive") :].strip()
                elif line.startswith("__label__negative"):
                    text = line[len("__label__negative") :].strip()
                else:
                    continue
                if text:
                    train_texts.append(text)

        self.vocab.build_from_texts(train_texts)
        print(f"Vocabulary size: {len(self.vocab)}")

    def _create_dataloaders(self) -> None:
        """Create train and validation dataloaders."""
        assert self.vocab is not None

        train_dataset = QualityDataset(self.train_path, self.vocab)
        val_dataset = QualityDataset(self.val_path, self.vocab)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def _init_model(self) -> None:
        """Initialize the model."""
        assert self.vocab is not None

        self.model = FastTextClassifier(
            vocab_size=len(self.vocab),
            embed_dim=self.embed_dim,
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")

    def _compute_val_metrics(self) -> tuple[float, float, float, float]:
        """Compute validation BCE loss, precision, recall, and F1."""
        assert self.model is not None
        assert self.val_loader is not None

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # For precision/recall calculation
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        with torch.no_grad():
            for token_ids, offsets, labels in self.val_loader:
                token_ids = token_ids.to(self.device)
                offsets = offsets.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(token_ids, offsets).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                total_loss += loss.item()
                num_batches += 1

                # Compute predictions (threshold at 0.5)
                preds = (torch.sigmoid(logits) >= 0.5).float()

                # Update counts for precision/recall
                true_positives += ((preds == 1) & (labels == 1)).sum().item()
                false_positives += ((preds == 1) & (labels == 0)).sum().item()
                false_negatives += ((preds == 0) & (labels == 1)).sum().item()

        self.model.train()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        precision = (
            true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        )
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return avg_loss, precision, recall, f1

    def train(self, epochs: int = 10, save_path: str = "quality_classifier.pt") -> FastTextClassifier:
        """
        Train the quality classifier.

        Args:
            epochs: Number of training epochs
            save_path: Path to save the trained model

        Returns:
            Trained FastTextClassifier model
        """
        self._build_vocabulary()
        self._create_dataloaders()
        self._init_model()

        assert self.model is not None
        assert self.train_loader is not None
        assert self.vocab is not None

        optimizer = torch.optim.Adam(self.model.parameters())

        global_step = 0
        print(f"\nTraining on {self.device}")
        print("-" * 50)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for token_ids, offsets, labels in self.train_loader:
                token_ids = token_ids.to(self.device)
                offsets = offsets.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(token_ids, offsets).squeeze(-1)
                # Apply higher weight to positive samples (label=1)
                weights = torch.where(
                    labels == 1.0,
                    torch.tensor(self.pos_weight, device=self.device),
                    torch.tensor(1.0, device=self.device),
                )
                loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % 20 == 0:
                    val_loss, precision, recall, f1 = self._compute_val_metrics()
                    print(
                        f"Step {global_step}: Val BCE = {val_loss:.4f}, P = {precision:.4f}, R = {recall:.4f}, F1 = {f1:.4f}"
                    )

            avg_train_loss = epoch_loss / len(self.train_loader)
            val_loss, precision, recall, f1 = self._compute_val_metrics()
            print(
                f"Epoch {epoch + 1}/{epochs} complete - Train BCE: {avg_train_loss:.4f}, Val BCE: {val_loss:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}"
            )

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "vocab_word2idx": self.vocab.word2idx,
                "vocab_min_freq": self.vocab.min_freq,
                "config": {
                    "embed_dim": self.embed_dim,
                    "hidden_dims": self.hidden_dims,
                    "vocab_size": len(self.vocab),
                },
            },
            save_path,
        )
        print(f"\nModel saved to {save_path}")

        return self.model


def load_classifier(model_path: str, device: torch.device | None = None) -> tuple[FastTextClassifier, Vocabulary]:
    """Load a trained classifier from disk."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    config = checkpoint["config"]
    model = FastTextClassifier(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Reconstruct vocabulary from saved data
    vocab = Vocabulary(min_freq=checkpoint.get("vocab_min_freq", 1))
    vocab.word2idx = checkpoint["vocab_word2idx"]
    vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}

    return model, vocab


def predict_quality(
    text: str,
    model: FastTextClassifier,
    vocab: Vocabulary,
    device: torch.device | None = None,
) -> tuple[str, float]:
    """
    Predict quality label for a single text.

    Args:
        text: Input text to classify
        model: Trained FastTextClassifier
        vocab: Vocabulary used during training
        device: Device to run inference on

    Returns:
        (label, probability) tuple where label is "positive" or "negative"
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    token_ids = vocab.encode(text)
    if not token_ids:
        token_ids = [vocab.word2idx["<UNK>"]]

    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).to(device)
    offsets_tensor = torch.tensor([0], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(token_ids_tensor, offsets_tensor)
        prob = torch.sigmoid(logits).item()

    label = "positive" if prob >= 0.5 else "negative"
    confidence = prob if prob >= 0.5 else 1 - prob

    return label, confidence


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train quality classifier")
    parser.add_argument("--train", default="quality_classifier_train.txt")
    parser.add_argument("--val", default="quality_classifier_validation.txt")
    parser.add_argument("--embed-dim", type=int, default=100)
    # parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--output", default="quality_classifier.pt")

    args = parser.parse_args()

    trainer = QualityClassifierTrainer(
        train_path=args.train,
        val_path=args.val,
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        min_word_freq=args.min_freq,
    )

    trainer.train(epochs=args.epochs, save_path=args.output)
