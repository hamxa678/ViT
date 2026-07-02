"""
Trainer: handles the training loop, periodic checkpointing, logging, and
evaluation with standard classification metrics.
"""

import logging
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import compute_metrics, format_metrics, setup_logger


class Trainer:
    """
    Example
    -------
    >>> model = ViT(img_size=224, num_classes=len(class_names))
    >>> trainer = Trainer(model, checkpoint_dir="checkpoints", save_every=5)
    >>> trainer.fit(train_loader, val_loader, epochs=20)
    >>> test_metrics = trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        model,
        checkpoint_dir="checkpoints",
        save_every=5,
        lr=3e-4,
        weight_decay=0.05,
        device=None,
        log_file="training.log",
        class_names=None,
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_every = save_every
        self.class_names = class_names

        self.logger = setup_logger(log_file)
        self.logger.info("Using device: %s", self.device)

        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}
        self.best_val_acc = 0.0

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def fit(self, train_loader, val_loader, epochs=20):
        for epoch in range(1, epochs + 1):
            start = time.time()

            train_loss = self._train_one_epoch(train_loader, epoch, epochs)
            val_loss, val_metrics = self._validate(val_loader)

            elapsed = time.time() - start

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_metrics["accuracy"])

            self.logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | %s | %.1fs",
                epoch, epochs, train_loss, val_loss,
                format_metrics(val_metrics), elapsed,
            )

            # Save periodic checkpoint
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, tag=f"epoch{epoch}")

            # Always track best model separately
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self._save_checkpoint(epoch, tag="best")
                self.logger.info(
                    "New best val_acc=%.4f -> saved best.pt", self.best_val_acc
                )

        # Always save final model at the end of training
        self._save_checkpoint(epochs, tag="final")
        self.logger.info("Training complete. Best val_acc=%.4f", self.best_val_acc)
        return self.history

    def _train_one_epoch(self, loader, epoch, total_epochs):
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())

        return running_loss / len(loader.dataset)

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        val_loss = running_loss / len(loader.dataset)
        metrics = compute_metrics(all_labels, all_preds, self.class_names)
        return val_loss, metrics

    @torch.no_grad()
    def evaluate(self, loader, split_name="test"):
        """Runs full evaluation and logs a detailed report (e.g. on the test set)."""
        self.model.eval()
        all_preds, all_labels = [], []

        for images, labels in tqdm(loader, desc=f"Evaluating [{split_name}]", leave=False):
            images = images.to(self.device)
            outputs = self.model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

        metrics = compute_metrics(all_labels, all_preds, self.class_names)

        self.logger.info("=== %s set evaluation ===", split_name)
        self.logger.info(format_metrics(metrics))
        self.logger.info("Confusion matrix:\n%s", metrics["confusion_matrix"])

        return metrics

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, epoch, tag):
        path = os.path.join(self.checkpoint_dir, f"{tag}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "class_names": self.class_names,
        }, path)
        self.logger.info("Saved checkpoint: %s", path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.logger.info("Loaded checkpoint from %s (epoch %d)", path, checkpoint["epoch"])
        return checkpoint
