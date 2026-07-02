"""
End-to-end example: point the framework at a data folder and train.

Usage:
    python train.py --data_dir path/to/data --epochs 20 --save_every 5
"""

import argparse

from ViT import ViT, Trainer, prepare_dataloaders


def main():
    parser = argparse.ArgumentParser(description="Train a ViT on your own image dataset.")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to folder with one subfolder per class.")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=5,
                         help="Save a checkpoint every N epochs.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # 1. Load and split data
    train_loader, val_loader, test_loader, class_names = prepare_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
    )

    # 2. Build model (sized for the number of classes found)
    model = ViT(
        img_size=args.img_size,
        num_classes=len(class_names),
        embed_dim=384,   # smaller default than the 768 ImageNet-scale config,
        depth=6,         # better suited to smaller custom datasets
        num_heads=6,
    )

    # 3. Train
    trainer = Trainer(
        model,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        lr=args.lr,
        class_names=class_names,
    )
    trainer.fit(train_loader, val_loader, epochs=args.epochs)

    # 4. Final evaluation on held-out test set
    trainer.evaluate(test_loader, split_name="test")


if __name__ == "__main__":
    main()
