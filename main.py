import argparse
from pathlib import Path
import time

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from pooling_models import sparse_pooling


def main(model_name: str, epochs: int) -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"

    dataset = TUDataset(root=data_dir, name="DD")
    output_path = data_dir / "DD.pt"
    torch.save(dataset, output_path)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = sparse_pooling(
        dataset.num_features,
        dataset.num_classes,
        model=model_name,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    training_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        total_loss = 0.0

        for batch in loader:
            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(loader)
        epoch_duration = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch}/{epochs} - loss: {average_loss:.4f} "
            f"- time: {epoch_duration:.2f}s"
        )

    total_training_time = time.perf_counter() - training_start
    print(f"Total training time for {epochs} epochs: {total_training_time:.2f}s")

    print(f"Loaded {len(dataset)} graphs from the DD TU dataset.")
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("sag", "topk", "ndrp"),
        default="topk",
        help="Pooling layer to use.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Number of training epochs.",
    )
    args = parser.parse_args()
    main(args.model, args.epochs)
