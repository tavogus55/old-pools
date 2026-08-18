import argparse
from pathlib import Path
import time

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from pooling_models import DensePool, sparse_pooling


DENSE_MAX_NODES = {
    "MUTAG": 150,
    "DD": 500,
    "IMDB-MULTI": 500,
    "PROTEINS": 700,
    "IMDB-BINARY": 500,
    "COLLAB": 150,
    "NCI1": 150,
    "NCI109": 150,
}


def main(
    dataset_name: str,
    model_name: str,
    epochs: int,
    hidden: int,
    pratio: float,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    batch_size: int,
) -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"

    dataset = TUDataset(root=data_dir, name=dataset_name)
    output_path = data_dir / f"{dataset_name}.pt"
    torch.save(dataset, output_path)

    is_dense = model_name in {
        "diff",
        "mincut",
        "gaus",
        "unif",
        "count1",
        "count2",
        "count4",
    }
    max_nodes = DENSE_MAX_NODES[dataset_name] if is_dense else None
    input_dim = max(1, dataset.num_features)
    num_classes = dataset.num_classes

    if is_dense:
        dataset = [data for data in dataset if data.num_nodes <= max_nodes]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if is_dense:
        model = DensePool(
            input_dim,
            num_classes,
            model=model_name,
            hidden=hidden,
            pratio=pratio,
            dropout=dropout,
            max_nodes=max_nodes,
        ).to(device)
    else:
        model = sparse_pooling(
            input_dim,
            num_classes,
            model=model_name,
            hidden=hidden,
            pratio=pratio,
            dropout=dropout,
        ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()
        total_loss = 0.0
        total_pool_time = 0.0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            if batch.x is None or batch.x.size(1) == 0:
                x = torch.ones(
                    (batch.num_nodes, 1),
                    dtype=torch.float,
                    device=batch.edge_index.device,
                )
            else:
                x = batch.x
            output = model(x, batch.edge_index, batch.batch)
            auxiliary_loss = getattr(model, "last_auxiliary_loss", None)
            if auxiliary_loss is None:
                auxiliary_loss = output.new_zeros(())
            loss = criterion(output, batch.y) + auxiliary_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_pool_time += model.last_pool_time

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        average_loss = total_loss / len(loader)
        epoch_duration = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch}/{epochs} - loss: {average_loss:.4f} "
            f"- Epoch Time: {epoch_duration:.2f}s "
            f"- Pool Time: {total_pool_time:.2f}s"
        )

    total_training_time = time.perf_counter() - training_start
    print(f"Total training time for {epochs} epochs: {total_training_time:.2f}s")

    print(f"Loaded {len(dataset)} graphs from the {dataset_name} TU dataset.")
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=(
            "PROTEINS",
            "DD",
            "IMDB-MULTI",
            "IMDB-BINARY",
            "MUTAG",
            "NCI1",
            "NCI109",
            "COLLAB",
        ),
        default="DD",
        help="TU dataset to use.",
    )
    parser.add_argument(
        "--model",
        choices=(
            "sag",
            "topk",
            "ndrp",
            "diff",
            "mincut",
            "gaus",
            "unif",
            "count1",
            "count2",
            "count4",
        ),
        default="topk",
        help="Pooling layer to use.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=32,
        help="Node embedding hidden dimension.",
    )
    parser.add_argument(
        "--pratio",
        type=float,
        default=0.5,
        help="Pooling ratio.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Optimizer weight decay.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size.",
    )
    args = parser.parse_args()
    main(
        args.dataset,
        args.model,
        args.epochs,
        args.hidden,
        args.pratio,
        args.learning_rate,
        args.weight_decay,
        args.dropout,
        args.batch_size,
    )
