import os

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset


class CSVMoleculeDataset(InMemoryDataset):
    """Load a molecular regression dataset from a SMILES CSV file."""

    def __init__(
        self,
        root,
        csv_file,
        target_cols=None,
        transform=None,
        pre_transform=None,
    ):
        self.csv_file = csv_file
        self.target_cols = target_cols
        super().__init__(root, transform, pre_transform)

        if hasattr(self, "load"):
            self.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.basename(self.csv_file)]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        # The CSV files are supplied locally with this project.
        pass

    def process(self):
        dataframe = pd.read_csv(self.csv_file)
        target_cols = self.target_cols
        if target_cols is None:
            target_cols = [column for column in dataframe.columns if column != "smiles"]

        data_list = []
        for _, row in dataframe.iterrows():
            molecule = Chem.MolFromSmiles(row["smiles"])
            if molecule is None:
                continue

            features = []
            for atom in molecule.GetAtoms():
                features.append([
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    int(atom.GetIsAromatic()),
                ])

            edges = []
            for bond in molecule.GetBonds():
                source = bond.GetBeginAtomIdx()
                target = bond.GetEndAtomIdx()
                edges.extend([[source, target], [target, source]])

            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            data_list.append(
                Data(
                    x=torch.tensor(features, dtype=torch.float),
                    edge_index=edge_index,
                    y=torch.tensor(
                        row[target_cols].values.astype(float),
                        dtype=torch.float,
                    ).view(1, -1),
                )
            )

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
