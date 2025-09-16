import hydra
import omegaconf
import torch
import pandas as pd
import pickle
from omegaconf import ValueNode
from torch.utils.data import Dataset

from torch_geometric.data import Data

from cdvae.common.utils import PROJECT_ROOT
from cdvae.common.numpy_compat import setup_numpy_compatibility
from cdvae.common.data_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop)
from cdvae.common.caching import get_data_cache, cached_preprocessing

# Ensure numpy compatibility is set up
setup_numpy_compatibility()


class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode,
                 use_cache: bool = True,
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.use_cache = use_cache
        
        # Initialize cache with absolute directory that's experiment-independent
        import os
        from cdvae.common.utils import PROJECT_ROOT
        cache_dir = os.path.join(PROJECT_ROOT, "shared_data_cache")
        self.cache = get_data_cache(cache_dir) if use_cache else None
        
        # Check if the file is a pickle file or CSV file
        if str(path).endswith('.pkl'):
            with open(path, 'rb') as f:
                self.df = pickle.load(f)
        else:
            self.df = pd.read_csv(path)
            
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        # Use cached preprocessing if available
        import hashlib
        import os
        
        # Create a more robust cache key using file path, size, and modification time
        try:
            file_stat = os.stat(path)
            file_info = f"{path}_{file_stat.st_size}_{file_stat.st_mtime}"
        except:
            file_info = str(path)
            
        cache_key_data = f"{file_info}_{niggli}_{primitive}_{graph_method}_{prop}"
        cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()
        
        if self.cache:
            self.cached_data = self.cache.get(cache_key)
            if self.cached_data is None:
                print(f"Preprocessing data for {name} (not cached)...")
                self.cached_data = preprocess(
                    self.path,
                    preprocess_workers,
                    niggli=self.niggli,
                    primitive=self.primitive,
                    graph_method=self.graph_method,
                    prop_list=[prop])
                self.cache.set(cache_key, self.cached_data)
                print(f"Cached preprocessing results for {name}")
            else:
                print(f"Using cached preprocessing results for {name}")
        else:
            print(f"Preprocessing data for {name} (caching disabled)...")
            self.cached_data = preprocess(
                self.path,
                preprocess_workers,
                niggli=self.niggli,
                primitive=self.primitive,
                graph_method=self.graph_method,
                prop_list=[prop])

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        prop = self.scaler.transform(data_dict[self.prop])
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
        )
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    from cdvae.common.data_utils import get_scaler_from_data_list
    dataset: CrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    lattice_scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key='scaled_lattice')
    scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key=dataset.prop)

    dataset.lattice_scaler = lattice_scaler
    dataset.scaler = scaler
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == "__main__":
    main()
