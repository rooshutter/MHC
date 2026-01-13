import os
import h5py
import torch
from torch.utils.data import Dataset
from Bio import PDB  # Biopython's PDB parser
from typing import Dict

# Define amino acid one-hot encoding for 20 standard residues
AA_LIST = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_LIST)}

def one_hot_encode_sequence(sequence):
    """One-hot encode the amino acid sequence based on standard residues."""
    encoding = torch.zeros((len(sequence), len(AA_LIST)), dtype=torch.float32)
    for i, aa in enumerate(sequence):
        if aa in AA_TO_INDEX:
            encoding[i, AA_TO_INDEX[aa]] = 1.0
        else:
            print(f"Warning: Non-standard amino acid '{aa}' found and ignored.")
    return encoding

class PDB_Dataset(Dataset):

    def __init__(self, datadir, split='train', fold="1"):
        """
        Args:
            datadir (str): Path to the directory where HDF5 files are located.
            split (str): Dataset split, one of 'train', 'valid', 'test'.
        """

        if split == 'test': # roos
            self.hdf5_path = os.path.join(datadir, f'BA_cluster{fold}.hdf5')
        else:
            self.hdf5_path = os.path.join(datadir, f'{split}_fold{fold}.hdf5')
        print(f"Loading dataset from {self.hdf5_path}...")

        # Open file to get the list of keys (Entry IDs like 'BA-55224')
        with h5py.File(self.hdf5_path, 'r') as f5:
            # Instead of looking for 'pdb_strings', we get the group keys
            self.entry_names = list(f5.keys())
            
        print(f"Loaded {len(self.entry_names)} entries from {split} split.")

    def __len__(self) -> int:
        return len(self.entry_names)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.get_entry(index)

    def get_entry(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves pre-processed tensors and forces them to 20 dimensions.
        """
        entry_name = self.entry_names[index]
        data = {}

        with h5py.File(self.hdf5_path, 'r') as f5:
            group = f5[entry_name]
            
            # --- 1. Load Peptide Data ---
            # Position: Take C-alpha (index 1)
            pep_all_pos = group['peptide']['all_atom_positions'][:]
            peptide_coords = torch.tensor(pep_all_pos[:, 1, :], dtype=torch.float32)
            
            # Features: SLICE the first 20 columns only
            full_pep_onehot = group['peptide']['sequence_onehot'][:]
            # Force shape (N, 20)
            peptide_onehot = torch.tensor(full_pep_onehot[:, :20], dtype=torch.float32)
            
            pep_pos_in_seq = torch.tensor(group['peptide']['residue_numbers'][:], dtype=torch.long)

            # --- 2. Load Protein Data ---
            pro_all_pos = group['protein']['all_atom_positions'][:]
            protein_coords = torch.tensor(pro_all_pos[:, 1, :], dtype=torch.float32)
            
            # Features: SLICE the first 20 columns only
            full_pro_onehot = group['protein']['sequence_onehot'][:]
            # Force shape (N, 20)
            protein_onehot = torch.tensor(full_pro_onehot[:, :20], dtype=torch.float32)

            # --- 3. Create Masks ---
            peptide_len = peptide_coords.shape[0]
            protein_len = protein_coords.shape[0]
            
            peptide_mask = torch.ones(peptide_len, dtype=torch.bool)
            protein_mask = torch.ones(protein_len, dtype=torch.bool)

            # --- 4. Package ---
            data['graph_name'] = entry_name
            
            # Peptide
            data['peptide_idx'] = peptide_mask
            data['peptide_positions'] = peptide_coords
            data['peptide_features'] = peptide_onehot  # Now guaranteed to be 20 dims
            data['num_peptide_residues'] = torch.tensor(peptide_len)
            data['pos_in_seq'] = pep_pos_in_seq 

            # Protein Pocket
            data['protein_pocket_idx'] = protein_mask
            data['protein_pocket_positions'] = protein_coords
            data['protein_pocket_features'] = protein_onehot  # Now guaranteed to be 20 dims
            data['num_protein_pocket_residues'] = torch.tensor(protein_len)

            # Batch index placeholders
            data['peptide_batch_idx'] = torch.zeros(peptide_len, dtype=torch.long)
            data['protein_batch_idx'] = torch.zeros(protein_len, dtype=torch.long)

        return data
    
    @staticmethod
    def collate_fn(batch):
        """
        Collation function to combine batch data into a single batch.
        """
        data_batch = {}
        
        # Keys that are lists of strings
        data_batch['graph_name'] = [x['graph_name'] for x in batch]
        
        # Keys that are single values per graph (1D tensor)
        for key in ['num_peptide_residues', 'num_protein_pocket_residues']:
             data_batch[key] = torch.stack([x[key] for x in batch])

        # Keys that need concatenation (Node features/positions)
        cat_keys = [
            'peptide_idx', 'peptide_positions', 'peptide_features', 'pos_in_seq',
            'protein_pocket_idx', 'protein_pocket_positions', 'protein_pocket_features'
        ]
        
        for key in cat_keys:
            data_batch[key] = torch.cat([x[key] for x in batch], dim=0)

        # Handle Batch Indices (needed for torch_scatter usually)
        # We create a new 'idx' key that maps every node to its graph index in the batch
        peptide_batch_indices = []
        protein_batch_indices = []
        
        for i, item in enumerate(batch):
            peptide_batch_indices.append(torch.full((item['num_peptide_residues'],), i, dtype=torch.long))
            protein_batch_indices.append(torch.full((item['num_protein_pocket_residues'],), i, dtype=torch.long))
            
        data_batch['idx_peptide'] = torch.cat(peptide_batch_indices)
        data_batch['idx_protein'] = torch.cat(protein_batch_indices)
        
        # Map back to generic 'idx' if your model uses that specific name
        # (Assuming your model looks for 'peptide_idx' as the scatter index, 
        # usually usually usually named 'batch' or 'idx' in PyG)
        data_batch['peptide_idx'] = data_batch['idx_peptide'] 
        data_batch['protein_pocket_idx'] = data_batch['idx_protein']

        return data_batch
    


class PDB_Dataset2(Dataset):
    
    def __init__(self, datadir, split='train', fold="1"): # Roos: fold hardcoded to 1 for now
        """
        Args:
            datadir (str): Path to the directory where HDF5 files are located.
            split (str): Dataset split, one of 'train', 'valid', 'test'.
        """
        # Define the HDF5 file paths for the dataset split
        if split == 'test': # roos
            self.hdf5_path = os.path.join(datadir, f'BA_cluster{fold}.hdf5')
        else:
            self.hdf5_path = os.path.join(datadir, f'{split}_fold{fold}.hdf5')

        print(f"Loading dataset from {self.hdf5_path}...")

        # Open the HDF5 file and load the pdb_strings dataset directly
        with h5py.File(self.hdf5_path, 'r') as f5:
            #####################################################################
            # self.entry_names = list(f5.keys())
            # print(f"Entries in the HDF5 file: {self.entry_names}")
            
            # group = f5['BA-55224']

            # for name, dataset in group.items():
            
            #     # Skip if it's a nested group, only print datasets
            #     if not isinstance(dataset, h5py.Dataset):
            #         print(f"Skipping: {name} is a nested group.")
            #         for sub_name, sub_dataset in dataset.items():
            #             print(f"  - {sub_name}: shape {sub_dataset.shape}, dtype {sub_dataset.dtype}")
            #             print(sub_dataset)
            #         continue

            #     print(f"\n[DATASET: {name}]")
            #     print(f"  Shape: {dataset.shape}")
            #     print(f"  Data Type: {dataset.dtype}")
            #     print(dataset)

            ####################################################################

            self.pdb_strings = f5['pdb_strings'][:]  # Load the pdb_strings array directly
            self.pdb_names = f5['pdb_names'][:]  # Load the pdb_names array
            print(f"Loaded {len(self.pdb_strings)} pdb strings and names from {split} split.")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Returns a data entry from the HDF5 file for a given index.
        """
        # print(f"Loading entry at index: {index}")
        return self.get_entry(index)

    def __len__(self) -> int:
        """
        Returns the total number of entries in the dataset.
        """
        return len(self.pdb_strings)

    def get_entry(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves and processes a single entry from the HDF5 file.
        
        Args:
            entry_name (str): The name of the entry in the HDF5 file.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the processed data.
        """
        data = {}
        with h5py.File(self.hdf5_path, 'r') as f5:
            
            # Access the pdb_string directly using the index
            pdb_string = self.pdb_strings[index].decode('utf-8')
            # pdb_string = entry[0].decode('utf-8')  # Access by index to ensure it's correct

            # Parse the PDB data using Biopython
            structure = self.parse_pdb_structure(pdb_string)

            # Extract peptide (P chain) and protein (M chain) data
            peptide_chain = structure[0]['P']
            protein_chain = structure[0]['M']

            # Extract C-alpha atom coordinates and sequence for peptide and protein
            peptide_coords, peptide_seq = self.extract_ca_coords_and_sequence(peptide_chain)
            # print(f"Peptide sequence: {peptide_seq} (length: {len(peptide_seq)})")
            
            # Truncate the M chain (protein chain) to the first 178 residues
            protein_coords, protein_seq = self.extract_ca_coords_and_sequence(protein_chain, max_residues=178)
            # print(f"Protein sequence: {protein_seq} (length: {len(protein_seq)})")

            # One-hot encode sequences
            peptide_onehot = one_hot_encode_sequence(peptide_seq)
            protein_onehot = one_hot_encode_sequence(protein_seq)

            # Generate masks (assuming all residues are valid for now)
            peptide_len = peptide_coords.shape[0]
            protein_len = protein_coords.shape[0]
            peptide_mask = torch.ones(peptide_len, dtype=torch.bool)
            protein_mask = torch.ones(protein_len, dtype=torch.bool)

            # Prepare the output dictionary
            data['graph_name'] = self.pdb_names[index]
            data['peptide_idx'] = peptide_mask
            data['peptide_positions'] = peptide_coords  # 3D C-alpha coordinates for peptide
            data['peptide_features'] = peptide_onehot  # One-hot encoded peptide sequence
            data['num_peptide_residues'] = peptide_len
            data['protein_pocket_idx'] = protein_mask
            data['protein_pocket_positions'] = protein_coords  # 3D C-alpha coordinates for protein
            data['protein_pocket_features'] = protein_onehot  # One-hot encoded protein sequence
            data['num_protein_pocket_residues'] = protein_len
            data['pos_in_seq'] = torch.arange(peptide_len) + 1  # Position in the sequence

        return data

    def parse_pdb_structure(self, pdb_string: str):
        """Parses the PDB string using Biopython and returns a structure."""
        parser = PDB.PDBParser(QUIET=True)
        from io import StringIO
        pdb_io = StringIO(pdb_string)
        structure = parser.get_structure("structure", pdb_io)
        return structure

    def extract_ca_coords_and_sequence(self, chain, max_residues=None):
        """
        Extract C-alpha coordinates and amino acid sequence from a PDB chain.

        Args:
            chain (Bio.PDB.Chain): The chain object from which to extract data.
            max_residues (int, optional): Maximum number of residues to include (for truncation).

        Returns:
            coords (torch.Tensor): C-alpha coordinates (Nx3 tensor).
            sequence (str): Corresponding amino acid sequence.
        """
        ca_coords = []
        sequence = []
        
        for i, residue in enumerate(chain):
            if max_residues is not None and i >= max_residues:
                break  # Truncate if the number of residues exceeds the limit

            if 'CA' in residue:
                ca_coords.append(residue['CA'].coord)
                sequence.append(PDB.Polypeptide.three_to_one(residue.resname))
            else:
                print(f"Warning: Missing CA atom for residue {residue.resname} in chain {chain.id}")

        # Convert to torch tensors
        coords_tensor = torch.tensor(ca_coords, dtype=torch.float32)  # Shape: (N, 3)
        sequence_str = ''.join(sequence)
        
        return coords_tensor, sequence_str


    @staticmethod
    def collate_fn(batch):
        """
        Collation function to combine batch data into a single batch.
        
        Args:
            batch (list of Dict): A list of individual data entries.

        Returns:
            Dict: A dictionary containing batched data.
        """
        data_batch = {}
        for key in batch[0].keys():

            if key == 'graph_name':
                data_batch[key] = [x[key] for x in batch]
            elif key == 'num_peptide_residues' or key == 'num_protein_pocket_residues':
                data_batch[key] = torch.tensor([x[key] for x in batch])
            elif 'idx' in key:
                # Ensure that indices in the batch start at zero (needed for torch_scatter)
                data_batch[key] = torch.cat([i * torch.ones(len(x[key]), dtype=torch.long) for i, x in enumerate(batch)], dim=0)
            else:
                data_batch[key] = torch.cat([x[key] for x in batch], dim=0)

        return data_batch
