from pathlib import Path

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

import re
import glob
import os
import pickle
import h5py
import numpy as np
from io import StringIO
from typing import Sequence

from pathlib import Path
import os
import h5py
from sympy import residue
import torch
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue, Atom
from io import StringIO
from typing import Sequence


def create_new_pdb_hdf52(
        peptide, peptide_idx, graph_name, run_id, data_dir, time_step, sample_id
):
    # hdf5_file = h5py.File(f'{data_dir}/test.hdf5', 'r')
    hdf5_file = h5py.File(f'{data_dir}BA_cluster1.hdf5', 'r') #roos
        
    pdb_names = hdf5_file['pdb_names'][:]
    pdb_strings = hdf5_file['pdb_strings'][:]
    pdb_string = pdb_strings[pdb_names.tolist().index(graph_name)].decode('utf-8')

    # Create a temporary file or use StringIO to make the string readable by parser
    pdb_fh = StringIO(pdb_string)
    
    pdb_output_path = f'./results/structures/{run_id}/{graph_name}_{time_step}_{sample_id}.pdb'

    directory = os.path.dirname(pdb_output_path)
    if not os.path.exists(directory):
         os.makedirs(directory)

    write_updated_peptide_coords_pdb2(peptide, peptide_idx, pdb_fh, pdb_output_path)

def create_new_pdb_hdf5_100k(
    peptide: np.ndarray,
    peptide_idx: Sequence[int],
    graph_name: str,
    run_id: str,
    data_dir: str,
    time_step: int,
    sample_id: int
):
    """
    Saves a new PDB for non-BA entries only.
    Loads the original PDB string via group[()] decoding,
    then overwrites the P-chain CA coords with `peptide`.
    """
    # 1) Only handle non-BA
    if graph_name.startswith("BA"):
        return

    # 2) Read the PDB string from the group
    hdf5_path = Path(data_dir) / '100k_test.hdf5'
    with h5py.File(hdf5_path, 'r') as f5:
        if graph_name not in f5:
            raise KeyError(f"{graph_name} not found in {hdf5_path}")
        group = f5[graph_name]
        pdb_string = group[()].decode('utf-8')

    # 3) Prepare in-memory file for parser
    pdb_fh = StringIO(pdb_string)

    # 4) Build output path
    out_dir = Path('results') / 'structures' / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_output_path = out_dir / f"{graph_name}_{time_step}_{sample_id}.pdb"

    # 5) Write updated PDB: replaces only P-chain CA coords
    write_updated_peptide_coords_pdb2(
        peptide=peptide,
        peptide_idx=peptide_idx,
        pdb_reference_path_or_stream=pdb_fh,
        pdb_output_path=str(pdb_output_path),
        atom_level=False
    )
    
def write_updated_peptide_coords_pdb2(
    peptide, peptide_idx, pdb_reference_path_or_stream, pdb_output_path, atom_level=False
):
    """
    Function from https://github.com/steusink/DiffSBDD.git

    Takes an existing pdb file with peptide and mhc and creates a new one
    with the same mhc pocket and the peptide with updated atom/residue
    coordinates given by the model.
    :param peptide: peptide with updated coordinates
    :param decoder: decoder, from index to atom/residue
    :param pdb_reference_path: path to the reference pdb file
    :param pdb_output_path: path to the output pdb file
    :param atom_level: whether to use atoms or residues

    :return: None
    """
    # Read the reference pdb file
    parser = PDBParser(QUIET=True)
    pdb_models = parser.get_structure("", pdb_reference_path_or_stream)

    # Get the peptide chain
    peptide_chain = pdb_models[0]["P"]

    if not atom_level:
        peptide_chain_new = Chain("P")

    # Get the peptide atoms/residues
    if atom_level:
        peptide_elements = peptide_chain.get_atoms()
    else:
        peptide_elements = peptide_chain.get_residues()

    # Update the peptide coordinates
    for i, element in enumerate(peptide_elements):
        if atom_level:
            element.set_coord(peptide[i])
        else:
            ca_atom = element["CA"] # might need to switch this to "CB"
            ca_atom.set_coord(peptide[i])
            id = element.get_id()
            id = (' ', int(peptide_idx[i]), ' ')
            new_residue = Residue(id, element.get_resname(), "")
            new_residue.add(ca_atom)
            peptide_chain_new.add(new_residue)

    # Write the new pdb file
    if not atom_level:
        pdb_models[0].detach_child("P")
        pdb_models[0].add(peptide_chain_new)

        # Write the new pdb file
    io = PDBIO()
    io.set_structure(pdb_models)
    io.save(str(pdb_output_path))


# Mapping for One-Hot recovery (Standard alphabetical order)
AA_MAP = "ACDEFGHIKLMNPQRSTVWY"

def create_new_pdb_hdf5(
        peptide, peptide_idx, graph_name, run_id, data_dir, time_step, sample_id, fold="1"
):
    """
    Creates a PDB file using the new Group-based HDF5 structure.
    Tries to load a raw PDB string if available, otherwise reconstructs from tensors.
    """
    
    # 1. Determine HDF5 Path
    # Try the cluster file first, fallback to valid/train if needed
    hdf5_path = os.path.join(data_dir, f'BA_cluster{fold}.hdf5')
    if not os.path.exists(hdf5_path):
        hdf5_path = os.path.join(data_dir, f'valid_fold{fold}.hdf5')

    pdb_output_path = f'./results/structures/{run_id}/{graph_name}_{time_step}_{sample_id}.pdb'
    directory = os.path.dirname(pdb_output_path)
    if not os.path.exists(directory):
         os.makedirs(directory)

    with h5py.File(hdf5_path, 'r') as f5:
        if graph_name not in f5:
            print(f"Warning: {graph_name} not found in {hdf5_path}")
            return

        group = f5[graph_name]

        # --- STRATEGY A: Use stored PDB string if available ---
        # Some datasets store the raw string in a dataset named 'pdb_string' inside the group
        if 'pdb_string' in group:
            pdb_string = group['pdb_string'][()].decode('utf-8')
            pdb_fh = StringIO(pdb_string)
            write_updated_peptide_coords_pdb(peptide, peptide_idx, pdb_fh, pdb_output_path)
            
        # --- STRATEGY B: Reconstruct from Tensors ---
        # If no raw string, build the protein structure from the stored coordinates
        else:
            # Load Protein Data
            pro_pos = group['protein']['all_atom_positions'][:] # Shape (N, atoms, 3)
            pro_seq_onehot = group['protein']['sequence_onehot'][:]
            
            # Reconstruct the reference structure (Protein M-chain)
            structure = reconstruct_structure_from_tensors(graph_name, pro_pos, pro_seq_onehot, chain_id="M")
            
            # Save the protein-only structure to a buffer to use your existing helper
            temp_io = StringIO()
            io = PDBIO()
            io.set_structure(structure)
            io.save(temp_io)
            temp_io.seek(0)
            
            write_updated_peptide_coords_pdb(peptide, peptide_idx, temp_io, pdb_output_path)


def reconstruct_structure_from_tensors(name, positions, one_hot, chain_id="M"):
    """
    Helper to build a Bio.PDB Structure from HDF5 tensors.
    """
    structure = Structure.Structure(name)
    model = Model.Model(0)
    chain = Chain.Chain(chain_id)
    
    # Convert one-hot to indices
    seq_indices = np.argmax(one_hot[:, :20], axis=1)
    
    for i, (residue_pos, aa_idx) in enumerate(zip(positions, seq_indices)):
        res_name = three_letter_code(AA_MAP[aa_idx])
        # Bio.PDB requires a tuple (hetero_flag, seq_id, insert_code)
        res_id = (' ', i + 1, ' ')
        residue = Residue.Residue(res_id, res_name, ' ')
        
        # Add CA atom. 
        # If input is (N, 3), use it directly. If (N, Atoms, 3), use index 1 (CA).
        if residue_pos.ndim == 2:
            ca_coord = residue_pos[1] 
        else:
            ca_coord = residue_pos 
            
        # Create CA Atom
        atom = Atom.Atom("CA", ca_coord, 20.0, 1.0, " ", " CA ", i, element="C")
        residue.add(atom)
        chain.add(residue)
        
    model.add(chain)
    structure.add(model)
    return structure

def three_letter_code(one_letter):
    """Simple converter"""
    d = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
         'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
         'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
         'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'}
    return d.get(one_letter, 'UNK')

def write_updated_peptide_coords_pdb(
    peptide, peptide_idx, pdb_reference_path_or_stream, pdb_output_path, atom_level=False
):
    """
    Takes an existing pdb file with peptide and mhc and creates a new one
    with the same mhc pocket and the peptide with updated atom/residue
    coordinates given by the model.
    """
    parser = PDBParser(QUIET=True)
    pdb_models = parser.get_structure("", pdb_reference_path_or_stream)

    # Remove existing peptide chain if it exists
    if "P" in pdb_models[0]:
        pdb_models[0].detach_child("P")
    
    peptide_chain_new = Chain.Chain("P")

    # Update the peptide coordinates
    for i, coord in enumerate(peptide):
        # Create new residue
        # Note: If you don't have the sequence here, we default to GLY. 
        # To fix this, you would need to pass 'peptide_seq' into this function.
        res_id = (' ', int(peptide_idx[i]) if peptide_idx is not None else i+1, ' ')
        new_residue = Residue.Residue(res_id, "GLY", "") 
        
        atom = Atom.Atom("CA", coord, 20.0, 1.0, " ", " CA ", i, element="C")
        new_residue.add(atom)
        peptide_chain_new.add(new_residue)

    pdb_models[0].add(peptide_chain_new)

    io = PDBIO()
    io.set_structure(pdb_models)
    io.save(str(pdb_output_path))