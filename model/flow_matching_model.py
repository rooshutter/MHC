import torch
import numpy as np
import math
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean

from model.noise_schedule import Noise_Schedule
from utils import create_new_pdb_hdf5, create_new_pdb_hdf5_100k


class Flow_Matching_Model(nn.Module):
    def __init__(
        self,
        neural_net: nn.Module,
        features_fixed: bool,
        confidence_score: bool,
        timesteps: int,
        position_encoding: bool,
        com_handling: str,
        sampling_stepsize: int,
        noise_scaling: int,
        high_noise_training: bool,
        num_atoms: int,
        num_residues: int,
        norm_values: list,

    ):
        super().__init__()

        self.neural_net = neural_net
        self.T = timesteps
        self.features_fixed = features_fixed
        self.position_encoding = position_encoding
        self.com_handling = com_handling
        self.sampling_stepsize = sampling_stepsize

        # dataset info
        self.num_atoms = num_atoms
        self.num_residues = num_residues
        self.norm_values = norm_values
        self.x_dim = 3

        # Noise Schedule
        self.noise_schedule = Noise_Schedule(self.T)

        # Further model hyperparameters
        if noise_scaling == None:
            self.noise_scaling = 1
        else:
            self.noise_scaling = noise_scaling
        self.high_noise_training = high_noise_training

        self.confidence_score = confidence_score

    def forward(self, z_data):

        molecule, protein_pocket = z_data

        if self.position_encoding:
            molecule_pos = molecule['pos_in_seq']
        else:
            molecule_pos = None

        # compute noised samples
        # z_t_mol: noised molecole at time t
        # z_t_pro: protein pocket at time t
        # eps_x_mol: actual noise added to molecule positions
        # epsilon_pro: actual noise added to protein pocket positions (should be 0)
        # z_t_mol, z_t_pro, eps_x_mol, epsilon_pro, t = self.noise_process(z_data)
        z_t_mol, z_t_pro, v_target_mol, t = self.compute_flow_match(z_data)
        # print("z_t_mol shape:", z_t_mol.shape) # z_t_mol shape: torch.Size([288, 23]
        # print("z_t_pro shape:", z_t_pro.shape) # z_t_pro shape: torch.Size([5760, 23])
        # print("v_target_mol shape:", v_target_mol.shape) # v_target_mol shape: torch.Size([288, 3])
        # print("t shape:", t.shape) # t shape: torch.Size([32, 1])

        # z_t_mol shape: torch.Size([288, 43])
        # z_t_pro shape: torch.Size([5760, 23])
        # v_target_mol shape: torch.Size([288, 3])
        # t shape: torch.Size([32, 1])

        # use neural netwrok to predict vector field
        # epsilon_hat_mol: predicted noise for molecule positions
        # epsilon_hat_pro: predicted noise for protein pocket positions
        # c_s: predicted confidence score
        # epsilon_hat_mol, epsilon_hat_pro, c_s = self.neural_net(z_t_mol, z_t_pro, t, molecule['idx'], protein_pocket['idx'], molecule_pos)
        v_hat_mol, v_hat_pro, c_s = self.neural_net(z_t_mol, z_t_pro, t, molecule['idx'], protein_pocket['idx'], molecule_pos)
        # print("v_hat_mol shape:", v_hat_mol.shape) 
        # print("v_hat_pro shape:", v_hat_pro.shape)
        # print("c_s:", c_s)
        
        if self.training:
            loss, info = self.train_loss(molecule, protein_pocket, z_t_mol, z_t_pro, v_target_mol, v_hat_mol, v_hat_pro, c_s)
        else: 
            loss, info = self.validation_loss(molecule, protein_pocket, z_t_mol, z_t_pro, v_target_mol, v_hat_mol, v_hat_pro, c_s)

        return loss.mean(0), info
    
    def compute_flow_match(self, z_data):

        molecule, protein_pocket = z_data
        batch_size = molecule['size'].size(0)
        device = molecule['x'].device

        # normalisation with norm_values (dataset dependend) -> changes likelyhood (adjusted for in vlb)!
        molecule['x'] = molecule['x'] / self.norm_values[0]
        molecule['h'] = molecule['h'] / self.norm_values[1]
        protein_pocket['x'] = protein_pocket['x'] / self.norm_values[0]
        protein_pocket['h'] = protein_pocket['h'] / self.norm_values[1]

        # sample t ~ U(0,...,T) for each graph individually
        t_low = 0 if self.train else 1
        t = torch.randint(t_low, self.T + 1, size=(batch_size, 1), device=device)

        # normalize t
        t = t / self.T

        # option for computing t = 0 representations
        # t = torch.zeros((batch_size, 1), device=device) if t_is_0 else t

        # prepare joint point cloud
        xh_mol = torch.cat((molecule['x'], molecule['h']), dim=1)
        xh_pro = torch.cat((protein_pocket['x'], protein_pocket['h']), dim=1)

        # center of mass handling
        # print("com_handling:", self.com_handling) peptide
        if self.com_handling == 'both':
            # old centering approach
            xh_mol[:,:self.x_dim] = xh_mol[:,:self.x_dim] - scatter_mean(xh_mol[:,:self.x_dim], molecule['idx'], dim=0)[molecule['idx']]
            xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - scatter_mean(xh_pro[:,:self.x_dim], protein_pocket['idx'], dim=0)[protein_pocket['idx']]
        elif self.com_handling == 'no_COM':
            dumy_variable = 0
        else:
            # data is translated to 0, COM noise added and again translated to 0
            mean = scatter_mean(xh_mol[:,:self.x_dim], molecule['idx'], dim=0)
            xh_mol[:,:self.x_dim] = xh_mol[:,:self.x_dim] - mean[molecule['idx']]
            xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - mean[protein_pocket['idx']]

        # compute noised sample z_t
        # for x cord. we mean center the normal noise for each graph
        # we only diffuse position of the molecules
        # print("noise_scaling:", self.noise_scaling) 1
        z_x_mol = torch.randn(size=(len(xh_mol), self.x_dim), device=device) * self.noise_scaling
        z_x_pro = torch.zeros(size=(len(xh_pro), self.x_dim), device=device)

        if self.com_handling == 'both':
            # alternative centering approach
            z_x_mol = z_x_mol - scatter_mean(z_x_mol, molecule['idx'], dim=0)[molecule['idx']]
            z_x_pro = torch.zeros(size=(len(xh_pro), self.x_dim), device=device)
        else:
            dumy_variable = 0

        # print("features_fixed:", self.features_fixed) true
        if self.features_fixed:
            z_h_mol = torch.zeros(size=(len(xh_mol), self.num_atoms), device=device)
            z_h_pro = torch.zeros(size=(len(xh_pro), self.num_residues), device=device)
        else:
            # for h we need standard normal noise (this would be sampling new peptides)
            z_h_mol = torch.randn(size=(len(xh_mol), self.num_atoms), device=device)
            z_h_pro = torch.randn(size=(len(xh_pro), self.num_residues), device=device)

        z_mol = torch.cat((z_x_mol, z_h_mol), dim=1)
        z_pro = torch.cat((z_x_pro, z_h_pro), dim=1)

        # compute noised representations
        # x_t = t * x_1 + (1 - t) * x_0
        # x_1: original data point
        # x_0: pure noise

        # print("xh_mol shape:", xh_mol.shape) 23
        # print("z_mol shape:", z_mol.shape) 23
        # print("t shape:", t.shape) # t shape: torch.Size([32, 1])
        # print("t[molecule['idx']] shape:", t[molecule['idx']].shape) # t[molecule['idx']] shape: torch.Size([288, 1])

        z_t_mol_x = t[molecule['idx']] * molecule['x'] + (1 - t[molecule['idx']]) * z_x_mol
        # print("molecule['x'] shape:", molecule['x'].shape) # molecule['x'] shape: torch.Size([288, 3])
        # print("z_x_mol shape:", z_x_mol.shape) # z_x_mol shape: torch.Size([288, 3])
        # print("z_t_mol_x shape:", z_t_mol_x.shape) # z_t_mol_x shape: torch.Size([288, 3])

        z_t_mol = torch.cat((z_t_mol_x, xh_mol[:,self.x_dim:]), dim=1)
        # print("xh_mol[:,self.x_dim:]):", xh_mol[:,self.x_dim:].shape) 
        # print("z_t_mol shape:", z_t_mol.shape) 
        z_t_pro = xh_pro.clone().detach()

        if self.com_handling == 'both':
            dumy_variable = 0
        elif self.com_handling == 'no_COM':
            dumy_variable = 0
        else:
            # data is translated to 0, COM noise added and again translated to 0 (turn off for old centering approach)
            mean = scatter_mean(z_t_mol[:,:self.x_dim], molecule['idx'], dim=0)
            z_t_mol[:,:self.x_dim] = z_t_mol[:,:self.x_dim] - mean[molecule['idx']]
            z_t_pro[:,:self.x_dim] = z_t_pro[:,:self.x_dim] - mean[protein_pocket['idx']]

        # v = x_1 - x_0
        # xh_mol[:,:self.x_dim] - z_x_mol
        v_target_mol = xh_mol - z_mol

        return z_t_mol, z_t_pro, v_target_mol, t
    
    def train_loss(self, molecule, protein_pocket, z_t_mol, z_t_pro, v_target_mol, v_hat_mol, v_hat_pro, c_s):

        # Sum squared error per graph
        loss_per_atom = torch.sum((v_target_mol - v_hat_mol)**2, dim=-1)
        loss_per_graph = scatter_add(loss_per_atom, molecule['idx'], dim=0)
        
        # Normalize by size
        loss_per_graph = loss_per_graph / (molecule['size'] * (self.x_dim + self.num_atoms))
        
        loss = loss_per_graph

        # Confidence handling
        if self.confidence_score:
            c_s_peptide = scatter_add(c_s, molecule['idx'], dim=0).squeeze(1) / molecule['size']
            loss_with_conf = 1/(c_s_peptide)**2 * loss + torch.log(c_s_peptide**2)
        else:
            c_s_peptide = torch.zeros_like(loss)
            loss_with_conf = loss

        info = {
            'loss': loss.mean(),
            'confidence': c_s_peptide.mean(),
            'loss_with_conf': loss_with_conf.mean(),
            'error_mol': loss.mean(),
        }

        if self.confidence_score:
            return loss_with_conf, info
        
        return loss, info
    
    def validation_loss(self, molecule, protein_pocket, z_t_mol, z_t_pro, v_target_mol, v_hat_mol, v_hat_pro, c_s):

        return self.train_loss(molecule, protein_pocket, z_t_mol, z_t_pro, v_target_mol, v_hat_mol, v_hat_pro, c_s)
    
    @torch.no_grad()
    def sample_structure(self, num_samples, molecule, protein_pocket, sampling_without_noise, data_dir, run_id):

        device = molecule['x'].device
        if self.position_encoding:
            molecule_pos = molecule['pos_in_seq']
        else:
            molecule_pos = None
        num_samples = len(molecule['size'])

        # Record protein_pocket center of mass before
        protein_pocket_com_before = scatter_mean(protein_pocket['x'], protein_pocket['idx'], dim=0)

        # define the target at the pocket position
        mol_target_p = molecule['x'] - scatter_mean(molecule['x'], molecule['idx'], dim=0)[molecule['idx']]
        mol_target_p = mol_target_p + protein_pocket_com_before[molecule['idx']]

        # define the target at 0-COM
        mol_target_0 = molecule['x'] - scatter_mean(molecule['x'], molecule['idx'], dim=0)[molecule['idx']]

        # Normalisation
        # molecule['x'] = molecule['x'] / self.norm_values[0]
        molecule['h'] = molecule['h'] / self.norm_values[1]
        protein_pocket['x'] = protein_pocket['x'] / self.norm_values[0]
        protein_pocket['h'] = protein_pocket['h'] / self.norm_values[1]

        # start with random peptide position (target hidden)
        # mean=COM, sigma=1, and sample epsioln (can add noise scaling, but works better without)
        rand_eps_x = torch.randn((len(molecule['x']), self.x_dim), device=device) * self.noise_scaling

        molecule_x = protein_pocket_com_before[molecule['idx']] + rand_eps_x

        # could generate new peptides (not implemented currently)
        if self.features_fixed:
            molecule_h = molecule['h'].clone().detach()
        else:
            raise NotImplementedError

        # combine position and features
        xh_mol = torch.cat((molecule_x, molecule_h), dim=1)
        xh_pro = torch.cat((protein_pocket['x'], protein_pocket['h']), dim=1)

        error_mol = scatter_add(torch.sum((mol_target_p - xh_mol[:,:3])**2, dim=-1), molecule['idx'], dim=0)
        rmse = torch.sqrt(error_mol / (3 * molecule['size']))

        if self.com_handling == 'both':
            # old centering approach
            xh_mol[:,:self.x_dim] = xh_mol[:,:self.x_dim] - scatter_mean(xh_mol[:,:self.x_dim], molecule['idx'], dim=0)[molecule['idx']]
            xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - scatter_mean(xh_pro[:,:self.x_dim], protein_pocket['idx'], dim=0)[protein_pocket['idx']]
        elif self.com_handling == 'no_COM':
                dumy_variable = 0
        else:
            # data is translated to 0, COM noise added and again translated to 0
            mean = scatter_mean(xh_mol[:,:self.x_dim], molecule['idx'], dim=0)
            xh_mol[:,:self.x_dim] = xh_mol[:,:self.x_dim] - mean[molecule['idx']]
            xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - mean[protein_pocket['idx']]

        max_T = self.T

        # Only for confidence testing
        if self.confidence_score == True:
            confidence = []

        z_t_mol = xh_mol.clone().detach()

        # Iterativly denoise stepwise for t = T,...,1; stepsize default is 1
        for s in reversed(range(0, max_T, self.sampling_stepsize)):
            
            # time arrays
            s_array = torch.full((num_samples, 1), fill_value=s, device=device)
            t_array = s_array + self.sampling_stepsize
            s_array_norm = s_array / self.T
            t_array_norm = t_array / self.T

            z_t_mol_old = z_t_mol.clone().detach()
            # x_t + self(x_t=x_t, t=t_start) * (t_end - t_start) / 2)
            v_hat_mol, v_hat_pro, c_s = self.neural_net(z_t_mol_old, xh_pro, s_array_norm, molecule['idx'], protein_pocket['idx'], molecule_pos)
            z_t_mol = z_t_mol_old + v_hat_mol * (s_array_norm[0] - t_array_norm[0]) / 2

            # x_t + (t_end - t_start) * self(t=t_start + (t_end - t_start) / 2, x_t=
            t = s_array_norm + (t_array_norm - s_array_norm) / 2
            v_hat_mol, v_hat_pro, c_s = self.neural_net(z_t_mol, xh_pro, t, molecule['idx'], protein_pocket['idx'], molecule_pos)
            z_t_mol = z_t_mol_old + (s_array_norm[0] - t_array_norm[0]) + v_hat_mol

            # Only for confidence testing
            if self.confidence_score == True:
                C_S = scatter_add(c_s, molecule['idx'], dim=0).squeeze(1) / molecule['size']
                confidence += [C_S]

            

            # t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
            # x_t = x_t + (t_end - t_start) * 
            # self(t=t_start + (t_end - t_start) / 2, x_t= x_t + self(x_t=x_t, t=t_start) * (t_end - t_start) / 2)


            if self.com_handling == 'both':
                dumy_variable = 0
            elif self.com_handling == 'no_COM':
                dumy_variable = 0
            else:
                # project both pocket and peptide to 0 COM again (only mol mean changes)
                mean = scatter_mean(z_t_mol[:,:self.x_dim], molecule['idx'], dim=0)
                z_t_mol[:,:self.x_dim] = z_t_mol[:,:self.x_dim] - mean[molecule['idx']]
                xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - mean[protein_pocket['idx']]

            if self.com_handling == 'both':
                # old centering approach
                z_t_mol[:,:self.x_dim] = z_t_mol[:,:self.x_dim] - scatter_mean(z_t_mol[:,:self.x_dim], molecule['idx'], dim=0)[molecule['idx']]
                xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - scatter_mean(xh_pro[:,:self.x_dim], protein_pocket['idx'], dim=0)[protein_pocket['idx']]
            else:
                dumy_variable = 0



        xh_mol_final = z_t_mol.clone().detach()
        xh_pro_final = xh_pro.clone().detach()

        if self.com_handling == 'both':
                dumy_variable = 0
        elif self.com_handling == 'no_COM':
                dumy_variable = 0
        else:
            # project both pocket and peptide to 0 COM again (only mol mean changes)
            mean = scatter_mean(xh_mol_final[:,:self.x_dim], molecule['idx'], dim=0)
            xh_mol_final[:,:self.x_dim] = xh_mol_final[:,:self.x_dim] - mean[molecule['idx']]
            xh_pro_final[:,:self.x_dim] = xh_pro_final[:,:self.x_dim] - mean[protein_pocket['idx']]

        # Unnormalisation
        x_mol_final = xh_mol_final[:,:self.x_dim] * self.norm_values[0]
        h_mol_final = xh_mol_final[:,self.x_dim:] * self.norm_values[0]
        x_pro_final = xh_pro_final[:,:self.x_dim] * self.norm_values[0]
        h_pro_final = xh_pro_final[:,self.x_dim:] * self.norm_values[0]

        # Round h to one_hot encoding
        h_mol_final = F.one_hot(torch.argmax(h_mol_final, dim=1), self.num_atoms)

        # Recombine x and h
        xh_mol_final = torch.cat([x_mol_final, h_mol_final], dim=1)
        xh_pro_final = torch.cat([x_pro_final, h_pro_final], dim=1)

        # Correct for center of mass difference
        protein_pocket_com_after = scatter_mean(x_pro_final, protein_pocket['idx'], dim=0)

        # Testing if we only learn the form
        # Moving mol targets COM to 0
        mol_target = molecule['x']  - scatter_mean(molecule['x'], molecule['idx'], dim=0)[molecule['idx']]

        xh_mol_final[:,:self.x_dim] += (protein_pocket_com_before - protein_pocket_com_after)[molecule['idx']]
        xh_pro_final[:,:self.x_dim] += (protein_pocket_com_before - protein_pocket_com_after)[protein_pocket['idx']]

        # Moving mol targets COM to original COM
        mol_target += (protein_pocket_com_before - protein_pocket_com_after)[molecule['idx']]

        sampled_structures = (xh_mol_final, xh_pro_final, c_s)

        self.safe_pdbs(xh_mol_final, molecule, run_id, data_dir, time_step='F')

        # Only for confidence testing
        if self.confidence_score == True:
            print(C_S)  
        
        return sampled_structures
    
    def safe_pdbs(self, pos, molecule, run_id, data_dir, time_step):

        for i in range(len(molecule['size'])):
            # (1) extract the peptide position
            pos = pos[:,:3]
            peptide_pos = pos[molecule['idx'] == i]
            # (2) bring peptides into correct order
            peptide_idx = molecule['pos_in_seq'][molecule['idx'] == i]
            # peptide_pos_orderd = peptide_pos[peptide_idx-1] # pos starts at 1
            # (3) get graph name for elemnt in batch
            if isinstance(molecule['graph_name'], str):
                graph_name = molecule['graph_name']
            else:
                graph_name = molecule['graph_name'][i]

            if '100K' in data_dir:

                create_new_pdb_hdf5_100k(peptide_pos, peptide_idx, graph_name, run_id, data_dir, time_step=time_step, sample_id=i)

            else:

                create_new_pdb_hdf5(peptide_pos, peptide_idx, graph_name, run_id, data_dir, time_step=time_step, sample_id=i)



