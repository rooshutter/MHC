import torch
import numpy as np
import math
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean
# from zmq import device

from model.noise_schedule import Noise_Schedule
from utils import create_new_pdb_hdf5, create_new_pdb_hdf5_100k

from torchdiffeq import odeint_adjoint as odeint

from tools.rigid import Rigid


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
        all_atom = False,

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

        self.all_atom = all_atom

    def forward(self, z_data):

        molecule, protein_pocket = z_data

        if self.position_encoding:
            molecule_pos = molecule['pos_in_seq']
        else:
            molecule_pos = None

        # compute noised samples
        # z_t_mol: noised molecule at time t
        # z_t_pro: protein pocket at time t
        # eps_x_mol: actual noise added to molecule positions
        # epsilon_pro: actual noise added to protein pocket positions (should be 0)
        # z_t_mol, z_t_pro, eps_x_mol, epsilon_pro, t = self.noise_process(z_data)
        if self.all_atom:
            z_t_mol, z_t_pro, v_x_mol, v_pro, t = self.compute_flow_match_all_atom(z_data)
        else:
            z_t_mol, z_t_pro, v_x_mol, v_pro, t = self.compute_flow_match(z_data)

        print(f"{t.shape=}")

        if self.noise_scaling > 0:
            z_t_mol = z_t_mol + torch.randn_like(z_t_mol) * self.noise_scaling

        
        # use neural netwrok to predict vector field
        # epsilon_hat_mol: predicted noise for molecule positions
        # epsilon_hat_pro: predicted noise for protein pocket positions
        # c_s: predicted confidence score
        # epsilon_hat_mol, epsilon_hat_pro, c_s = self.neural_net(z_t_mol, z_t_pro, t, molecule['idx'], protein_pocket['idx'], molecule_pos)
        v_hat_mol, v_hat_pro, c_s = self.neural_net(z_t_mol, z_t_pro, t, molecule['idx'], protein_pocket['idx'], molecule_pos)


        # --- RMSE Calculation ---
        # with torch.no_grad():
        #     x0_hat = z_t_mol[:, :3] + t[molecule['idx']] * v_hat_mol[:, :3]

        #     # 2. Scale back to original Angstroms (Unnormalization)
        #     x0_hat_ang = x0_hat * self.norm_values[0]
        #     target_ang = molecule['x'] # Ground truth in Angstroms

        #     # 3. Structural RMSE (Optional but recommended: align COM)
        #     # This prevents translation errors from blowing up your RMSE
        #     x0_hat_centered = x0_hat_ang - scatter_mean(x0_hat_ang, molecule['idx'], dim=0)[molecule['idx']]
        #     target_centered = target_ang - scatter_mean(target_ang, molecule['idx'], dim=0)[molecule['idx']]

        #     # 4. Compute RMSE per peptide (Matching your Eval Script logic)
        #     print(f"Centered predicted positions (Angstroms): {x0_hat_centered[0]}")
        #     print(f"Centered target positions (Angstroms):    {target_centered[0]}")
        #     error_mol = scatter_add(torch.sum((x0_hat_centered - target_centered)**2, dim=-1), molecule['idx'], dim=0)
        #     rmse_per_peptide = torch.sqrt(error_mol / molecule['size'])

        #     # 5. Average across the batch
        #     batch_rmse = rmse_per_peptide.mean().item()
            
        #     print(f"Batch RMSE: {batch_rmse:.4f}") # Helpful for debugging
        # ------------------------
        
        if self.training:
            loss, info = self.train_loss(molecule, z_t_mol, v_x_mol, 
                                        v_hat_mol, protein_pocket, 
                                        z_t_pro, v_pro, v_hat_pro, t, c_s)
        else: 
            loss, info = self.validation_loss(z_data, molecule, z_t_mol, v_x_mol, 
                                        v_hat_mol, protein_pocket, 
                                        z_t_pro, v_pro, v_hat_pro, t)

        return loss.mean(0), info
    
    def compute_flow_match_all_atom(self, z_data, t_is_0 = False):
        molecule, protein_pocket = z_data
        batch_size = molecule['size'].size(0)
        device = molecule['x'].device
                
        size_mol = molecule['size'][0]
        size_pro = protein_pocket['size'][0]
        # print(f"{molecule['x'].shape=}")
        print(f"{molecule['h'].shape=}")
        # molecule['x'] = molecule['x'].view(batch_size, size_mol, *molecule['x'].shape[1:])
        molecule['h'] = molecule['h'].view(batch_size, size_mol, *molecule['h'].shape[1:])
        # print(f"{molecule['x'].shape=}")
        print(f"{molecule['h'].shape=}")
        # print(f"{protein_pocket['x'].shape=}")
        print(f"{protein_pocket['h'].shape=}")
        # protein_pocket['x'] = protein_pocket['x'].view(batch_size, size_pro, *protein_pocket['x'].shape[1:])
        protein_pocket['h'] = protein_pocket['h'].view(batch_size, size_pro, *protein_pocket['h'].shape[1:])
        # print(f"{protein_pocket['x'].shape=}")
        print(f"{protein_pocket['h'].shape=}")

        print(f"{molecule['torsion_angles_sin_cos'].shape=}")
        print(f"{molecule['backbone_rigid_tensor'].shape=}")
        print(f"{protein_pocket['backbone_rigid_tensor'].shape=}")
        molecule['torsion_angles_sin_cos'] = molecule['torsion_angles_sin_cos'].view(batch_size, size_mol, *molecule['torsion_angles_sin_cos'].shape[1:])
        molecule['backbone_rigid_tensor'] = molecule['backbone_rigid_tensor'].view(batch_size, size_mol, *molecule['backbone_rigid_tensor'].shape[1:])
        protein_pocket['backbone_rigid_tensor'] = protein_pocket['backbone_rigid_tensor'].view(batch_size, size_pro, *protein_pocket['backbone_rigid_tensor'].shape[1:])
        print(f"{molecule['torsion_angles_sin_cos'].shape=}")
        print(f"{molecule['backbone_rigid_tensor'].shape=}")
        print(f"{protein_pocket['backbone_rigid_tensor'].shape=}")

        
        # normalisation with norm_values (dataset dependend) -> changes likelyhood (adjusted for in vlb)!
        # molecule['x'] = molecule['x'] / self.norm_values[0]
        molecule['h'] = molecule['h'] / self.norm_values[1]
        # protein_pocket['x'] = protein_pocket['x'] / self.norm_values[0]
        protein_pocket['h'] = protein_pocket['h'] / self.norm_values[1]

        
        # sample t ~ U(0,...,T) for each graph individually
        t_low = 0 if self.train else 1
        t = torch.randint(t_low, self.T + 1, size=(batch_size, 1, 1), device=device)
        
        # normalize t
        t = t / self.T

        # option for computing t = 0 representations
        t = torch.zeros((batch_size, 1, 1), device=device) if t_is_0 else t
        
        # prepare joint point cloud
        peptide_backbone_rigid_tensor = molecule['backbone_rigid_tensor']
        T_peptide = Rigid.from_tensor_4x4(peptide_backbone_rigid_tensor)
        T_peptide = T_peptide.to_tensor_7()
        print(f"{T_peptide.shape=}")

        xh_mol = torch.cat((T_peptide, molecule['h']), dim=-1)
        print(f"{xh_mol.shape=}")

        protein_backbone_rigid_tensor = protein_pocket['backbone_rigid_tensor']
        T_protein = Rigid.from_tensor_4x4(protein_backbone_rigid_tensor)
        T_protein = T_protein.to_tensor_7()
        print(f"{T_protein.shape=}")

        xh_pro = torch.cat((T_protein, protein_pocket['h']), dim=-1)
        print(f"{xh_pro.shape=}")
        

        # center of mass handling
        # print("com_handling:", self.com_handling) peptide
        # if self.com_handling == 'both':
        #     # old centering approach
        #     xh_mol[:,:self.x_dim] = xh_mol[:,:self.x_dim] - scatter_mean(xh_mol[:,:self.x_dim], molecule['idx'], dim=0)[molecule['idx']]
        #     xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - scatter_mean(xh_pro[:,:self.x_dim], protein_pocket['idx'], dim=0)[protein_pocket['idx']]
        # elif self.com_handling == 'no_COM':
        #     dumy_variable = 0
        # else:
        #     # data is translated to 0, COM noise added and again translated to 0
        #     mean = scatter_mean(xh_mol[:,:self.x_dim], molecule['idx'], dim=0)
        #     xh_mol[:,:self.x_dim] = xh_mol[:,:self.x_dim] - mean[molecule['idx']]
        #     xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - mean[protein_pocket['idx']]

        T_peptide_z = Rigid.identity(
            molecule['h'].shape[:-1],
            molecule['h'].dtype,
            device,
            self.training,
            fmt="quat",
        )
        print(f"Using all-atom noise with shape: {T_peptide_z.shape}")
        
        z_x_pro = torch.zeros(size=(len(xh_pro), xh_pro.shape[1], T_peptide.shape[-1]), device=device)
        print(f"{z_x_pro.shape=}")

        # print(f"{self.com_handling=}") peptide
        # if self.com_handling == 'both':
        #     # alternative centering approach
        #     z_x_mol = z_x_mol - scatter_mean(z_x_mol, molecule['idx'], dim=0)[molecule['idx']]
        #     z_x_pro = torch.zeros(size=(len(xh_pro), self.x_dim), device=device)
        # else:
        #     dumy_variable = 0

        # print("features_fixed:", self.features_fixed) true
        if self.features_fixed:
            print(f"{len(xh_mol)=}")
            z_h_mol = torch.zeros(size=(len(xh_mol), xh_mol.shape[1], self.num_atoms), device=device)
            # z_h_mol = torch.zeros(size=(len(xh_mol), self.num_atoms), device=device)
            z_h_pro = torch.zeros(size=(len(xh_pro), xh_pro.shape[1], self.num_residues), device=device)
        else:
            # for h we need standard normal noise (this would be sampling new peptides)
            z_h_mol = torch.randn(size=(len(xh_mol), xh_mol.shape[1], self.num_atoms), device=device)
            z_h_pro = torch.randn(size=(len(xh_pro), xh_pro.shape[1], self.num_residues), device=device)

        # z_mol = torch.cat((z_x_mol, z_h_mol), dim=1)
        print(f"{z_x_pro.shape=}")
        print(f"{z_h_pro.shape=}")
        z_pro = torch.cat((z_x_pro, z_h_pro), dim=-1)

        # z_t_mol_x = t[molecule['idx']] * molecule['x'] + (1 - t[molecule['idx']]) * z_x_mol
        # z_t_mol_x = (1 - t[molecule['idx']]) * molecule['x'] + (t[molecule['idx']]) * z_x_mol
        print(f"{T_peptide.shape=}")
        print(f"{T_peptide_z.shape=}")
        T_peptide_z = T_peptide_z.to_tensor_7()
        print(f"{T_peptide_z.shape=}")

        print(f"{t.shape=}")
        # print(f"{t[molecule['idx']].shape=}")
        T_peptide_t = (1 - t) * T_peptide
        T_peptide_t += t * T_peptide_z
        print(f"{T_peptide_t.shape=}")
        print(f"{z_h_mol.shape=}")

        z_t_mol = torch.cat((T_peptide_t, z_h_mol), dim=-1)
        
        z_t_pro = xh_pro.clone().detach()


        # if self.com_handling == 'both':
        #     dumy_variable = 0
        # elif self.com_handling == 'no_COM':
        #     dumy_variable = 0
        # else:
        #     # data is translated to 0, COM noise added and again translated to 0 (turn off for old centering approach)
        #     mean = scatter_mean(z_t_mol[:,:self.x_dim], molecule['idx'], dim=0)
        #     z_t_mol[:,:self.x_dim] = z_t_mol[:,:self.x_dim] - mean[molecule['idx']]
        #     z_t_pro[:,:self.x_dim] = z_t_pro[:,:self.x_dim] - mean[protein_pocket['idx']]


        # v = x_1 - x_0
        print(f"{xh_mol.shape=}")
        print(f"{xh_mol[:,:,:T_peptide_z.shape[-1]].shape=}")
        print(f"{T_peptide_z.shape=}")
        v_x_mol = xh_mol[:,:,:T_peptide_z.shape[-1]] - T_peptide_z 
        print(f"{v_x_mol.shape=}")
        
        print(f"{xh_pro.shape=}")
        print(f"{z_pro.shape=}")
        v_pro = xh_pro - z_pro

        # if self.com_handling == 'both':
        #     dumy_variable = 0
        # elif self.com_handling == 'no_COM':
        #     dumy_variable = 0
        # else:
        #     mean = scatter_mean(v_x_mol, molecule['idx'], dim=0)
        #     v_x_mol = v_x_mol - mean[molecule['idx']]
        #     v_pro[:,:self.x_dim] = v_pro[:,:self.x_dim] - mean[protein_pocket['idx']]


        return z_t_mol, z_t_pro, v_x_mol, v_pro, t

    def compute_flow_match(self, z_data, t_is_0 = False):

        molecule, protein_pocket = z_data
        batch_size = molecule['size'].size(0)
        device = molecule['x'].device
        print(f"{batch_size=}")
    
        
        # normalisation with norm_values (dataset dependend) -> changes likelyhood (adjusted for in vlb)!
        molecule['x'] = molecule['x'] / self.norm_values[0]
        molecule['h'] = molecule['h'] / self.norm_values[1]
        protein_pocket['x'] = protein_pocket['x'] / self.norm_values[0]
        protein_pocket['h'] = protein_pocket['h'] / self.norm_values[1]

        
        # sample t ~ U(0,...,T) for each graph individually
        t_low = 0 if self.train else 1
        if self.all_atom:
            t = torch.randint(t_low, self.T + 1, size=(batch_size, 1, 1), device=device)
        else:
            t = torch.randint(t_low, self.T + 1, size=(batch_size, 1), device=device)

        # normalize t
        t = t / self.T

        # option for computing t = 0 representations
        if self.all_atom:
            t = torch.zeros((batch_size, 1, 1), device=device) if t_is_0 else t
        else:
            t = torch.zeros((batch_size, 1), device=device) if t_is_0 else t
        
        # prepare joint point cloud
        # print(f"{molecule['x'].shape=}")
        # print(f"{molecule['h'].shape=}")
        # print(f"{protein_pocket['x'].shape=}")
        # print(f"{protein_pocket['h'].shape=}")
        if self.all_atom:
            peptide_backbone_rigid_tensor = molecule['backbone_rigid_tensor']
            print(f"peptide_backbone_rigid_tensor shape: {peptide_backbone_rigid_tensor.shape}")
            T_peptide = Rigid.from_tensor_4x4(peptide_backbone_rigid_tensor)
            # T_peptide = T_peptide.to_tensor_7()
            print(f"T_peptide shape: {T_peptide.shape}")

            mol_h = molecule['h'].unsqueeze(1)
            mol_h = mol_h.expand(-1, molecule['x'].shape[1], -1)
            xh_mol = torch.cat((molecule['x'], mol_h), dim=-1)

            protein_backbone_rigid_tensor = protein_pocket['backbone_rigid_tensor']
            T_protein = Rigid.from_tensor_4x4(protein_backbone_rigid_tensor)
            print(f"T_protein shape: {T_protein.shape}")

            pro_h = protein_pocket['h'].unsqueeze(1)
            pro_h = pro_h.expand(-1, protein_pocket['x'].shape[1], -1)
            xh_pro = torch.cat((protein_pocket['x'], pro_h), dim=-1)
        else:
            xh_mol = torch.cat((molecule['x'], molecule['h']), dim=1)
            # xh_mol = torch.cat((molecule['x'], molecule['h']), dim=-1)
            xh_pro = torch.cat((protein_pocket['x'], protein_pocket['h']), dim=1)
            # xh_pro = torch.cat((protein_pocket['x'], protein_pocket['h']), dim=-1)

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

        # print(f"Centered molecule positions:                             {xh_mol[:,:self.x_dim][0]}")
        
        if self.all_atom:
            print(f"{molecule['h'].shape=}")
            print(f"{molecule['x'].shape=}")
            print(f"{protein_pocket['h'].shape=}")
            print(f"{protein_pocket['x'].shape=}")
            print(f"{molecule['size']=}")
            print(f"{molecule['size'].shape=}")
            T_peptide = Rigid.identity(
                (molecule['x'].shape[0], molecule['size']),
                molecule['x'].dtype,
                device,
                self.training,
                fmt="quat",
            )
            z_x_mol = T_peptide
            print(f"Using all-atom noise with shape: {z_x_mol.shape}")
        else:
            # compute noised sample z_t
            # for x cord. we mean center the normal noise for each graph
            # we only diffuse position of the molecules
            z_x_mol = torch.randn(size=(len(xh_mol), self.x_dim), device=device) #* self.noise_scaling
        
        z_x_pro = torch.zeros(size=(len(xh_pro), self.x_dim), device=device)


        # print(f"Noise positions:                                         {z_x_mol[0]}")

        # print(f"{self.com_handling=}") peptide
        # if self.com_handling == 'both':
        #     # alternative centering approach
        #     z_x_mol = z_x_mol - scatter_mean(z_x_mol, molecule['idx'], dim=0)[molecule['idx']]
        #     z_x_pro = torch.zeros(size=(len(xh_pro), self.x_dim), device=device)
        # else:
        #     dumy_variable = 0
        # print(f"Noised molecule positions after centering:               {z_x_mol[0]}")

        # print("features_fixed:", self.features_fixed) true
        if self.features_fixed:
            z_h_mol = torch.zeros(size=(len(xh_mol), self.num_atoms), device=device)
            z_h_pro = torch.zeros(size=(len(xh_pro), self.num_residues), device=device)
        else:
            # for h we need standard normal noise (this would be sampling new peptides)
            z_h_mol = torch.randn(size=(len(xh_mol), self.num_atoms), device=device)
            z_h_pro = torch.randn(size=(len(xh_pro), self.num_residues), device=device)

        # z_mol = torch.cat((z_x_mol, z_h_mol), dim=1)
        z_pro = torch.cat((z_x_pro, z_h_pro), dim=1)

        # z_t_mol_x = t[molecule['idx']] * molecule['x'] + (1 - t[molecule['idx']]) * z_x_mol
        # z_t_mol_x = (1 - t[molecule['idx']]) * molecule['x'] + (t[molecule['idx']]) * z_x_mol
        # print(f"{t[molecule['idx']].shape=}")
        # print(f"{xh_mol.shape=}")
        # print(f"{xh_mol[:,:,:self.x_dim].shape=}")
        # print(f"{z_x_mol.shape=}")
        # print(f"{T_peptide.shape=}")
        if self.all_atom:
            z_t_mol_x = (1 - t[molecule['idx']]) * T_peptide
            z_t_mol_x += (t[molecule['idx']]) * z_x_mol
        else:
            z_t_mol_x = (1 - t[molecule['idx']]) * xh_mol[:,:self.x_dim] + (t[molecule['idx']]) * z_x_mol
        z_t_mol = torch.cat((z_t_mol_x, z_h_mol), dim=1)
        z_t_pro = xh_pro.clone().detach()

        # print(f"Noised molecule positions at time t:                     {z_t_mol_x[0]}")

        if self.com_handling == 'both':
            dumy_variable = 0
        elif self.com_handling == 'no_COM':
            dumy_variable = 0
        else:
            # data is translated to 0, COM noise added and again translated to 0 (turn off for old centering approach)
            mean = scatter_mean(z_t_mol[:,:self.x_dim], molecule['idx'], dim=0)
            z_t_mol[:,:self.x_dim] = z_t_mol[:,:self.x_dim] - mean[molecule['idx']]
            z_t_pro[:,:self.x_dim] = z_t_pro[:,:self.x_dim] - mean[protein_pocket['idx']]

        # print(f"Noised molecule positions at time t after centering:     {z_t_mol[:,:self.x_dim][0]}")

        # v = x_1 - x_0
        # xh_mol[:,:self.x_dim] - z_x_mol
        # v_target_mol = xh_mol - z_mol
        # v_target_pro = xh_pro - z_pro
        # print(f"Original molecule positions:                             {xh_mol[:,:self.x_dim][0]}")
        # print(f"Noised molecule positions:                               {z_x_mol[0]}")
        v_x_mol = xh_mol[:,:self.x_dim] - z_x_mol 
        # v_x_mol = z_x_mol - xh_mol[:,:self.x_dim]

        # print(f"True velocity:                                           {v_x_mol[0]}")
        v_pro = xh_pro - z_pro
        # v_pro = z_pro - xh_pro

        if self.com_handling == 'both':
            dumy_variable = 0
        elif self.com_handling == 'no_COM':
            dumy_variable = 0
        else:
            mean = scatter_mean(v_x_mol, molecule['idx'], dim=0)
            v_x_mol = v_x_mol - mean[molecule['idx']]
            v_pro[:,:self.x_dim] = v_pro[:,:self.x_dim] - mean[protein_pocket['idx']]

        # print(f"True velocity after centering:                           {v_x_mol[0]}")

        return z_t_mol, z_t_pro, v_x_mol, v_pro, t

    def train_loss(
            self, molecule, z_t_mol, v_x_mol, 
            v_hat_mol, protein_pocket, 
            z_t_pro, v_pro, v_hat_pro, t, c_s
    ):
        
        # compute the sum squared error loss per graph # TODO: modified to not take the h_dims
        error_mol = scatter_add(torch.sum((v_x_mol[:,:3] - v_hat_mol[:,:3])**2, dim=-1), molecule['idx'], dim=0)
        error_pro = torch.zeros(protein_pocket['size'].size(0), device=molecule['x'].device)

        kl_prior = self.kl_prior(molecule)

        # Add a SNR modulation term to upweight highly noised samples (default: turned off)
        # SNR_t = (1 / self.SNR_t(t).squeeze(1))

        # t = 0 and t != 0 masks for seperate computation of log p(x | z0)
        t_0_mask = (t == 0).float().squeeze()
        t_not_0_mask = 1 - t_0_mask

        # likelyhood of drawing our structure from our completley denoised distribution
        loss_x_mol_t0, loss_x_protein_t0, loss_h_t0 = self.loss_t0(
            molecule, z_t_mol, v_x_mol, v_hat_mol,
            protein_pocket, z_t_pro, v_pro, v_hat_pro, t
        )

        # seperate loss computation for t = 0 and t != 0
        print(f"{loss_x_mol_t0.shape=}")
        print(f"{t_0_mask.shape=}")
        loss_x_mol_t0 = - loss_x_mol_t0 * t_0_mask
        loss_x_protein_t0 = - loss_x_protein_t0 * t_0_mask
        loss_h_t0 = - loss_h_t0 * t_0_mask
        error_mol = error_mol * t_not_0_mask
        error_pro = error_pro * t_not_0_mask

        # Normalize loss_t by graph size
        error_mol = error_mol / ((self.x_dim) * molecule['size'])
        error_pro = error_pro / ((self.x_dim + self.num_residues * protein_pocket['size']))
        loss_t = 0.5 * (error_mol + error_pro) # * SNR_t

        # Normalize loss_0 by graph size
        loss_x_mol_t0 = loss_x_mol_t0 / (self.x_dim * molecule['size'])
        loss_x_protein_t0 = loss_x_protein_t0 / (self.x_dim * protein_pocket['size'])
        loss_0 = loss_x_mol_t0 + loss_x_protein_t0 + loss_h_t0

        loss = loss_t + loss_0 + kl_prior
        # loss = loss_t + kl_prior

        if self.confidence_score == True:

            c_s_peptide = scatter_add(c_s, molecule['idx'], dim=0).squeeze(1) / molecule['size']

            # confidence weighted loss
            loss_with_conf = 1/(c_s_peptide)**2 * loss + torch.log(c_s_peptide**2)
        else:
            c_s_peptide = torch.zeros_like(loss)
            loss_with_conf = torch.zeros_like(loss)
            

        info = {
            'loss_t': loss_t.mean(0),
            'loss_0': loss_0.mean(0),
            'error_mol': error_mol.mean(0),
            'loss_x_mol_t0': loss_x_mol_t0.mean(0),
            'kl_prior': kl_prior.mean(0),
            'confidence': c_s_peptide.mean(0),
            'loss_with_conf': loss_with_conf.mean(0)
        }

        if self.confidence_score == True:
            return loss_with_conf, info

        return loss, info
    
    def validation_loss(
            self, z_data, molecule, z_t_mol, v_x_mol, v_hat_mol,
            protein_pocket, z_t_pro, v_pro, v_hat_pro, 
            t,
    ):
        
        ### Additional evaluation (VLB) variables
        if self.position_encoding:
            molecule_pos = molecule['pos_in_seq']
        else:
            molecule_pos = None

        # compute the sum squared error loss per graph
        error_mol = scatter_add(torch.sum((v_x_mol[:,:3] - v_hat_mol[:,:3])**2, dim=-1), molecule['idx'], dim=0)
        error_pro = torch.zeros(protein_pocket['size'].size(0), device=molecule['x'].device)

        kl_prior = self.kl_prior(molecule)

        # if pocket not fixed then molecule['size'] + protein_pocket['size']
        neg_log_const = self.neg_log_const(molecule['size'], molecule['size'].size(0), device=molecule['x'].device)
        delta_log_px = self.delta_log_px(molecule['size'])

        # SNR is computed between timestep s and t (with s = t-1)
        SNR_weight = (1 - self.SNR_s_t(t).squeeze(1))

        # TODO: add log_pN computation using the dataset histogram
        log_pN = self.log_pN(molecule['size'], protein_pocket['size'])

        # TODO optional: can add auxiliary loss / lennard-jones potential

        ## For evaluation we want to compute t = 0 losses for all z_data samples that we have

        # compute noised sample for t = 0
        # z_0_mol, z_0_pro, epsilon_0_mol, epsilon_0_pro, t_0 = self.noise_process(z_data, t_is_0 = True)
        z_0_mol, z_0_pro, v_target_0_mol, v_target_0_pro, t_0 = self.compute_flow_match(z_data, t_is_0 = True)

        # use neural network to predict noise for t = 0
        # v_hat_mol, v_hat_pro, c_s = self.neural_net(z_t_mol, z_t_pro, t, molecule['idx'], protein_pocket['idx'], molecule_pos)
        # epsilon_hat_0_mol, epsilon_hat_0_pro, _ = self.neural_net(z_0_mol, z_0_pro, t_0, molecule['idx'], protein_pocket['idx'], molecule_pos)
        v_hat_0_mol, v_hat_0_pro, _ = self.neural_net(z_0_mol, z_0_pro, t_0, molecule['idx'], protein_pocket['idx'], molecule_pos)

        loss_x_mol_t0, loss_x_protein_t0, loss_h_t0 = self.loss_t0(
            molecule, z_0_mol, v_target_0_mol, v_hat_0_mol,
            protein_pocket, z_0_pro, v_target_0_pro, v_hat_0_pro, t_0
        )

        loss_x_mol_t0 = - loss_x_mol_t0
        loss_x_protein_t0 = - loss_x_protein_t0
        loss_h_t0 = - loss_h_t0


        # loss_t = - self.T * 0.5 * SNR_weight * (error_mol + error_pro)
        loss_t = self.T * 0.5 * (error_mol + error_pro)
        loss_0 = loss_x_mol_t0 + loss_x_protein_t0 + loss_h_t0
        loss_0 = loss_0 + neg_log_const

        # Two added loss terms for vlb
        loss = loss_t + loss_0 + kl_prior - delta_log_px - log_pN
        # loss = loss_t + kl_prior

        info = {
            'loss_t': loss_t.mean(0),
            'loss_0': loss_0.mean(0),
            'error_mol': error_mol.mean(0),
            'loss_x_mol_t0': loss_x_mol_t0.mean(0),
            'kl_prior': kl_prior.mean(0),
            'neg_log_const': neg_log_const.mean(0),
            'SNR_weight': SNR_weight.mean(0)
        }

        return loss, info
        
    
    def loss_t0(
            self, molecule, z_t_mol, v_target_mol, v_hat_mol,
            protein_pocket, z_t_pro, v_target_pro, v_hat_pro, 
            t, epsilon=1e-10
    ):
        """
        This function calculate log(p(xh|z_0))
        """

        ## Normal computation of position error when sampling from fully denoised distribution

        v_target_mol_x = v_target_mol[:,:self.x_dim]
        v_hat_mol_x = v_hat_mol[:,:self.x_dim]
        loss_x_mol_t0 = - 0.5 * scatter_add(torch.sum((v_target_mol_x - v_hat_mol_x)**2, dim=-1), molecule['idx'], dim=0)

        loss_x_protein_t0 = torch.zeros(protein_pocket['size'].size(0), device=molecule['x'].device)

        # print(f"{self.features_fixed=}")

        if self.features_fixed:

            loss_h_t0 = torch.zeros(molecule['size'].size(0), device=molecule['x'].device)

        else:
            ## Computation for changed features

            
            sigma_0 = self.noise_schedule(t, 'sigma')
            sigma_0_unnormalized = sigma_0 * self.norm_values[1]
            # unnormalize not necessary for molecule['h'] because molecule was only locally normalized (can change that if necessary later)
            mol_h_hat = z_t_mol[:, self.x_dim:] * self.norm_values[1]
            mol_h_hat_centered = mol_h_hat - 1

            # Compute integrals from 0.5 to 1.5 of the normal distribution
            # N(mean=z_h_cat, stdev=sigma_0_cat)
            # 0.5 * (1. + torch.erf(x / math.sqrt(2)))
            log_probabilities_mol_unnormalized = torch.log(
                0.5 * (1. + torch.erf((mol_h_hat_centered + 0.5) / sigma_0_unnormalized[molecule['idx']]) / math.sqrt(2)) \
                - 0.5 * (1. + torch.erf((mol_h_hat_centered - 0.5) / sigma_0_unnormalized[molecule['idx']]) / math.sqrt(2)) \
                + epsilon
            )

            # Normalize the distribution over the categories.
            log_Z = torch.logsumexp(log_probabilities_mol_unnormalized, dim=1,
                                    keepdim=True)
            
            log_probabilities_mol = log_probabilities_mol_unnormalized - log_Z

            loss_h_t0 = scatter_add(torch.sum(log_probabilities_mol * molecule['h'], dim=-1), molecule['idx'], dim=0)
        
        return loss_x_mol_t0, loss_x_protein_t0, loss_h_t0
    
    def kl_prior(self, molecule):

        device=molecule['x'].device

        molecule['x'] = molecule['x'] - scatter_mean(molecule['x'], molecule['idx'], dim=0)[molecule['idx']]
       
        # z_t_mol_x = alpha_t[molecule['idx']] * xh_mol[:, :self.x_dim] + sigma_t[molecule['idx']] * eps_x_mol
        # z_t_mol_x = (1 - t[molecule['idx']]) * molecule['x'] + (t[molecule['idx']]) * z_x_mol


        T_normalized = torch.ones((len(molecule['size']), 1), device=device)
        # alpha_T = self.noise_schedule(T_normalized, 'alpha')
        # sigma_T = self.noise_schedule(T_normalized, 'sigma')
        # sigma_T_value = sigma_T[0,0].item()
        alpha_T = 0.0  # At t=1, (1-t) is 0
        sigma_T_val = 1.0

        # mu_x_mol = molecule['x'] * alpha_T[molecule['idx']] # [:,3]
        mu_x_mol = molecule['x'] * alpha_T # [:,3]
        # mu_h_mol = molecule['h'] * alpha_T[molecule['idx']] # [:,20]
        
        # sigma_T_x = torch.full(alpha_T.shape, fill_value=sigma_T_value, device=device).squeeze() # [64,1]
        # sigma_T_h = torch.full(alpha_T.shape, fill_value=sigma_T_value, device=device).squeeze() # [64,1]
        # sigma_T_x = torch.full((len(molecule['size']),), fill_value=sigma_T_val * self.noise_scaling, device=device)
        sigma_T_x = torch.full((len(molecule['size']),), fill_value=sigma_T_val, device=device)
        # sigma_T_h = torch.full((len(molecule['size']),), fill_value=sigma_T_val, device=device)


        # KL computation h (if features are diffused)
        kl_h = 0
        # zeros = torch.zeros_like(mu_h_mol)
        # ones = torch.ones_like(sigma_T_h)
        # mu_norm2 = scatter_add(torch.sum((mu_h_mol - zeros) ** 2, dim=1), molecule['idx'], dim=0)
        # kl_h = torch.log(ones / sigma_T_h) + 0.5 * (sigma_T_h**2 + mu_norm2) / (ones**2) - 0.5

        # KL computation x
        zeros = torch.zeros_like(mu_x_mol)
        ones = torch.ones_like(sigma_T_x) #* self.noise_scaling
        mu_norm2 = scatter_add(torch.sum((mu_x_mol - zeros) ** 2, dim=-1), molecule['idx'], dim=0)
        d = (molecule['size'] - 1) * self.x_dim
        kl_x = d * torch.log(ones / sigma_T_x) + 0.5 * (d * sigma_T_x**2 + mu_norm2) / (ones**2) - 0.5 * d

        kl_loss = kl_x + kl_h

        return kl_loss
    
    def delta_log_px(self, num_nodes):

        delta_log_px = - (num_nodes - 1) * self.x_dim * np.log(self.norm_values[0])

        return delta_log_px
    
    def log_pN(self, molecule_N, protein_pocket_N):

        # add log_pN computation using the dataset histogram
        # only matters for diverse molecule sizes, therefore we set it to 0
        log_pN = 0

        return log_pN
    
    def neg_log_const(self, num_nodes, batch_size, device):

        # t0 = torch.zeros((batch_size, 1), device=device)
        # log_sigma_0 = torch.log(self.noise_schedule(t0, 'sigma')).view(batch_size)

        # neg_log_const = - ((num_nodes - 1) * self.x_dim) * (- log_sigma_0 - 0.5 * np.log(2 * np.pi))

        # return neg_log_const
        return torch.zeros(batch_size, device=device)
    
    def SNR_s_t(self, t):

        # 1. Define previous timestep s
        s = torch.clamp(torch.round(t * self.T).long() - 1, min=0)
        s = s / self.T

        # 2. In Flow Matching: x_t = (1-t)x_0 + t*x_1
        # Effective alpha (signal weight) is (1-t)
        # Effective sigma (noise weight) is t
        alpha2_t = (1.0 - t)**2
        alpha2_s = (1.0 - s)**2
        
        sigma2_t = t**2
        sigma2_s = s**2

        # 3. Compute SNR as the ratio of signal-to-noise ratios
        # Adding epsilon to avoid division by zero at t=0
        eps = 1e-8
        snr_t = alpha2_t / (sigma2_t + eps)
        snr_s = alpha2_s / (sigma2_s + eps)

        # This represents the "step-wise" change in SNR
        SNR_s_t = snr_s / (snr_t + eps)

        return SNR_s_t
    

    @torch.no_grad()
    def sample_structure(self, num_samples, molecule, protein_pocket, sampling_without_noise, data_dir, run_id):
        
        device = molecule['x'].device
        num_graphs = molecule['size'].size(0)
        
        if self.position_encoding:
            molecule_pos = molecule['pos_in_seq']
        else:
            molecule_pos = None

        z_x_mol = torch.randn(size=(len(molecule['x']), self.x_dim), device=device) #* self.noise_scaling
        
        mol_norm_x = molecule['x'] / self.norm_values[0]
        protein_pocket['x'] = protein_pocket['x'] / self.norm_values[0]
        protein_pocket['h'] = protein_pocket['h'] / self.norm_values[1]
        xh_pro = torch.cat((protein_pocket['x'], protein_pocket['h']), dim=1)

        if self.com_handling == 'both':
            # old centering approach
            z_x_mol = z_x_mol - scatter_mean(z_x_mol, molecule['idx'], dim=0)[molecule['idx']]
            xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - scatter_mean(xh_pro[:,:self.x_dim], protein_pocket['idx'], dim=0)[protein_pocket['idx']]
        elif self.com_handling == 'no_COM':
                dumy_variable = 0
        else:
            # data is translated to 0, COM noise added and again translated to 0
            mean = scatter_mean(z_x_mol, molecule['idx'], dim=0)
            z_x_mol = z_x_mol - mean[molecule['idx']]
            xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - mean[protein_pocket['idx']]

        if self.features_fixed:
            z_h_mol = (molecule['h'] / self.norm_values[1]).clone().detach()
        else:
            raise NotImplementedError

        current_xh_mol = torch.cat((z_x_mol, z_h_mol), dim=1)
        
        steps = self.T // self.sampling_stepsize
        # print(f"Sampling with {steps} steps and step size {self.sampling_stepsize}")
        dt = 1.0 / steps

        solver = "euler"

        if solver == "loop":

            for i in reversed(range(1, steps + 1)):
                t_val = i / steps
                t_array = torch.full((num_graphs, 1), fill_value=t_val, device=device)

                # Predict velocity v_hat
                v_hat_mol, _, c_s = self.neural_net(
                    current_xh_mol, xh_pro, t_array, 
                    molecule['idx'], protein_pocket['idx'], molecule_pos
                )

                # Euler Step: x_{t+dt} = x_t + v(x_t, t) * dt
                if self.features_fixed:
                    current_xh_mol[:, :self.x_dim] += v_hat_mol[:, :self.x_dim] * dt
                else:
                    current_xh_mol += v_hat_mol * dt

                # if self.com_handling != 'no_COM':
                #     mean = scatter_mean(current_xh_mol[:, :self.x_dim], molecule['idx'], dim=0)
                #     current_xh_mol[:, :self.x_dim] -= mean[molecule['idx']]
                if self.com_handling == 'both':
                    dumy_variable = 0
                elif self.com_handling == 'no_COM':
                    dumy_variable = 0
                else:
                    # project both pocket and peptide to 0 COM again (only mol mean changes)
                    mean = scatter_mean(current_xh_mol[:,:self.x_dim], molecule['idx'], dim=0)
                    current_xh_mol[:,:self.x_dim] = current_xh_mol[:,:self.x_dim] - mean[molecule['idx']]
                    xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - mean[protein_pocket['idx']]

                if self.com_handling == 'both':
                    # old centering approach
                    current_xh_mol[:,:self.x_dim] = current_xh_mol[:,:self.x_dim] - scatter_mean(current_xh_mol[:,:self.x_dim], molecule['idx'], dim=0)[molecule['idx']]
                    xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - scatter_mean(xh_pro[:,:self.x_dim], protein_pocket['idx'], dim=0)[protein_pocket['idx']]
                else:
                    dumy_variable = 0


                # print(f"Centered predicted positions (Angstroms): {current_xh_mol[:,:self.x_dim][0]}")
                # print(f"Centered target positions (Angstroms):    {mol_norm_x[0]}")
                # error_mol = scatter_add(torch.sum((current_xh_mol[:,:self.x_dim] - mol_norm_x)**2, dim=-1), molecule['idx'], dim=0)
                # rmse_per_peptide = torch.sqrt(error_mol / molecule['size'])

                # batch_rmse = rmse_per_peptide.mean().item()
                
                # print(f"Batch RMSE: {batch_rmse:.4f}") 

        elif solver == "euler" or "rk4":

            ode_func = ODEWrapper(
                self, xh_pro, molecule, molecule_pos, protein_pocket
            )
            
            t_span = torch.tensor([1.0, 0.0], device=device)
            # t_span = torch.linspace(1.0, 0.0, 11, device=device)
            
            trajectory = odeint(
                ode_func, 
                current_xh_mol, 
                t_span, 
                method=solver, 
                options={'step_size': 1.0 / self.T} 
            )

            # print("\n--- ODE Integration Progress ---")
            # for idx, t_val in enumerate(t_span):
            #     step_xh = trajectory[idx]
            #     step_x = step_xh[:, :3]
                
            #     # Calculate RMSE for this specific intermediate step
            #     # Note: target (mol_norm_x) is already normalized to the same scale
            #     error_mol = scatter_add(torch.sum((step_x - mol_norm_x)**2, dim=-1), molecule['idx'], dim=0)
            #     rmse_per_peptide = torch.sqrt(error_mol / molecule['size'])
            #     batch_rmse = rmse_per_peptide.mean().item()
                
            #     print(f"Time t={t_val.item():.2f} | Avg Coord: {step_x[0]} | Batch RMSE: {batch_rmse:.4f}")
            # print("---------------------------------\n")
            
            current_xh_mol = trajectory[-1]
            c_s = ode_func.last_c_s

        x_mol_final = current_xh_mol[:,:self.x_dim] * self.norm_values[0]
        h_mol_final = current_xh_mol[:,self.x_dim:] * self.norm_values[0]
        x_pro_final = xh_pro[:,:self.x_dim] * self.norm_values[0]
        h_pro_final = xh_pro[:,self.x_dim:] * self.norm_values[0]

        if not self.features_fixed:
            h_mol_final = F.one_hot(torch.argmax(current_xh_mol[:, self.x_dim:], dim=1), self.num_atoms)
        else:
            h_mol_final = molecule['h'] 

        xh_mol_final = torch.cat([x_mol_final, h_mol_final], dim=1)
        xh_pro_final = torch.cat([x_pro_final, h_pro_final], dim=1)

        self.safe_pdbs(xh_mol_final, molecule, run_id, data_dir, time_step='F')

        return (xh_mol_final, xh_pro_final, c_s)
    
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

class ODEWrapper(nn.Module):
    def __init__(self, model, xh_pro, molecule, molecule_pos, protein_pocket):
        super().__init__()
        self.model = model
        self.xh_pro = xh_pro
        self.molecule_idx = molecule['idx']
        self.protein_idx = protein_pocket['idx']
        self.molecule_pos = molecule_pos
        self.num_graphs = molecule['size'].size(0)
        self.last_c_s = None

    def forward(self, t, xh_mol):
        t_vec = torch.full((self.num_graphs, 1), fill_value=t.item(), device=xh_mol.device)
        
        v_hat_mol, _, c_s = self.model.neural_net(
            xh_mol, self.xh_pro, t_vec, 
            self.molecule_idx, self.protein_idx, self.molecule_pos
        )
        self.last_c_s = c_s
        
        v_xh_final = torch.zeros_like(xh_mol)
        
        v_x = -v_hat_mol[:, :3] 

        v_x = v_x - scatter_mean(v_x, self.molecule_idx, dim=0)[self.molecule_idx]
        
        v_xh_final[:, :3] = v_x
        
        return v_xh_final
    

@torch.no_grad()
def sample_structure2(self, num_samples, molecule, protein_pocket, sampling_without_noise, data_dir, run_id):

    device = molecule['x'].device
    if self.position_encoding:
        molecule_pos = molecule['pos_in_seq']
    else:
        molecule_pos = None
    num_samples = len(molecule['size'])

    # print(f"True molecule positions:                                          {molecule['x'][0]}")

    # Record protein_pocket center of mass before
    protein_pocket_com_before = scatter_mean(protein_pocket['x'], protein_pocket['idx'], dim=0)

    # Normalisation
    # molecule['x'] = molecule['x'] / self.norm_values[0]
    molecule['h'] = molecule['h'] / self.norm_values[1]
    protein_pocket['x'] = protein_pocket['x'] / self.norm_values[0]
    protein_pocket['h'] = protein_pocket['h'] / self.norm_values[1]

    # start with random peptide position (target hidden)
    rand_eps_x = torch.randn((len(molecule['x']), self.x_dim), device=device) #* self.noise_scaling
    # print(f"Random noise added to molecule positions:                         {rand_eps_x[0]}")

    molecule_x = protein_pocket_com_before[molecule['idx']] + rand_eps_x
    # print(f"Initial molecule positions with noise added:                      {molecule_x[0]}")

    # could generate new peptides (not implemented currently)
    if self.features_fixed:
        molecule_h = molecule['h'].clone().detach()
    else:
        raise NotImplementedError

    # combine position and features
    xh_mol = torch.cat((molecule_x, molecule_h), dim=1)
    xh_pro = torch.cat((protein_pocket['x'], protein_pocket['h']), dim=1)

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

    # print(f"Noised molecule positions after centering:                        {xh_mol[:,:self.x_dim][0]}")

    max_T = self.T

    # Only for confidence testing
    if self.confidence_score == True:
        confidence = []

    z_t_mol = xh_mol.clone().detach()

    # print(f"Initial z_t_mol:                                                  {z_t_mol[0]}")

    solver = "simple"

    if solver == "simple":

        # Iterativly denoise stepwise for t = T,...,1; stepsize default is 1
        for s in reversed(range(0, max_T, self.sampling_stepsize)):
        # for s in range(0, max_T, self.sampling_stepsize):
            # print(f"s={s}")
            # time arrays
            # s_array = torch.full((num_samples, 1), fill_value=s, device=device)
            # t_array = s_array + self.sampling_stepsize
            # s_array_norm = s_array / self.T
            # t_array_norm = t_array / self.T
            # dt = t_array_norm[0] - s_array_norm[0]
            dt = 1.0 / self.T
            t_val = s / self.T
            t_vec = torch.full((num_samples, 1), fill_value=t_val, device=device)

            # print(f"{dt=}")
            # print(f"In loop z_t_mol:                                          {z_t_mol[:,:self.x_dim][0]}")

            # z_t_mol_old = z_t_mol.clone().detach()

            # x_t + self(x_t=x_t, t=t_start) * (t_end - t_start) / 2)
            # v_hat_mol, v_hat_pro, c_s = self.neural_net(z_t_mol_old, xh_pro, s_array_norm, molecule['idx'], protein_pocket['idx'], molecule_pos)
            # z_t_mol = z_t_mol_old + v_hat_mol * dt / 2

            # print(f"{z_t_mol[:,:self.x_dim][0]=}")

            # # x_t + (t_end - t_start) * self(t=t_start + (t_end - t_start) / 2, x_t=
            # t = s_array_norm + (t_array_norm - s_array_norm) / 2
            # v_hat_mol, v_hat_pro, c_s = self.neural_net(z_t_mol, xh_pro, t, molecule['idx'], protein_pocket['idx'], molecule_pos)
            # # z_t_mol = z_t_mol_old + (s_array_norm[0] - t_array_norm[0]) + v_hat_mol
            # z_t_mol = z_t_mol_old + dt * v_hat_mol


            v_hat_mol, v_hat_pro, c_s = self.neural_net(
                z_t_mol, xh_pro, t_vec, 
                molecule['idx'], protein_pocket['idx'], molecule_pos
            ) 

            # v_x = v_hat_mol[:, :self.x_dim]
            print(f"In loop v_hat_mol:                                              {v_hat_mol[0]}")

            z_t_mol = z_t_mol - v_hat_mol * dt

            print(f"In loop reconstructed z_t_mol:                            {z_t_mol[:,:self.x_dim][0]}")
            

        
            # Only for confidence testing
            if self.confidence_score == True:
                C_S = scatter_add(c_s, molecule['idx'], dim=0).squeeze(1) / molecule['size']
                confidence += [C_S]

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



            # x0_hat = z_t_mol[:, :3] + t_vec[molecule['idx']] * v_hat_mol[:, :3]

            # 4. Compute RMSE per peptide (Matching your Eval Script logic)
            print(f"Centered predicted positions (Angstroms): {z_t_mol[:,:self.x_dim][0]}")
            print(f"Centered target positions (Angstroms):    {molecule['x'][0]}")
            error_mol = scatter_add(torch.sum((z_t_mol[:,:self.x_dim] - molecule['x'])**2, dim=-1), molecule['idx'], dim=0)
            rmse_per_peptide = torch.sqrt(error_mol / molecule['size'])

            # 5. Average across the batch
            batch_rmse = rmse_per_peptide.mean().item()
            
            print(f"Batch RMSE: {batch_rmse:.4f}") # Helpful for debugging

    elif solver == "euler":
    
        # Define the ODE function
        ode_func = VelocityODEWrapper(
            model=self,
            xh_pro=xh_pro,
            molecule=molecule,
            molecule_pos=molecule_pos,
            protein_pocket=protein_pocket
        )

        # Integration: From t=1 (Noise) to t=0 (Data)
        # Flow Matching uses t=1 as the source (noise) and t=0 as the target
        t_span = torch.tensor([1.0, 0.0], device=device)

        trajectory = odeint(
            ode_func, 
            xh_mol[:, :self.x_dim], 
            t_span, 
            method='euler', 
            options={'step_size': 1.0 / self.T}
        )
        
        # Extract the final result at t=0.0
        print(f"{trajectory.shape=}")
        xh_mol_final_x = trajectory[-1]
        c_s = ode_func.last_c_s
        print(f"{xh_mol_final_x[0]=}")
        
        # Re-attach the features to get your final z_t_mol
        z_t_mol = torch.cat([xh_mol_final_x, molecule['h']], dim=1)

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

    print(f"Noised molecule positions after final centering:                 {xh_mol_final[:,:self.x_dim][0]}")

    if self.com_handling == 'both':
            dumy_variable = 0
    elif self.com_handling == 'no_COM':
            dumy_variable = 0
    else:
        # project both pocket and peptide to 0 COM again (only mol mean changes)
        mean = scatter_mean(xh_mol_final[:,:self.x_dim], molecule['idx'], dim=0)
        xh_mol_final[:,:self.x_dim] = xh_mol_final[:,:self.x_dim] - mean[molecule['idx']]
        xh_pro_final[:,:self.x_dim] = xh_pro_final[:,:self.x_dim] - mean[protein_pocket['idx']]

    print(f"Noised molecule positions after final centering (second time):   {xh_mol_final[:,:self.x_dim][0]}")

    # Unnormalisation
    x_mol_final = xh_mol_final[:,:self.x_dim] * self.norm_values[0]
    h_mol_final = xh_mol_final[:,self.x_dim:] * self.norm_values[0]
    x_pro_final = xh_pro_final[:,:self.x_dim] * self.norm_values[0]
    h_pro_final = xh_pro_final[:,self.x_dim:] * self.norm_values[0]

    # print(f"Unnormalized molecule positions:                                 {x_mol_final[0]=}")
    # Round h to one_hot encoding
    h_mol_final = F.one_hot(torch.argmax(h_mol_final, dim=1), self.num_atoms)

    # Recombine x and h
    xh_mol_final = torch.cat([x_mol_final, h_mol_final], dim=1)
    xh_pro_final = torch.cat([x_pro_final, h_pro_final], dim=1)

    # print(f"Recombined molecule positions and features:                      {xh_mol_final[:,:self.x_dim][0]}")

    # Correct for center of mass difference
    protein_pocket_com_after = scatter_mean(x_pro_final, protein_pocket['idx'], dim=0)

    # Testing if we only learn the form
    # Moving mol targets COM to 0
    # mol_target = molecule['x']  - scatter_mean(molecule['x'], molecule['idx'], dim=0)[molecule['idx']]

    # print(f"Moved molecule targets COM to 0:                                 {mol_target[:,:self.x_dim][0]}")

    xh_mol_final[:,:self.x_dim] += (protein_pocket_com_before - protein_pocket_com_after)[molecule['idx']]
    xh_pro_final[:,:self.x_dim] += (protein_pocket_com_before - protein_pocket_com_after)[protein_pocket['idx']]

    print(f"Adjusted molecule positions for COM difference:                  {xh_mol_final[0]}")
    print(f"True molecule positions:                                         {molecule['x'][0]}")
    print(f"True protein features:                                           {molecule['h'][0]}")
    # Moving mol targets COM to original COM
    # mol_target += (protein_pocket_com_before - protein_pocket_com_after)[molecule['idx']]
    # print(f"Moved molecule targets COM to original COM:                      {mol_target[:,:self.x_dim][0]}")
    sampled_structures = (xh_mol_final, xh_pro_final, c_s)

    self.safe_pdbs(xh_mol_final, molecule, run_id, data_dir, time_step='F')

    # Only for confidence testing
    if self.confidence_score == True:
        print(C_S)  
    
    return sampled_structures

class VelocityODEWrapper(nn.Module):
    def __init__(self, model, xh_pro, molecule, molecule_pos, protein_pocket):
        super().__init__()
        self.model = model
        self.xh_pro = xh_pro
        self.molecule = molecule
        self.molecule_pos = molecule_pos
        self.protein_pocket = protein_pocket
        self.last_c_s = None

    def forward(self, t, x):
        # x is the current position integrated by the solver [N_nodes, 3]
        # Reconstruct z_t_mol using the solver's x and the original fixed features h
        print(f"{t=}")
        print(f"{x[0]=}")
        z_t_mol = torch.cat([x, self.molecule['h']], dim=1)
        print(f"{z_t_mol[0]=}")
        
        # Create the time tensor t for the neural network call
        batch_size = self.molecule['size'].size(0)
        t_vec = torch.ones((batch_size, 1), device=x.device) * t
        print(f"{batch_size=}")
        print(f"{t_vec[0]=}")

        print(f"{z_t_mol[0]=}")
        print(f"{self.xh_pro[0]=}")
        print(f"{self.molecule['idx'][0]=}")
        print(f"{self.molecule_pos[0]=}")
        
        # Call neural_net with your variable names
        v_hat_mol, v_hat_pro, c_s = self.model.neural_net(z_t_mol, self.xh_pro, t_vec, self.molecule['idx'], self.protein_pocket['idx'], self.molecule_pos)
        print(f"{v_hat_mol[0]=}")

        self.last_c_s = c_s

        # Extract coordinate velocity
        v_x_mol = v_hat_mol[:, :self.model.x_dim]
        
        # Project velocity to center-of-mass = 0 subspace to maintain translational invariance
        # This matches the logic in your training loop [cite: 142, 146]
        # mean_v = scatter_mean(v_x_mol, self.molecule['idx'], dim=0)
        # v_x_mol = v_x_mol - mean_v[self.molecule['idx']]
        
        return v_x_mol

