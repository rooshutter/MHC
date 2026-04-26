import torch
import numpy as np
import math
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch.nn as nn
import torch.nn.functional as F
from openfold.utils.rigid_utils import Rotation
from torch_scatter import scatter_add, scatter_mean
# from zmq import device

from model.noise_schedule import Noise_Schedule
from utils import create_new_pdb_hdf5, create_new_pdb_hdf5_100k

from torchdiffeq import odeint_adjoint as odeint

from tools.rigid import Rigid 

from openfold.model.primitives import Linear, LayerNorm
from openfold.model.structure_module import AngleResnet, StructureModuleTransition, BackboneUpdate
from openfold.utils.tensor_utils import dict_multimap
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from openfold.np.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
from rvf.manifolds.sphere import SphereManifold


class Flow_Matching_Model_all_atom(nn.Module):
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
        self.eps = 1e-7

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

        c_s = 32
        c_resnet = 64
        no_resnet_blocks = 2
        no_angles = 7
        epsilon = 1e-12

        # # for predicting torsion angles
        # self.angle_resnet = AngleResnet(
        #     c_s,
        #     c_resnet,
        #     no_resnet_blocks,
        #     no_angles,
        #     epsilon,
        # )

        # # init the angle bias to be nonzero
        # with torch.no_grad():
        #     self.angle_resnet.linear_out.bias.fill_(epsilon)


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
        z_t_mol, z_t_pro, v_x_mol, v_pro, t = self.compute_flow_match(z_data)
        

        # z_t_mol = z_t_mol.reshape(-1, z_t_mol.shape[-1])
        # z_t_pro = z_t_pro.reshape(-1, z_t_pro.shape[-1])
        # v_x_mol = v_x_mol.reshape(-1, v_x_mol.shape[-1])
        # v_pro = v_pro.reshape(-1, v_pro.shape[-1])
        # t = t.squeeze(-1)

        # z_t_mol = z_t_mol[:, 4:]
        # z_t_pro = z_t_pro[:, 4:]
        # v_x_mol = v_x_mol[:, :3]
        # v_pro = v_pro[:, 4:]


        if self.noise_scaling > 0:
            z_t_mol = z_t_mol + torch.randn_like(z_t_mol) * self.noise_scaling
            z_t_mol_q = z_t_mol[:, :, :4] / (z_t_mol[:, :, :4].norm(dim=-1, keepdim=True) + self.eps)
            z_t_mol = torch.cat((z_t_mol_q, z_t_mol[:,:,4:]), dim=-1)

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
    
    def compute_flow_match(self, z_data, t_is_0 = False):
        molecule, protein_pocket = z_data
        batch_size = molecule['size'].size(0)
        device = molecule['x'].device
                
        size_mol = molecule['size'][0]
        size_pro = protein_pocket['size'][0]

        if molecule['h'].shape[0] != batch_size:
            molecule['h'] = molecule['h'].view(batch_size, size_mol, *molecule['h'].shape[1:])
        
        if protein_pocket['h'].shape[0] != batch_size:
            protein_pocket['h'] = protein_pocket['h'].view(batch_size, size_pro, *protein_pocket['h'].shape[1:])
        if molecule['torsion_angles_sin_cos'].shape[0] != batch_size:
            molecule['torsion_angles_sin_cos'] = molecule['torsion_angles_sin_cos'].view(batch_size, size_mol, *molecule['torsion_angles_sin_cos'].shape[1:])
        if molecule['backbone_rigid_tensor'].shape[0] != batch_size:
            molecule['backbone_rigid_tensor'] = molecule['backbone_rigid_tensor'].view(batch_size, size_mol, *molecule['backbone_rigid_tensor'].shape[1:])
        if protein_pocket['backbone_rigid_tensor'].shape[0] != batch_size:
            protein_pocket['backbone_rigid_tensor'] = protein_pocket['backbone_rigid_tensor'].view(batch_size, size_pro, *protein_pocket['backbone_rigid_tensor'].shape[1:])
        
        
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
        xh_mol = torch.cat((T_peptide, molecule['h']), dim=-1)

        protein_backbone_rigid_tensor = protein_pocket['backbone_rigid_tensor']
        T_protein = Rigid.from_tensor_4x4(protein_backbone_rigid_tensor)
        T_protein = T_protein.to_tensor_7()
        xh_pro = torch.cat((T_protein, protein_pocket['h']), dim=-1)
        

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

        # T_peptide_z = Rigid.identity(
        #     molecule['h'].shape[:-1],
        #     molecule['h'].dtype,
        #     device,
        #     self.training,
        #     fmt="quat",
        # )
        z_trans = torch.randn((*molecule['h'].shape[:-1], 3), device=device)
        z_quat = torch.randn((*molecule['h'].shape[:-1], 4), device=device)
        z_quat = torch.nn.functional.normalize(z_quat, dim=-1)
        T_peptide_z = torch.cat((z_quat, z_trans), dim=-1)
        T_peptide_z = Rigid.from_tensor_7(T_peptide_z)

        
        z_x_pro = torch.zeros(size=(len(xh_pro), xh_pro.shape[1], T_peptide.shape[-1]), device=device)

        # print(f"{self.com_handling=}") peptide
        # if self.com_handling == 'both':
        #     # alternative centering approach
        #     z_x_mol = z_x_mol - scatter_mean(z_x_mol, molecule['idx'], dim=0)[molecule['idx']]
        #     z_x_pro = torch.zeros(size=(len(xh_pro), self.x_dim), device=device)
        # else:
        #     dumy_variable = 0

        # print("features_fixed:", self.features_fixed) true
        if self.features_fixed:
            z_h_mol = torch.zeros(size=(len(xh_mol), xh_mol.shape[1], self.num_atoms), device=device)
            # z_h_mol = torch.zeros(size=(len(xh_mol), self.num_atoms), device=device)
            z_h_pro = torch.zeros(size=(len(xh_pro), xh_pro.shape[1], self.num_residues), device=device)
        else:
            # for h we need standard normal noise (this would be sampling new peptides)
            z_h_mol = torch.randn(size=(len(xh_mol), xh_mol.shape[1], self.num_atoms), device=device)
            z_h_pro = torch.randn(size=(len(xh_pro), xh_pro.shape[1], self.num_residues), device=device)


        z_pro = torch.cat((z_x_pro, z_h_pro), dim=-1)

        # z_t_mol_x = t[molecule['idx']] * molecule['x'] + (1 - t[molecule['idx']]) * z_x_mol
        # z_t_mol_x = (1 - t[molecule['idx']]) * molecule['x'] + (t[molecule['idx']]) * z_x_mol
    
        T_peptide_z = T_peptide_z.to_tensor_7()

        # from rvf loss_sphere.py VariationalLossSphere -> loss_intrinsic
        x0 = T_peptide_z[:,:,:4]
        x1 = T_peptide[:,:,:4]
        v_0 = SphereManifold().log_map(x0, x1)
        x_t = SphereManifold().exp_map(x0, (1 - t) * v_0)

        T_peptide_t = (1 - t) * T_peptide + t * T_peptide_z
        T_peptide_t[:,:,:4] = x_t

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
        v_x_mol = xh_mol[:,:,:T_peptide_z.shape[-1]] - T_peptide_z 
        
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
    
    def predict_pos(self, molecule, T_peptide):
        
        T_peptide = T_peptide.view(molecule['size'].size(0), -1, *T_peptide.shape[1:])
        T_peptide = Rigid.from_tensor_7(T_peptide)
        # print(f"{T_peptide.shape=}")
        #normalize quat:
        # quat = T_peptide[:,:,:4] / (T_peptide[:,:,:4].norm(dim=-1, keepdim=True) + self.eps)
        # T_peptide = torch.cat((quat, T_peptide[:,:,4:]), dim=-1)
        # print(f"{T_peptide[0]=}")
        # TODO: normalize?
        # peptide_backbone_rigid_tensor = molecule['backbone_rigid_tensor']
        # T_peptide = Rigid.from_tensor_4x4(peptide_backbone_rigid_tensor)
        
        backb_to_global = Rigid(
            Rotation(
                rot_mats=T_peptide.get_rots().get_rot_mats(),
                quats=None
            ),
            T_peptide.get_trans(),
        )
        
        # apply the scale factor, to get the final backbone frames
        # trans_scale_factor = 10 
        # backb_to_global = backb_to_global.scale_translation(trans_scale_factor)

        #TODO implement angles
        # s_peptide = molecule['h']
        # s_peptide_initial = torch.zeros_like(s_peptide)
        # unnormalized_angles, angles = self.angle_resnet(s_peptide, s_peptide_initial)
        # print(f"{angles.shape=}")
        # print(f"{angles[0]=}")

        # TODO: later change to predicted angles
        angles = molecule['torsion_angles_sin_cos']
        peptide_aatype = molecule['aatype']
        # angles = angles.view(molecule['size'].size(0), -1, *angles.shape[1:])
        peptide_aatype = peptide_aatype.view(molecule['size'].size(0), -1, *peptide_aatype.shape[1:])

        all_frames_to_global = self.torsion_angles_to_frames(
            backb_to_global,
            angles,
            peptide_aatype,
        )

        pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global,
            peptide_aatype,
        )

        return pred_xyz
    
    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    restype_atom14_to_rigid_group,
                    device=device,
                    dtype=torch.long,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, T, alpha, aatype):

        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)

        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(T, alpha, aatype, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
        self, T, aatype  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        T_rots = T.get_rots()
        self._init_residue_constants(T_rots.dtype, T_rots.device)

        return frames_and_literature_positions_to_atom14_pos(
            T,
            aatype,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )

    def train_loss(
            self, molecule, z_t_mol, v_x_mol, 
            v_hat_mol, protein_pocket, 
            z_t_pro, v_pro, v_hat_pro, t, c_s
    ):
        v_x_mol = v_x_mol.reshape(-1, v_x_mol.shape[-1])
        v_hat_mol_quat = v_hat_mol[:,:4]
        v_hat_mol_pos = v_hat_mol[:,4:7]

        peptide_backbone_rigid_tensor = molecule['backbone_rigid_tensor']
        T_peptide = Rigid.from_tensor_4x4(peptide_backbone_rigid_tensor)
        T_peptide = T_peptide.to_tensor_7()
        x1 = T_peptide[:,:,:4]
        #reshape to 288, 4
        x1 = x1.reshape(-1, x1.shape[-1])
        # print(f"{v_hat_mol_quat[0]=}") 
        # print(f"{x1[0]=}")
        distance = SphereManifold().distance(v_hat_mol_quat, x1)
        error_quat = error_quat = distance.pow(2).mean()
        
        x_mol = molecule['x']
        x_hat_mol = self.predict_pos(molecule, v_hat_mol[:,:7])
        x_hat_mol = x_hat_mol.reshape(-1, x_hat_mol.shape[-2], x_hat_mol.shape[-1])
        error_x = scatter_add(torch.sum((x_mol - x_hat_mol)**2, dim=(1,2)), molecule['idx'], dim=0)
        # print(f"{x_hat_mol.shape=}")
        # print(f"{x_mol.shape=}")
        # print(f"{x_mol[0]=}")
        # print(f"{x_hat_mol[0]=}")
        #compute rmse and average over peptides in batch:
        rmse = torch.sqrt(error_x / molecule['size'])
        # print(f"{rmse[0]=}")    

        # compute the sum squared error loss per graph # TODO: modified to not take the h_dims
        x1_pos = T_peptide[:, :, 4:7]
        x1_pos = x1_pos.reshape(-1, x1_pos.shape[-1])
        error_trans = scatter_add(torch.sum((x1_pos - v_hat_mol_pos)**2, dim=-1), molecule['idx'], dim=0)
        # print(f"{x1_pos[0]=}")
        # print(f"{v_hat_mol_pos[0]=}")

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
        loss_x_mol_t0 = - loss_x_mol_t0 * t_0_mask
        loss_x_protein_t0 = - loss_x_protein_t0 * t_0_mask
        loss_h_t0 = - loss_h_t0 * t_0_mask
        error_x = error_x * t_not_0_mask
        error_pro = error_pro * t_not_0_mask

        # Normalize loss_t by graph size
        error_x = error_x / ((self.x_dim) * molecule['size'])
        error_pro = error_pro / ((self.x_dim + self.num_residues * protein_pocket['size']))
        loss_t = 0.5 * (error_x + error_pro) # * SNR_t

        # Normalize loss_0 by graph size
        loss_x_mol_t0 = loss_x_mol_t0 / (self.x_dim * molecule['size'])
        loss_x_protein_t0 = loss_x_protein_t0 / (self.x_dim * protein_pocket['size'])
        loss_0 = loss_x_mol_t0 + loss_x_protein_t0 + loss_h_t0

        loss = loss_t + loss_0 + kl_prior + error_quat + error_trans
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
            'error_x': error_x.mean(0),
            'loss_x_mol_t0': loss_x_mol_t0.mean(0),
            'kl_prior': kl_prior.mean(0),
            'confidence': c_s_peptide.mean(0),
            'loss_with_conf': loss_with_conf.mean(0),
            'error_trans': error_trans.mean(0),
            'error_quat': error_quat.mean(0),
        }

        if self.confidence_score == True:
            return loss_with_conf, info

        return loss, info
    
    def validation_loss(
            self, z_data, molecule, z_t_mol, v_x_mol, v_hat_mol,
            protein_pocket, z_t_pro, v_pro, v_hat_pro, 
            t,
    ):
        v_x_mol = v_x_mol.reshape(-1, v_x_mol.shape[-1])

        v_hat_mol_quat = v_hat_mol[:,:4]
        v_hat_mol_pos = v_hat_mol[:,4:7]

        peptide_backbone_rigid_tensor = molecule['backbone_rigid_tensor']
        T_peptide = Rigid.from_tensor_4x4(peptide_backbone_rigid_tensor)
        T_peptide = T_peptide.to_tensor_7()
        x1 = T_peptide[:,:,:4]
        x1 = x1.reshape(-1, x1.shape[-1])
        distance = SphereManifold().distance(v_hat_mol_quat, x1)
        error_quat = distance.pow(2).mean()

        
        ### Additional evaluation (VLB) variables
        if self.position_encoding:
            molecule_pos = molecule['pos_in_seq']
        else:
            molecule_pos = None

        x_mol = molecule['x']
        x_hat_mol = self.predict_pos(molecule, v_hat_mol[:,:7])
        x_hat_mol = x_hat_mol.reshape(-1, x_hat_mol.shape[-2], x_hat_mol.shape[-1])
        #reshape from ([288, 14, 3]) to ([32, 9, 14, 3])
        # x_mol = x_mol.reshape(molecule['size'].size(0), -1, x_mol.shape[-2], x_mol.shape[-1])
        sum = torch.sum((x_mol - x_hat_mol)**2, dim=(1,2))
        error_x = scatter_add(sum, molecule['idx'], dim=0)

        # compute the sum squared error loss per graph
        x1_pos = T_peptide[:, :, 4:7]
        x1_pos = x1_pos.reshape(-1, x1_pos.shape[-1])
        error_trans = scatter_add(torch.sum((x1_pos - v_hat_mol_pos)**2, dim=-1), molecule['idx'], dim=0)
        
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


        # z_0_mol = z_0_mol.reshape(-1, z_0_mol.shape[-1])
        # z_0_pro = z_0_pro.reshape(-1, z_0_pro.shape[-1])
        # v_target_0_mol = v_target_0_mol.reshape(-1, v_target_0_mol.shape[-1])
        # v_target_0_pro = v_target_0_pro.reshape(-1, v_target_0_pro.shape[-1])
        # t_0 = t_0.squeeze(-1)

        # z_0_mol = z_0_mol[:, :23]
        # z_0_pro = z_0_pro[:, :23]
        # v_target_0_mol = v_target_0_mol[:, :3]
        # v_target_0_pro = v_target_0_pro[:, :3]
        # z_0_mol = torch.cat((z_0_mol[:,:3], z_0_mol[:,7:]), dim=-1)
        # z_0_pro = torch.cat((z_0_pro[:,:3], z_0_pro[:,7:]), dim=-1)
        # v_target_0_mol = v_target_0_mol[:,:3]
        # v_target_0_pro = torch.cat((v_target_0_pro[:,:3], v_target_0_pro[:,7:]), dim=-1)
        
        v_hat_0_mol, v_hat_0_pro, _ = self.neural_net(z_0_mol, z_0_pro, t_0, molecule['idx'], protein_pocket['idx'], molecule_pos)

        # v_hat_0_mol_pos = v_hat_0_mol[:,:3]
        # v_hat_0_mol_quat = v_hat_0_mol[:,23:]

        loss_x_mol_t0, loss_x_protein_t0, loss_h_t0 = self.loss_t0(
            molecule, z_0_mol, v_target_0_mol, v_hat_0_mol,
            protein_pocket, z_0_pro, v_target_0_pro, v_hat_0_pro, t_0
        )

        loss_x_mol_t0 = - loss_x_mol_t0
        loss_x_protein_t0 = - loss_x_protein_t0
        loss_h_t0 = - loss_h_t0


        # loss_t = - self.T * 0.5 * SNR_weight * (error_mol + error_pro)
        loss_t = self.T * 0.5 * (error_x + error_pro)
        loss_0 = loss_x_mol_t0 + loss_x_protein_t0 + loss_h_t0
        loss_0 = loss_0 + neg_log_const

        # Two added loss terms for vlb
        loss = loss_t + loss_0 + kl_prior - delta_log_px - log_pN + error_quat + error_trans
        # loss = loss_t + kl_prior

        info = {
            'loss_t': loss_t.mean(0),
            'loss_0': loss_0.mean(0),
            'error_x': error_x.mean(0),
            'loss_x_mol_t0': loss_x_mol_t0.mean(0),
            'kl_prior': kl_prior.mean(0),
            'neg_log_const': neg_log_const.mean(0),
            'SNR_weight': SNR_weight.mean(0),
            'error_trans': error_trans.mean(0),
            'error_quat': error_quat.mean(0),
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

        v_target_mol = v_target_mol.reshape(-1, v_target_mol.shape[-1])
        v_hat_mol_x = v_hat_mol[:,4:7]
        v_target_mol_x = v_target_mol[:,4:7]
        loss_x_mol_t0 = - 0.5 * scatter_add(torch.sum((v_target_mol_x - v_hat_mol_x)**2, dim=-1), molecule['idx'], dim=0)

        loss_x_protein_t0 = torch.zeros(protein_pocket['size'].size(0), device=molecule['x'].device)

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

        peptide_backbone_rigid_tensor = molecule['backbone_rigid_tensor']
        T_peptide = Rigid.from_tensor_4x4(peptide_backbone_rigid_tensor)
        T_peptide = T_peptide.to_tensor_7()
        T_peptide = T_peptide.reshape(-1, T_peptide.shape[-1])
        T_peptide = T_peptide[:, 4:7]


        # molecule['x'] = molecule['x'] - scatter_mean(molecule['x'], molecule['idx'], dim=0)[molecule['idx']]
        # T_peptide = T_peptide - scatter_mean(T_peptide[:,:,:self.x_dim], molecule['idx'], dim=0)[molecule['idx']]

        # z_t_mol_x = alpha_t[molecule['idx']] * xh_mol[:, :self.x_dim] + sigma_t[molecule['idx']] * eps_x_mol
        # z_t_mol_x = (1 - t[molecule['idx']]) * molecule['x'] + (t[molecule['idx']]) * z_x_mol


        T_normalized = torch.ones((len(molecule['size']), 1), device=device)
        # alpha_T = self.noise_schedule(T_normalized, 'alpha')
        # sigma_T = self.noise_schedule(T_normalized, 'sigma')
        # sigma_T_value = sigma_T[0,0].item()
        alpha_T = 0.0  # At t=1, (1-t) is 0
        sigma_T_val = 1.0

        # mu_x_mol = molecule['x'] * alpha_T[molecule['idx']] # [:,3]
        # mu_x_mol = molecule['x'] * alpha_T # [:,3]
        mu_x_mol = T_peptide * alpha_T # [:,3]
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
        
        size_mol = molecule['size'][0]
        size_pro = protein_pocket['size'][0]

        if molecule['h'].shape[0] != num_graphs:
            molecule['h'] = molecule['h'].view(num_graphs, size_mol, *molecule['h'].shape[1:])
        if protein_pocket['h'].shape[0] != num_graphs:
            protein_pocket['h'] = protein_pocket['h'].view(num_graphs, size_pro, *protein_pocket['h'].shape[1:])
        if molecule['torsion_angles_sin_cos'].shape[0] != num_graphs:
            molecule['torsion_angles_sin_cos'] = molecule['torsion_angles_sin_cos'].view(num_graphs, size_mol, *molecule['torsion_angles_sin_cos'].shape[1:])
        if molecule['backbone_rigid_tensor'].shape[0] != num_graphs:
            molecule['backbone_rigid_tensor'] = molecule['backbone_rigid_tensor'].view(num_graphs, size_mol, *molecule['backbone_rigid_tensor'].shape[1:])
        if protein_pocket['backbone_rigid_tensor'].shape[0] != num_graphs:
            protein_pocket['backbone_rigid_tensor'] = protein_pocket['backbone_rigid_tensor'].view(num_graphs, size_pro, *protein_pocket['backbone_rigid_tensor'].shape[1:])

        # z_x_mol = torch.randn(size=(len(molecule['x']), self.x_dim), device=device) #* self.noise_scaling
        # T_peptide_z = Rigid.identity(
        #     molecule['h'].shape[:-1],
        #     molecule['h'].dtype,
        #     device,
        #     self.training,
        #     fmt="quat",
        # )
        z_trans = torch.randn((*molecule['h'].shape[:-1], 3), device=device)
        z_quat = torch.randn((*molecule['h'].shape[:-1], 4), device=device)
        z_quat = torch.nn.functional.normalize(z_quat, dim=-1)
        T_peptide_z = torch.cat((z_quat, z_trans), dim=-1)
        T_peptide_z = Rigid.from_tensor_7(T_peptide_z)

        T_peptide_z = T_peptide_z.to_tensor_7()

        # mol_norm_x = molecule['x'] / self.norm_values[0]
        protein_pocket['x'] = protein_pocket['x'] / self.norm_values[0]
        protein_pocket['h'] = protein_pocket['h'] / self.norm_values[1]
        
        protein_backbone_rigid_tensor = protein_pocket['backbone_rigid_tensor']
        T_protein = Rigid.from_tensor_4x4(protein_backbone_rigid_tensor)
        T_protein = T_protein.to_tensor_7()
        xh_pro = torch.cat((T_protein, protein_pocket['h']), dim=-1)
        # xh_pro = torch.cat((protein_pocket['x'], protein_pocket['h']), dim=1)

        # if self.com_handling == 'both':
        #     # old centering approach
        #     z_x_mol = z_x_mol - scatter_mean(z_x_mol, molecule['idx'], dim=0)[molecule['idx']]
        #     xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - scatter_mean(xh_pro[:,:self.x_dim], protein_pocket['idx'], dim=0)[protein_pocket['idx']]
        # elif self.com_handling == 'no_COM':
        #         dumy_variable = 0
        # else:
        #     # data is translated to 0, COM noise added and again translated to 0
        #     mean = scatter_mean(z_x_mol, molecule['idx'], dim=0)
        #     z_x_mol = z_x_mol - mean[molecule['idx']]
        #     xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - mean[protein_pocket['idx']]

        if self.features_fixed:
            z_h_mol = (molecule['h'] / self.norm_values[1]).clone().detach()
        else:
            raise NotImplementedError

        current_xh_mol = torch.cat((T_peptide_z, z_h_mol), dim=-1)

        current_xh_mol = current_xh_mol.reshape(-1, current_xh_mol.shape[-1])
        xh_pro = xh_pro.reshape(-1, xh_pro.shape[-1])

        # current_xh_mol = torch.cat([current_xh_mol[:, :3], current_xh_mol[:, 7:]], dim=1)
        # xh_pro = torch.cat([xh_pro[:, :3], xh_pro[:, 7:]], dim=1)
        # current_xh_mol = current_xh_mol[:, 4:]
        # xh_pro = xh_pro[:, 4:]

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
                # if self.com_handling == 'both':
                #     dumy_variable = 0
                # elif self.com_handling == 'no_COM':
                #     dumy_variable = 0
                # else:
                #     # project both pocket and peptide to 0 COM again (only mol mean changes)
                #     mean = scatter_mean(current_xh_mol[:,:self.x_dim], molecule['idx'], dim=0)
                #     current_xh_mol[:,:self.x_dim] = current_xh_mol[:,:self.x_dim] - mean[molecule['idx']]
                #     xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - mean[protein_pocket['idx']]

                # if self.com_handling == 'both':
                #     # old centering approach
                #     current_xh_mol[:,:self.x_dim] = current_xh_mol[:,:self.x_dim] - scatter_mean(current_xh_mol[:,:self.x_dim], molecule['idx'], dim=0)[molecule['idx']]
                #     xh_pro[:,:self.x_dim] = xh_pro[:,:self.x_dim] - scatter_mean(xh_pro[:,:self.x_dim], protein_pocket['idx'], dim=0)[protein_pocket['idx']]
                # else:
                #     dumy_variable = 0


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
            #     step_x = step_xh[:, :7]
            #     print(f"Step x: {step_x[0]}")
                
            #     # Calculate RMSE for this specific intermediate step
            #     # Note: target (mol_norm_x) is already normalized to the same scale
            #     # error_mol = scatter_add(torch.sum((step_x - mol_norm_x)**2, dim=-1), molecule['idx'], dim=0)
            #     # rmse_per_peptide = torch.sqrt(error_mol / molecule['size'])
            #     # batch_rmse = rmse_per_peptide.mean().item()
                
            #     # print(f"Time t={t_val.item():.2f} | Avg Coord: {step_x[0]} | Batch RMSE: {batch_rmse:.4f}")
            # print("---------------------------------\n")
            
            current_xh_mol = trajectory[-1]
            c_s = ode_func.last_c_s

        # x_mol_final = current_xh_mol[:,:self.x_dim] * self.norm_values[0]
        T_peptide_hat = current_xh_mol[:, :7]
        h_mol_final = current_xh_mol[:,7:] * self.norm_values[0]
        x_pro_final = protein_pocket['x']
        h_pro_final = xh_pro[:,7:] * self.norm_values[0]
        print(f"{T_peptide_hat[0]=}")
        peptide_backbone_rigid_tensor = molecule['backbone_rigid_tensor']
        T_peptide = Rigid.from_tensor_4x4(peptide_backbone_rigid_tensor)
        T_peptide = T_peptide.to_tensor_7()
        print(f"{T_peptide[0][0]=}")

        if not self.features_fixed:
            h_mol_final = F.one_hot(torch.argmax(current_xh_mol[:, self.x_dim:], dim=1), self.num_atoms)
        else:
            h_mol_final = molecule['h'] 

        h_pro_final = h_pro_final.unsqueeze(1).expand(-1, x_pro_final.shape[1], -1)
        xh_pro_final = torch.cat([x_pro_final, h_pro_final], dim=-1)

        x_hat_mol = self.predict_pos(molecule, T_peptide_hat)
        h_mol_final = h_mol_final.unsqueeze(2).expand(-1, -1, x_hat_mol.shape[2], -1)
        x_hat_mol = x_hat_mol.reshape(-1, x_hat_mol.shape[-2], x_hat_mol.shape[-1])
        print(f'{x_hat_mol[0][0]=}')
        x_hat_mol = x_hat_mol.reshape(-1, x_hat_mol.shape[-1])
        print(f"{molecule['x'][0][0]=}")
        
        h_mol_final = h_mol_final.reshape(-1, h_mol_final.shape[-1])
        xh_mol_final = torch.cat([x_hat_mol, h_mol_final], dim=-1)

        self.safe_pdbs(xh_mol_final, molecule, run_id, data_dir, time_step='F')

        return (xh_mol_final, xh_pro_final, c_s)
    
    def safe_pdbs(self, pos, molecule, run_id, data_dir, time_step):

        for i in range(len(molecule['size'])):
            # (1) extract the peptide position
            pos = pos[:,:3]
            idx_mol = molecule['idx'].repeat_interleave(molecule['x'].shape[1])
            peptide_pos = pos[idx_mol == i]
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
        t_vec = torch.full((self.num_graphs, 1, 1), fill_value=t.item(), device=xh_mol.device)

        v_hat_mol, _, c_s = self.model.neural_net(
            xh_mol, self.xh_pro, t_vec, 
            self.molecule_idx, self.protein_idx, self.molecule_pos
        )
        self.last_c_s = c_s
        
        v_xh_final = torch.zeros_like(xh_mol)

        v_x = -v_hat_mol[:, :7] #TODO check 0-1 vs 1-0

        scale = torch.clip(torch.ones_like(t) / t, 0, 10)
        v_x_quat = SphereManifold().log_map(xh_mol[:, :4], v_x[:, :4]) * scale
        v_x = torch.cat([v_x_quat, v_x[:, 4:]], dim=-1)

        # v_x = v_x - scatter_mean(v_x, self.molecule_idx, dim=0)[self.molecule_idx]
        
        v_xh_final[:, :7] = v_x
        
        return v_xh_final
    