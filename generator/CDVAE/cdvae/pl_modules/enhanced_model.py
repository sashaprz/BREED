from typing import Any, Dict
import math

import hydra
import numpy as np
import omegaconf
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm

from cdvae.common.utils import PROJECT_ROOT
from cdvae.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume,
    frac_to_cart_coords, min_distance_sqr_pbc)
from cdvae.pl_modules.embeddings import MAX_ATOMIC_NUM
from cdvae.pl_modules.embeddings import KHOT_EMBEDDINGS
from cdvae.pl_modules.model import BaseModule, build_mlp


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for improved latent representations."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Self-attention
        residual = x
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.w_o(context)
        
        # Residual connection and layer norm
        return self.layer_norm(residual + output)


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward layers."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention
        x = self.attention(x)
        
        # Feed-forward with residual connection
        residual = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        return self.layer_norm(residual + x)


class ImprovedAtomCountPredictor(nn.Module):
    """Enhanced atom count predictor with attention mechanism and dynamic range."""
    
    def __init__(self, latent_dim, hidden_dim, max_atoms, num_layers=3, num_heads=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_atoms = max_atoms
        # Use a much larger range to handle any realistic crystal structure
        self.extended_max_atoms = max(max_atoms * 4, 500)  # Support up to 500 atoms
        
        # Transformer layers for better representation
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(latent_dim, num_heads, hidden_dim * 2)
            for _ in range(num_layers)
        ])
        
        # Multi-scale feature extraction
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=k, padding=k//2)
            for k in [1, 3, 5]
        ])
        
        # Atom count specific layers with extended range
        self.atom_count_layers = nn.Sequential(
            nn.Linear(latent_dim + len(self.multi_scale_conv) * hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.extended_max_atoms + 1)
        )
        
        # Auxiliary regression head for continuous atom count
        self.continuous_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, z):
        batch_size = z.size(0)
        
        # Add sequence dimension for transformer
        z_seq = z.unsqueeze(1)  # (batch_size, 1, latent_dim)
        
        # Apply transformer layers
        for transformer in self.transformer_layers:
            z_seq = transformer(z_seq)
        
        z_transformed = z_seq.squeeze(1)  # Back to (batch_size, latent_dim)
        
        # Multi-scale convolution features
        z_conv_input = z.unsqueeze(2)  # (batch_size, latent_dim, 1)
        conv_features = []
        for conv in self.multi_scale_conv:
            conv_out = F.relu(conv(z_conv_input))  # (batch_size, hidden_dim, 1)
            conv_features.append(conv_out.squeeze(2))  # (batch_size, hidden_dim)
        
        # Combine features
        combined_features = torch.cat([z_transformed] + conv_features, dim=1)
        
        # Predict discrete atom count
        discrete_logits = self.atom_count_layers(combined_features)
        
        # Predict continuous atom count for auxiliary loss
        continuous_pred = self.continuous_head(z_transformed)
        
        return discrete_logits, continuous_pred


class BetaScheduler:
    """Dynamic beta scheduling for KLD loss."""
    
    def __init__(self, beta_start=0.0, beta_end=0.01, warmup_epochs=10, schedule_type='linear'):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.warmup_epochs = warmup_epochs
        self.schedule_type = schedule_type
        
    def get_beta(self, epoch):
        if epoch < self.warmup_epochs:
            if self.schedule_type == 'linear':
                return self.beta_start + (self.beta_end - self.beta_start) * (epoch / self.warmup_epochs)
            elif self.schedule_type == 'cosine':
                return self.beta_start + (self.beta_end - self.beta_start) * (1 - math.cos(math.pi * epoch / self.warmup_epochs)) / 2
        return self.beta_end


class EnhancedCDVAE(BaseModule):
    """Enhanced CDVAE with improved atom count prediction and regularization."""
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = hydra.utils.instantiate(
            self.hparams.encoder, num_targets=self.hparams.latent_dim)
        self.decoder = hydra.utils.instantiate(self.hparams.decoder)

        self.fc_mu = nn.Linear(self.hparams.latent_dim,
                               self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.latent_dim,
                                self.hparams.latent_dim)

        # Enhanced atom count predictor
        self.atom_count_predictor = ImprovedAtomCountPredictor(
            self.hparams.latent_dim, 
            self.hparams.hidden_dim,
            self.hparams.max_atoms,
            num_layers=getattr(self.hparams, 'transformer_layers', 3),
            num_heads=getattr(self.hparams, 'attention_heads', 8)
        )
        
        # Improved lattice predictor with residual connections
        self.fc_lattice = nn.Sequential(
            nn.Linear(self.hparams.latent_dim, self.hparams.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hparams.hidden_dim * 2, self.hparams.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hparams.hidden_dim, 6)
        )
        
        # Enhanced composition predictor
        self.fc_composition = build_mlp(
            self.hparams.latent_dim, 
            self.hparams.hidden_dim * 2,
            self.hparams.fc_num_layers + 1, 
            MAX_ATOMIC_NUM
        )
        
        # Property prediction with uncertainty
        if self.hparams.predict_property:
            self.fc_property_mean = build_mlp(
                self.hparams.latent_dim, 
                self.hparams.hidden_dim,
                self.hparams.fc_num_layers, 
                1
            )
            self.fc_property_var = build_mlp(
                self.hparams.latent_dim, 
                self.hparams.hidden_dim,
                self.hparams.fc_num_layers, 
                1
            )

        # Noise parameters
        sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.sigma_begin),
            np.log(self.hparams.sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

        type_sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.type_sigma_begin),
            np.log(self.hparams.type_sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)
        self.type_sigmas = nn.Parameter(type_sigmas, requires_grad=False)

        self.embedding = torch.zeros(100, 92)
        for i in range(100):
            self.embedding[i] = torch.tensor(KHOT_EMBEDDINGS[i + 1])

        # Beta scheduler for KLD loss
        self.beta_scheduler = BetaScheduler(
            beta_start=getattr(self.hparams, 'beta_start', 0.0),
            beta_end=getattr(self.hparams, 'beta_end', self.hparams.beta),
            warmup_epochs=getattr(self.hparams, 'beta_warmup_epochs', 10),
            schedule_type=getattr(self.hparams, 'beta_schedule', 'linear')
        )

        # Obtain from datamodule
        self.lattice_scaler = None
        self.scaler = None

    def reparameterize(self, mu, logvar):
        """Reparameterization trick with improved numerical stability."""
        std = torch.exp(0.5 * logvar)
        # Clamp to prevent numerical issues
        std = torch.clamp(std, min=1e-8, max=10.0)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, batch):
        """Encode crystal structures to latents with improved regularization."""
        hidden = self.encoder(batch)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        
        # Clamp log_var to prevent numerical issues
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z

    def predict_num_atoms(self, z):
        """Predict number of atoms with enhanced architecture."""
        discrete_logits, continuous_pred = self.atom_count_predictor(z)
        return discrete_logits, continuous_pred

    def predict_property(self, z):
        """Predict property with uncertainty estimation."""
        if not self.hparams.predict_property:
            return None, None
            
        self.scaler.match_device(z)
        mean = self.scaler.inverse_transform(self.fc_property_mean(z))
        log_var = self.fc_property_var(z)
        return mean, log_var

    def predict_lattice(self, z, num_atoms):
        """Predict lattice parameters with improved architecture."""
        self.lattice_scaler.match_device(z)
        pred_lengths_and_angles = self.fc_lattice(z)
        scaled_preds = self.lattice_scaler.inverse_transform(pred_lengths_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        
        if self.hparams.data.lattice_scale_method == 'scale_length':
            pred_lengths = pred_lengths * num_atoms.view(-1, 1).float()**(1/3)
        
        return pred_lengths_and_angles, pred_lengths, pred_angles

    def predict_composition(self, z, num_atoms):
        """Predict composition with enhanced features."""
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom

    def enhanced_num_atom_loss(self, pred_logits, continuous_pred, batch):
        """Enhanced atom count loss with robust handling of large atom counts."""
        # Get the maximum number of classes our model can handle
        max_classes = pred_logits.size(-1) - 1  # -1 because classes are 0-indexed
        
        # Clamp target atom counts to valid range [0, max_classes]
        target_atoms = torch.clamp(batch.num_atoms, 0, max_classes)
        
        # Create a mask for samples that exceed our model's capacity
        out_of_range_mask = batch.num_atoms > max_classes
        
        if out_of_range_mask.any():
            # For out-of-range samples, use only continuous loss
            valid_mask = ~out_of_range_mask
            if valid_mask.any():
                discrete_loss = F.cross_entropy(pred_logits[valid_mask], target_atoms[valid_mask])
            else:
                discrete_loss = torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
        else:
            # All samples are in valid range
            discrete_loss = F.cross_entropy(pred_logits, target_atoms)
        
        # Continuous regression loss (always computed for all samples)
        continuous_target = batch.num_atoms.float().unsqueeze(1)
        continuous_loss = F.mse_loss(continuous_pred, continuous_target)
        
        # Weight continuous loss higher for out-of-range samples
        continuous_weight = torch.where(out_of_range_mask, 1.0, 0.1).mean()
        
        # Combine losses
        total_loss = discrete_loss + continuous_weight * continuous_loss
        
        return total_loss, discrete_loss, continuous_loss

    def improved_kld_loss(self, mu, log_var):
        """Improved KLD loss with better regularization."""
        # Standard KLD loss
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        
        # Add capacity constraint to prevent posterior collapse
        capacity = getattr(self.hparams, 'kld_capacity', 0.0)
        if capacity > 0:
            kld = torch.abs(kld - capacity)
        
        return torch.mean(kld)

    def forward(self, batch, teacher_forcing, training):
        """Enhanced forward pass with improved loss computation."""
        mu, log_var, z = self.encode(batch)

        # Enhanced atom count prediction
        pred_num_atoms_logits, pred_num_atoms_continuous = self.predict_num_atoms(z)
        pred_num_atoms = pred_num_atoms_logits  # For compatibility

        # Other predictions
        pred_lengths_and_angles, pred_lengths, pred_angles = self.predict_lattice(z, batch.num_atoms)
        pred_composition_per_atom = self.predict_composition(z, batch.num_atoms)

        # Noise sampling (same as original)
        noise_level = torch.randint(0, self.sigmas.size(0),
                                    (batch.num_atoms.size(0),),
                                    device=self.device)
        used_sigmas_per_atom = self.sigmas[noise_level].repeat_interleave(
            batch.num_atoms, dim=0)

        type_noise_level = torch.randint(0, self.type_sigmas.size(0),
                                         (batch.num_atoms.size(0),),
                                         device=self.device)
        used_type_sigmas_per_atom = (
            self.type_sigmas[type_noise_level].repeat_interleave(
                batch.num_atoms, dim=0))

        # Atom type sampling (same as original)
        pred_composition_probs = F.softmax(
            pred_composition_per_atom.detach(), dim=-1)
        atom_type_probs = (
            F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM) +
            pred_composition_probs * used_type_sigmas_per_atom[:, None])
        rand_atom_types = torch.multinomial(
            atom_type_probs, num_samples=1).squeeze(1) + 1

        # Coordinate noise (same as original)
        cart_noises_per_atom = (
            torch.randn_like(batch.frac_coords) *
            used_sigmas_per_atom[:, None])
        cart_coords = frac_to_cart_coords(
            batch.frac_coords, pred_lengths, pred_angles, batch.num_atoms)
        cart_coords = cart_coords + cart_noises_per_atom
        noisy_frac_coords = cart_to_frac_coords(
            cart_coords, pred_lengths, pred_angles, batch.num_atoms)

        pred_cart_coord_diff, pred_atom_types = self.decoder(
            z, noisy_frac_coords, rand_atom_types, batch.num_atoms, pred_lengths, pred_angles)

        # Enhanced loss computation
        num_atom_loss, discrete_atom_loss, continuous_atom_loss = self.enhanced_num_atom_loss(
            pred_num_atoms_logits, pred_num_atoms_continuous, batch)
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        composition_loss = self.composition_loss(
            pred_composition_per_atom, batch.atom_types, batch)
        coord_loss = self.coord_loss(
            pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch)
        type_loss = self.type_loss(pred_atom_types, batch.atom_types,
                                   used_type_sigmas_per_atom, batch)

        kld_loss = self.improved_kld_loss(mu, log_var)

        if self.hparams.predict_property:
            property_mean, property_var = self.predict_property(z)
            if property_mean is not None:
                property_loss = self.property_loss_with_uncertainty(
                    property_mean, property_var, batch)
            else:
                property_loss = 0.
        else:
            property_loss = 0.

        return {
            'num_atom_loss': num_atom_loss,
            'discrete_atom_loss': discrete_atom_loss,
            'continuous_atom_loss': continuous_atom_loss,
            'lattice_loss': lattice_loss,
            'composition_loss': composition_loss,
            'coord_loss': coord_loss,
            'type_loss': type_loss,
            'kld_loss': kld_loss,
            'property_loss': property_loss,
            'pred_num_atoms': pred_num_atoms,
            'pred_lengths_and_angles': pred_lengths_and_angles,
            'pred_lengths': pred_lengths,
            'pred_angles': pred_angles,
            'pred_cart_coord_diff': pred_cart_coord_diff,
            'pred_atom_types': pred_atom_types,
            'pred_composition_per_atom': pred_composition_per_atom,
            'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,
            'rand_frac_coords': noisy_frac_coords,
            'rand_atom_types': rand_atom_types,
            'z': z,
        }

    def property_loss_with_uncertainty(self, pred_mean, pred_log_var, batch):
        """Property loss with uncertainty estimation."""
        if pred_mean is None:
            return 0.
        
        # Negative log-likelihood loss
        pred_var = torch.exp(pred_log_var)
        loss = 0.5 * (torch.log(pred_var) + (batch.y - pred_mean)**2 / pred_var)
        return torch.mean(loss)

    def compute_stats(self, batch, outputs, prefix):
        """Enhanced statistics computation with dynamic beta scheduling."""
        num_atom_loss = outputs['num_atom_loss']
        lattice_loss = outputs['lattice_loss']
        coord_loss = outputs['coord_loss']
        type_loss = outputs['type_loss']
        kld_loss = outputs['kld_loss']
        composition_loss = outputs['composition_loss']
        property_loss = outputs['property_loss']

        # Dynamic beta scheduling
        current_beta = self.beta_scheduler.get_beta(self.current_epoch)

        # Enhanced loss weighting
        atom_count_weight = getattr(self.hparams, 'cost_natom_enhanced', self.hparams.cost_natom * 2.0)
        
        loss = (
            atom_count_weight * num_atom_loss +
            self.hparams.cost_lattice * lattice_loss +
            self.hparams.cost_coord * coord_loss +
            self.hparams.cost_type * type_loss +
            current_beta * kld_loss +
            self.hparams.cost_composition * composition_loss +
            self.hparams.cost_property * property_loss)

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_natom_loss': num_atom_loss,
            f'{prefix}_discrete_atom_loss': outputs.get('discrete_atom_loss', 0),
            f'{prefix}_continuous_atom_loss': outputs.get('continuous_atom_loss', 0),
            f'{prefix}_lattice_loss': lattice_loss,
            f'{prefix}_coord_loss': coord_loss,
            f'{prefix}_type_loss': type_loss,
            f'{prefix}_kld_loss': kld_loss,
            f'{prefix}_composition_loss': composition_loss,
            f'{prefix}_current_beta': current_beta,
        }

        if prefix != 'train':
            # Validation/test specific metrics (same as original)
            loss = (
                self.hparams.cost_coord * coord_loss +
                self.hparams.cost_type * type_loss)

            pred_num_atoms = outputs['pred_num_atoms'].argmax(dim=-1)
            num_atom_accuracy = (
                pred_num_atoms == batch.num_atoms).sum() / batch.num_graphs

            pred_lengths_and_angles = outputs['pred_lengths_and_angles']
            scaled_preds = self.lattice_scaler.inverse_transform(
                pred_lengths_and_angles)
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]

            if self.hparams.data.lattice_scale_method == 'scale_length':
                pred_lengths = pred_lengths * \
                    batch.num_atoms.view(-1, 1).float()**(1/3)
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))

            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(
                batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)

            pred_atom_types = outputs['pred_atom_types']
            target_atom_types = outputs['target_atom_types']
            type_accuracy = pred_atom_types.argmax(
                dim=-1) == (target_atom_types - 1)
            type_accuracy = scatter(type_accuracy.float(
            ), batch.batch, dim=0, reduce='mean').mean()

            log_dict.update({
                f'{prefix}_loss': loss,
                f'{prefix}_property_loss': property_loss,
                f'{prefix}_natom_accuracy': num_atom_accuracy,
                f'{prefix}_lengths_mard': lengths_mard,
                f'{prefix}_angles_mae': angles_mae,
                f'{prefix}_volumes_mard': volumes_mard,
                f'{prefix}_type_accuracy': type_accuracy,
            })

        return log_dict, loss

    # Keep all other methods from the original CDVAE class
    def lattice_loss(self, pred_lengths_and_angles, batch):
        """Same as original."""
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        if self.hparams.data.lattice_scale_method == 'scale_length':
            target_lengths = batch.lengths / \
                batch.num_atoms.view(-1, 1).float()**(1/3)
        target_lengths_and_angles = torch.cat(
            [target_lengths, batch.angles], dim=-1)
        target_lengths_and_angles = self.lattice_scaler.transform(
            target_lengths_and_angles)
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch):
        """Same as original."""
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_composition_per_atom,
                               target_atom_types, reduction='none')
        return scatter(loss, batch.batch, reduce='mean').mean()

    def coord_loss(self, pred_cart_coord_diff, noisy_frac_coords,
                   used_sigmas_per_atom, batch):
        """Same as original."""
        noisy_cart_coords = frac_to_cart_coords(
            noisy_frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        target_cart_coords = frac_to_cart_coords(
            batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        _, target_cart_coord_diff = min_distance_sqr_pbc(
            target_cart_coords, noisy_cart_coords, batch.lengths, batch.angles,
            batch.num_atoms, self.device, return_vector=True)

        target_cart_coord_diff = target_cart_coord_diff / \
            used_sigmas_per_atom[:, None]**2
        pred_cart_coord_diff = pred_cart_coord_diff / \
            used_sigmas_per_atom[:, None]

        loss_per_atom = torch.sum(
            (target_cart_coord_diff - pred_cart_coord_diff)**2, dim=1)

        loss_per_atom = 0.5 * loss_per_atom * used_sigmas_per_atom**2
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()

    def type_loss(self, pred_atom_types, target_atom_types,
                  used_type_sigmas_per_atom, batch):
        """Same as original."""
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(
            pred_atom_types, target_atom_types, reduction='none')
        loss = loss / used_type_sigmas_per_atom
        return scatter(loss, batch.batch, reduce='mean').mean()

    # Keep all other methods from original CDVAE...
    def decode_stats(self, z, gt_num_atoms=None, gt_lengths=None, gt_angles=None,
                     teacher_forcing=False):
        """Decode stats with enhanced atom count prediction."""
        if gt_num_atoms is not None:
            num_atoms_logits, _ = self.predict_num_atoms(z)
            num_atoms = num_atoms_logits  # For compatibility
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, gt_num_atoms))
            composition_per_atom = self.predict_composition(z, gt_num_atoms)
            if self.hparams.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
        else:
            num_atoms_logits, _ = self.predict_num_atoms(z)
            num_atoms = num_atoms_logits.argmax(dim=-1)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, num_atoms))
            composition_per_atom = self.predict_composition(z, num_atoms)
        return num_atoms, lengths_and_angles, lengths, angles, composition_per_atom

    # Include all other methods from the original CDVAE class...
    @torch.no_grad()
    def langevin_dynamics(self, z, ld_kwargs, gt_num_atoms=None, gt_atom_types=None):
        """Optimized Langevin dynamics with inference speedups."""
        if ld_kwargs.save_traj:
            all_frac_coords = []
            all_pred_cart_coord_diff = []
            all_noise_cart = []
            all_atom_types = []

        # obtain key stats.
        num_atoms, _, lengths, angles, composition_per_atom = self.decode_stats(
            z, gt_num_atoms)
        if gt_num_atoms is not None:
            num_atoms = gt_num_atoms

        # obtain atom types.
        composition_per_atom = F.softmax(composition_per_atom, dim=-1)
        if gt_atom_types is None:
            cur_atom_types = self.sample_composition(
                composition_per_atom, num_atoms)
        else:
            cur_atom_types = gt_atom_types

        # init coords.
        cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=z.device)

        # Optimized annealed langevin dynamics - skip very small sigmas for speed
        active_sigmas = self.sigmas[self.sigmas >= ld_kwargs.min_sigma]
        
        for sigma in tqdm(active_sigmas, total=active_sigmas.size(0), disable=ld_kwargs.disable_bar):
            step_size = ld_kwargs.step_lr * (sigma / self.sigmas[-1]) ** 2

            for step in range(ld_kwargs.n_step_each):
                noise_cart = torch.randn_like(
                    cur_frac_coords) * torch.sqrt(step_size * 2)
                pred_cart_coord_diff, pred_atom_types = self.decoder(
                    z, cur_frac_coords, cur_atom_types, num_atoms, lengths, angles)
                cur_cart_coords = frac_to_cart_coords(
                    cur_frac_coords, lengths, angles, num_atoms)
                pred_cart_coord_diff = pred_cart_coord_diff / sigma
                cur_cart_coords = cur_cart_coords + step_size * pred_cart_coord_diff + noise_cart
                cur_frac_coords = cart_to_frac_coords(
                    cur_cart_coords, lengths, angles, num_atoms)

                if gt_atom_types is None:
                    cur_atom_types = torch.argmax(pred_atom_types, dim=1) + 1

                if ld_kwargs.save_traj:
                    all_frac_coords.append(cur_frac_coords)
                    all_pred_cart_coord_diff.append(
                        step_size * pred_cart_coord_diff)
                    all_noise_cart.append(noise_cart)
                    all_atom_types.append(cur_atom_types)

        output_dict = {'num_atoms': num_atoms, 'lengths': lengths, 'angles': angles,
                       'frac_coords': cur_frac_coords, 'atom_types': cur_atom_types,
                       'is_traj': False}

        if ld_kwargs.save_traj:
            output_dict.update(dict(
                all_frac_coords=torch.stack(all_frac_coords, dim=0),
                all_atom_types=torch.stack(all_atom_types, dim=0),
                all_pred_cart_coord_diff=torch.stack(
                    all_pred_cart_coord_diff, dim=0),
                all_noise_cart=torch.stack(all_noise_cart, dim=0),
                is_traj=True))

        return output_dict

    def sample(self, num_samples, ld_kwargs):
        """Sample new structures with inference optimizations."""
        # Set model to eval mode and disable gradients for faster inference
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.hparams.latent_dim,
                            device=self.device)
            samples = self.langevin_dynamics(z, ld_kwargs)
        return samples

    def sample_composition(self, composition_prob, num_atoms):
        """Same as original."""
        batch = torch.arange(
            len(num_atoms), device=num_atoms.device).repeat_interleave(num_atoms)
        assert composition_prob.size(0) == num_atoms.sum() == batch.size(0)
        composition_prob = scatter(
            composition_prob, index=batch, dim=0, reduce='mean')

        all_sampled_comp = []

        for comp_prob, num_atom in zip(list(composition_prob), list(num_atoms)):
            comp_num = torch.round(comp_prob * num_atom)
            atom_type = torch.nonzero(comp_num, as_tuple=True)[0] + 1
            atom_num = comp_num[atom_type - 1].long()

            sampled_comp = atom_type.repeat_interleave(atom_num, dim=0)

            if sampled_comp.size(0) < num_atom:
                left_atom_num = num_atom - sampled_comp.size(0)
                left_comp_prob = comp_prob - comp_num.float() / num_atom
                left_comp_prob[left_comp_prob < 0.] = 0.
                left_comp = torch.multinomial(
                    left_comp_prob, num_samples=left_atom_num, replacement=True)
                left_comp = left_comp + 1
                sampled_comp = torch.cat([sampled_comp, left_comp], dim=0)

            sampled_comp = sampled_comp[torch.randperm(sampled_comp.size(0))]
            sampled_comp = sampled_comp[:num_atom]
            all_sampled_comp.append(sampled_comp)

        all_sampled_comp = torch.cat(all_sampled_comp, dim=0)
        assert all_sampled_comp.size(0) == num_atoms.sum()
        return all_sampled_comp

    def generate_rand_init(self, pred_composition_per_atom, pred_lengths,
                           pred_angles, num_atoms, batch):
        """Same as original."""
        rand_frac_coords = torch.rand(num_atoms.sum(), 3,
                                      device=num_atoms.device)
        pred_composition_per_atom = F.softmax(pred_composition_per_atom,
                                              dim=-1)
        rand_atom_types = self.sample_composition(
            pred_composition_per_atom, num_atoms)
        return rand_frac_coords, rand_atom_types

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Enhanced training step."""
        teacher_forcing = (
            self.current_epoch <= self.hparams.teacher_forcing_max_epoch)
        outputs = self(batch, teacher_forcing, training=True)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Enhanced validation step."""
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Enhanced test step."""
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
        self.log_dict(
            log_dict,
        )
        return loss