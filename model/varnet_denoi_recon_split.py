"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import fastmri
import torch
import torch.nn as nn

from fastmri.models.varnet import NormUnet, SensitivityModel, VarNetBlock
from utils_n2n.utils import compute_number_of_params

class VarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        denoi_model: str = 'unet',
        nlayers_dncnn: int = 6
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
        """
        super().__init__()

        ngpus = min([torch.cuda.device_count(), 2])
        available_gpus = ['cuda:' + str(ii) for ii in range(ngpus)]
        # available_gpus = ['cuda:0']
        # ngpus = 1

        if ngpus > 0:
            self.ncasc_on_gpu = [num_cascades // ngpus] * (ngpus - 1)
            self.ncasc_on_gpu.append(num_cascades - sum(self.ncasc_on_gpu))
            self.dev_on_cascade = []
            for ii in range(ngpus):
                for _ in range(self.ncasc_on_gpu[ii]):
                    self.dev_on_cascade.append(available_gpus[ii])
        else:
            self.dev_on_cascade = ['cpu'] * num_cascades

        self.sens_net = SensitivityModel(sens_chans, sens_pools).to(self.dev_on_cascade[0])

        if denoi_model == "split_unet": # proposed split unet i.e. Denoising VarNet
            self.cascades = nn.ModuleList([VarNetBlockReconDenoi(NormUnet(chans, pools), NormUnet(chans, pools))
                                           for _ in range(num_cascades)])
        elif denoi_model == "unet": # cascade for standard varnet
            self.cascades = nn.ModuleList([VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)])
        else:
            raise Exception('{} is not a valid denoi_model type'.format(denoi_model))

        for ii in range(num_cascades):
            self.cascades[ii] = self.cascades[ii].to(self.dev_on_cascade[ii])

        self._initialize_weights()

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_kspace = masked_kspace.to(self.dev_on_cascade[0])
        mask = mask.to(self.dev_on_cascade[0])
        sens_maps = self.sens_net(masked_kspace, mask)
        # sens_maps = torch.ones(sens_maps.shape) # !!
        kspace_pred = masked_kspace.clone()

        casc_idx = 0
        for cascade in self.cascades:
            dev = self.dev_on_cascade[casc_idx]
            kspace_pred = kspace_pred.to(dev)
            masked_kspace = masked_kspace.to(dev)
            mask = mask.to(dev)
            sens_maps = sens_maps.to(dev)
            cascade = cascade.to(dev)
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)
            casc_idx += 1

        return kspace_pred

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, 1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)


class VarNetBlockReconDenoi(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model_recon, model_denoi):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()
        self.model_recon = model_recon
        self.model_denoi = model_denoi
        # pytorch_total_params = compute_number_of_params(model_recon)
        # print('total number of parameters in recon is {:e}'.format(pytorch_total_params))
        # pytorch_total_params = compute_number_of_params(model_denoi)
        # print('total number of parameters in denoi is {:e}'.format(pytorch_total_params))

        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight.to(current_kspace)

        im = self.sens_reduce(current_kspace, sens_maps)
        model_term_recon = self.sens_expand(self.model_recon(im), sens_maps)
        model_term_denoi = self.sens_expand(self.model_denoi(im), sens_maps)

        return current_kspace - soft_dc - (~mask)*model_term_recon - mask*model_term_denoi

