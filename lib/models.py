import glob
import json
import os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from . import layers
from .hparams import HParams

def get_model(*, hparams=None, model_dir=None, ckpt_file=None, load_parameters=True):
    assert (hparams is None) != (model_dir is None), 'Exactly one of `hparams` or `model_dir` should be given'

    if model_dir is not None:
        assert os.path.exists(model_dir), f'model_dir does not exist: {model_dir}'
        assert os.path.exists(os.path.join(model_dir, 'hparams.json')), f'hparams.json does not exist in {model_dir}'

        if ckpt_file is None:
            ckpts = glob.glob(os.path.join(model_dir, 'ckpt_*.pt'))
            ckpt_file = sorted(ckpts)[-1]  # use latest checkpoint
        assert os.path.exists(ckpt_file), f'Checkpoint file "{ckpt_file}" does not exist'

        # Load hyperparameters
        with open(os.path.join(model_dir, 'hparams.json'), 'r') as f:
            hp_dict = json.load(f)
        hps = HParams(**hp_dict)
        dump = torch.load(ckpt_file)

        if hps.model == 'realnvp':
            img_shape = [hps.n_channels, hps.size, hps.size]
            model = RealNVP(img_shape, hps.d_hidden, hps.n_blocks, hps.n_scales, hps.logit_eps, hps.n_bits)

        if load_parameters:
            # # TODO: clean this up
            # dd = dump['model_state_dict']
            # if 'mu' in dd: del dd['mu']
            # if 'sigma' in dd: del dd['sigma']
            # if '_dummy' not in dd:
            #     dd['_dummy'] = torch.Tensor([0], device=dd['model.layers.0.mask'].device)
            # model.load_state_dict(dd)

            model.load_state_dict(dump['model_state_dict'])

        print(f'Loaded {hps.model} model: {model.param_count()} trainable parameters (total {model.param_count(requires_grad=False)})')

    else:
        if hparams.model == 'realnvp':
            img_shape = [hparams.n_channels, hparams.size, hparams.size]
            model = RealNVP(img_shape, hparams.d_hidden, hparams.n_blocks, hparams.n_scales,
                            hparams.logit_eps, hparams.n_bits)

        print(f'Created {hparams.model} model: {model.param_count()} trainable parameters (total {model.param_count(requires_grad=False)})')

    return model

class FlowModel(nn.Module):
    def sample_x(self):
        raise NotImplementedError
    def sample_z(self):
        raise NotImplementedError
    def inverse(self, z):
        raise NotImplementedError
    def flatten_z(self, z):
        raise NotImplementedError
    def unflatten_z(self, z):
        raise NotImplementedError
    def log_prob(self, x):
        raise NotImplementedError
    def log_prior(self, z):
        raise NotImplementedError
    def preprocess(self, x):
        raise NotImplementedError
    def param_count(self, requires_grad=True):
        if requires_grad:
            return sum(int(np.prod(p.shape)) for p in self.parameters() if p.requires_grad)
        else:
            return sum(int(np.prod(p.shape)) for p in self.parameters())

class RealNVP(FlowModel):
    def __init__(self, img_shape, d_hid, n_blocks, n_scales, logit_eps, n_bits):
        self.img_shape = img_shape
        self.D = int(np.prod(img_shape))
        self.d_hid = d_hid
        self.n_blocks = n_blocks
        self.n_scales = n_scales
        self.logit_eps = logit_eps
        self.n_bits = n_bits

        # modules = [layers.Scale(torch.ones(1, *self.img_shape) / 256), layers.Logit()]
        shape = list(self.img_shape)
        self.z_shapes = []
        modules = []
        for _ in range(self.n_scales-1):
            modules.append(layers.CouplingLayer(shape, self.d_hid, 'checker0', num_blocks=self.n_blocks))
            modules.append(layers.CouplingLayer(shape, self.d_hid, 'checker1', num_blocks=self.n_blocks))
            modules.append(layers.CouplingLayer(shape, self.d_hid, 'checker0', num_blocks=self.n_blocks))
            modules.append(layers.Squeeze(2))

            assert shape[1] % 2 == 0 and shape[2] % 2 == 0
            shape = [shape[0]*4, shape[1]//2, shape[2]//2]

            modules.append(layers.CouplingLayer(shape, self.d_hid, 'channel0', num_blocks=self.n_blocks))
            modules.append(layers.CouplingLayer(shape, self.d_hid, 'channel1', num_blocks=self.n_blocks))
            modules.append(layers.CouplingLayer(shape, self.d_hid, 'channel0', num_blocks=self.n_blocks))
            modules.append(layers.Factor())

            assert shape[0] % 2 == 0
            shape = [shape[0]//2, shape[1], shape[2]]
            self.z_shapes.append(list(shape))

        # Final layer
        modules.append(layers.CouplingLayer(shape, self.d_hid, 'checker0', num_blocks=self.n_blocks))
        modules.append(layers.CouplingLayer(shape, self.d_hid, 'checker1', num_blocks=self.n_blocks))
        modules.append(layers.CouplingLayer(shape, self.d_hid, 'checker0', num_blocks=self.n_blocks))
        modules.append(layers.CouplingLayer(shape, self.d_hid, 'checker1', num_blocks=self.n_blocks))

        self.z_shapes.append(list(shape))
        assert len(self.z_shapes) == self.n_scales
        assert sum(int(np.prod(s)) for s in self.z_shapes) == self.D
        print('z_shapes: ' + ', '.join([str(sh) for sh in self.z_shapes]))

        super().__init__()
        self.model = layers.FlowSequential(modules)
        self._dummy = nn.Parameter(torch.empty(1), requires_grad=False)  # Used to keep track of this model's device

    @property
    def device(self):
        return self._dummy.device

    def forward(self, x):
        assert x.ndim == 4
        final_z, logdet, factored = self.model(x)
        zs = factored + [final_z]
        return self.flatten_z(zs), logdet

    def inverse(self, z):
        assert z.ndim == 2
        z = self.unflatten_z(z)
        x, _, _ = self.model(z[-1], None, z[:-1], inverse=True)
        return x

    def log_prob(self, x):
        assert x.ndim == 4
        # x, transform_logdet = self.preprocess(x)
        z, logdet = self.forward(x)
        log_pz = self.log_prior(z)
        assert log_pz.shape == logdet.shape
        # log_px = log_pz + logdet + transform_logdet
        log_px = log_pz + logdet
        return log_px, log_pz, logdet

    def sample_x(self, n=10, temp=0.7):
        z = self.unflatten_z(self.sample_z(n, temp=temp))
        x = self.inverse(z)
        assert x.shape == torch.Size(n, *self.img_shape)
        return x

    def sample_z(self, n=10, temp=0.7):
        z = torch.randn(n, self.D, device=self.device) * temp
        return z

    def flatten_z(self, zs):
        assert len(zs) == self.n_scales
        assert sum(int(np.prod(z.shape[1:])) for z in zs) == self.D
        N = len(zs[0])
        out = torch.cat([z.view(N,-1) for z in zs], dim=1)
        assert out.shape == torch.Size([N, self.D])
        return out

    def unflatten_z(self, z_unflat):
        assert z_unflat.ndim == 2 and z_unflat.shape[1] == self.D
        N = z_unflat.shape[0]
        zs = z_unflat.split([int(np.prod(s)) for s in self.z_shapes], dim=1)
        zs = [z.view(N, *s) for z,s in zip(zs,self.z_shapes)]
        return zs

    def log_prior(self, z):
        assert z.ndim == 2
        log_pdf = -0.5*(self.D*np.log(2*np.pi)+z.norm(dim=1,keepdim=True)**2)
        return log_pdf.to(self.device)

    def preprocess(self, x):
        assert x.dtype == torch.float
        # x = x.float()
        # if dequantize:
        #     x = x + torch.rand_like(x)
        # x = x / 256
        xx = (1-2*self.logit_eps) * x.float() + self.logit_eps
        out = torch.log(xx) - torch.log(1-xx)
        assert torch.isfinite(out).all(), 'nan/inf found during logit transformation'
        # Logdet from preprocessing input
        logdet = (self.D * np.log(1-2*self.logit_eps) -
                self.D * np.log(2**self.n_bits) -
                torch.sum(torch.log(1-xx)+torch.log(xx), dim=[1,2,3])).view(-1,1)
        assert logdet.shape == torch.Size([len(x),1])
        return out, logdet

    def postprocess(self, x, undo_logit=True):
        x = torch.sigmoid(x)
        if undo_logit:
            x = (x-self.logit_eps) / (1-2*self.logit_eps)
        # x = torch.clamp((x-self.logit_eps) / (1-2*self.logit_eps), 0, 1)
        # assert (x.min() >= 0) and (x.max() <= 1)
        return x
