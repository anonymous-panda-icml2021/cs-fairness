import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import weight_norm

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
                nn.BatchNorm2d(dim, eps=1e-4),
                nn.ReLU(),
                weight_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=1)),
                nn.BatchNorm2d(dim, eps=1e-4),
                nn.ReLU(),
                weight_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=1)),
        )

    def forward(self, x):
        return x + self.block(x)

class Resnet(nn.Module):
    def __init__(self, d_in, d_hid, d_out, num_blocks, skip=True):
        assert num_blocks > 0
        self.num_blocks = num_blocks
        self.skip = skip
        super().__init__()

        # Initial conv + residual blocks
        self.blocks = nn.ModuleList(
                [weight_norm(nn.Conv2d(d_in, d_hid, kernel_size=3, padding=1))] +
                [ResidualBlock(d_hid) for _ in range(num_blocks)])

        # Skip connections
        if skip:
            self.skip_convs = nn.ModuleList([
                weight_norm(nn.Conv2d(d_hid, d_hid, kernel_size=1, padding=0)) for _ in range(num_blocks+1)])

        # Final output
        self.out_block = nn.Sequential(
                nn.BatchNorm2d(d_hid, eps=1e-4),
                nn.ReLU(),
                weight_norm(nn.Conv2d(d_hid, d_out, kernel_size=1, padding=0)))

    def forward(self, x):
        out = 0
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.skip: out += self.skip_convs[i](x)
        if self.skip: x = out
        x = self.out_block(x)
        return x

def get_mask(mask, shape):
    C, H, W = shape  # pytorch uses NCHW

    if mask.startswith('checker'):
        HH, WW = (H+1)//2, (W+1)//2
        out = torch.eye(2).repeat(C,HH,WW).view(1,C,2*HH,2*WW)[:,:,:H,:W]
    elif mask.startswith('channel'):
        assert C % 2 == 0
        out = torch.cat([torch.ones(1,C//2,H,W), torch.zeros(1,C//2,H,W)], 1)
    else:
        raise ValueError(f'Invalid mask {mask}')

    if mask[-1] == '1':
        out = 1 - out

    assert out.shape == torch.Size([1,C,H,W]) and out.dtype == torch.float32
    return out

def _L(x, logdet):
    if logdet is None:
        return torch.zeros(x.shape[0], 1, device=x.device)
    else:
        return logdet

class Flow(nn.Module):
    def forward(self, x, logdet=0, factored=[], inverse=False):
        raise NotImplementedError

class CouplingLayer(Flow):
    def __init__(self, shape, d_hid, mask, num_blocks=5):
        assert mask in ('checker0', 'checker1', 'channel0', 'channel1')
        C, H, W = shape
        mask = get_mask(mask, shape)

        super().__init__()
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.init_bn = nn.BatchNorm2d(C, eps=1e-4) # Initial batch norm
        self.logs_and_t = Resnet(2*C, d_hid, 2*C, num_blocks)
        self.scale_param = nn.Parameter(torch.zeros(1, *shape), requires_grad=True)

    def forward(self, x, logdet=None, factored=[], inverse=False):
        N, C, H, W = x.shape
        out = self.mask * x
        out = self.init_bn(out) * 2
        out = torch.cat([out, -out], dim=1)
        out = F.relu(out)
        assert out.shape == torch.Size([N, 2*C, H, W])

        logs, shift = self.logs_and_t(out).split(C, dim=1)
        assert logs.shape == shift.shape == x.shape
        assert logs.shape[1:] == self.scale_param.shape[1:]
        logs = (logs.tanh() * self.scale_param) * (1-self.mask)
        shift = shift * (1-self.mask)

        if inverse:
            scale = (-logs).exp()
            assert torch.isfinite(scale).all(), f'inf/nan in scale during inverse'
            out = (x - shift) * scale
            # TODO: implement backward logdet
            return out, None, factored
        else:
            scale = logs.exp()
            assert torch.isfinite(scale).all(), f'inf/nan in scale'
            out = x * scale + shift
            logdet = _L(x,logdet) + logs.sum(dim=list(range(1,x.ndim))).view(-1,1)
            return out, logdet, factored

class Squeeze(Flow):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x, logdet=None, factored=[], inverse=False):
        k = self.factor

        if inverse:
            N, Ckk, HH, WW = x.shape
            assert Ckk % (k*k) == 0, f'channel count not divisible by {k}^2'
            C = Ckk // (k*k)
            x = x.view(N, C, k, k, HH, WW)  # N, C, k_H, k_W, H/k, W/k
            x = x.permute(0, 1, 4, 2, 5, 3) # N, C, H/k, k_H, W/k, k_W
            x = x.contiguous().view(N, C, HH*k, WW*k)
            return x, _L(x,logdet), factored

        else:
            N, C, H, W = x.shape
            assert H % k == 0 and W % k == 0, f'image size not divisible by squeezing factor {k}'
            x = x.view(N, C, H//k, k, W//k, k)  # N, C, H/k, k_H, W/k, k_W
            x = x.permute(0, 1, 3, 5, 2, 4)  # N, C, k_H, k_W, H/k, W/k
            x = x.contiguous().view(N, C*k*k, H//k, W//k)

            # Squeezing is volume-preserving, so logdet = 0
            return x, _L(x,logdet), factored

class Factor(Flow):
    def __init__(self):
        super().__init__()

    def forward(self, x, logdet=None, factored=[], inverse=False):
        N, C, H, W = x.shape

        if inverse:
            assert len(factored) > 0, 'Cannot invert Factoring layer without z'
            out = torch.cat([x, factored[-1]], dim=1)
            return out, _L(x,logdet), factored[:-1]
        else:
            assert C % 2 == 0, f'channel count not even during Factoring'
            xp, zp = x.split(x.shape[1]//2, dim=1)
            return xp, _L(x,logdet), factored+[zp]

class FlowSequential(nn.Module):
    def __init__(self, layers, debug=False):
        for layer in layers:
            assert isinstance(layer, Flow)
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.debug = debug

    def forward(self, x, logdet=None, factored=[], inverse=False):
        for layer in self.layers[::1-int(inverse)*2]:
            if self.debug:
                msg = 'Inverting' if inverse else 'Forwarding'
                print(f'[FlowSequential] {msg} {layer.__class__.__name__} layer -> x.shape: {x.shape}, len(factored): {len(factored)}')
            x, logdet, factored = layer(x, logdet, factored, inverse=inverse)
            if self.debug:
                print(f'                 Result -> x.shape: {x.shape}, len(factored): {len(factored)}')
        return x, logdet, factored

class Sigmoid(Flow):
    def forward(self, x):
        z = torch.sigmoid(x)
        # TODO: fix shapes
        logdet = torch.log((1-z)*z).sum(-1, keepdims=True)
        return z, logdet

    def backward(self, z):
        return torch.log(z) - torch.log(1-z)

class Logit(Flow):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = nn.Parameter(torch.Tensor([eps]), requires_grad=False)

    def forward(self, x):
        # x_clamped = self.eps + (1-2*self.eps)*x
        # z = torch.log(x_clamped) - torch.log(1-x_clamped)
        # logdet = (torch.log(1-2*self.eps) - torch.log(x_clamped) - torch.log(1-x_clamped)).sum(list(range(1,x_clamped.ndim))).view(-1,1)

        x_clamped = self.eps + (1-2*self.eps)*x
        z = torch.log(x_clamped) - torch.log(1-x_clamped)
        logdet = (-torch.log(x_clamped) - torch.log(1-x_clamped)).sum(list(range(1,x_clamped.ndim))).view(-1,1)

        assert logdet.shape == torch.Size([x.shape[0],1])
        assert torch.isfinite(z).all(), f'[DEBUG] Logit output has nan/inf. x={x.mean()}, z={z.mean()}, logdet={logdet.mean()}'
        assert torch.isfinite(logdet).all(), '[DEBUG] Logit logdet has nan/inf. x={x.mean()}, z={z.mean()}, logdet={logdet.mean()}'
        return z, logdet

    def backward(self, z):
        # x = (torch.sigmoid(z)-self.eps) / (1-2*self.eps)
        x = torch.sigmoid(z)
        assert torch.isfinite(x).all(), f'[DEBUG] Logit backward output has nan/inf. z={z.mean()}, x={x.mean()}, eps={eps.item()}'
        assert x.min() >= 0 and x.max() <= 1, f'[DEBUG] Logit backward output out of range [0,1]'
        return x

class Scale(Flow):
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(scale, requires_grad=False)

    def forward(self, x):
        z =  x * self.scale
        logdet = self.scale.log().sum(list(range(1,x.ndim))).view(1, 1)
        logdet = logdet.repeat(x.shape[0], 1)
        assert logdet.shape == torch.Size([x.shape[0],1]), ipdb.set_trace()
        return z, logdet

    def backward(self, z):
        return z / self.scale

## TESTS

allclose = lambda x, y: torch.allclose(x, y, atol=1e-6, rtol=1e-4)

def test_get_mask():
    m = get_mask('checker0', [3, 4, 4])
    assert allclose(m, torch.eye(2).unsqueeze(0).repeat(3,2,2).unsqueeze(0))
    m = get_mask('checker1', [3, 4, 4])
    assert allclose(m, 1-torch.eye(2).unsqueeze(0).repeat(3,2,2).unsqueeze(0))
    m = get_mask('channel0', [4, 3, 3])
    x = torch.cat([torch.ones(2,3,3), torch.zeros(2,3,3)], dim=0).unsqueeze(0)
    assert allclose(m, torch.cat([torch.ones(2,3,3), torch.zeros(2,3,3)], dim=0).unsqueeze(0))
    m = get_mask('channel1', [4, 3, 3])
    assert allclose(m, 1-torch.cat([torch.ones(2,3,3), torch.zeros(2,3,3)], dim=0).unsqueeze(0))

def test_coupling_layer(mask):
    x = torch.randn(17, 12, 16, 16)
    c = CouplingLayer([12, 16, 16], 5, mask=mask, num_blocks=2)

    # x -> xinv -> x
    xinv = c(x, inverse=True)[0]
    xp, logdet, _ = c(xinv)
    assert x.shape == xinv.shape == xp.shape
    assert logdet.shape == (17, 1)
    assert allclose(x, xp)

    # xinv -> x -> xinv
    y, logdet, _ = c(x)
    xp = c(y, inverse=True)[0]
    assert x.shape == y.shape == xp.shape
    assert logdet.shape == (17, 1)
    assert allclose(x, xp)

def test_squeeze():
    x = torch.randn(17, 8, 4, 4)
    s = Squeeze(2)

    # x -> xinv -> x
    xinv = s(x, inverse=True)[0]
    xp, logdet, _ = s(xinv)
    assert xinv.shape == torch.Size([17,2,8,8])
    assert x.shape == xp.shape
    assert allclose(torch.zeros(17,1), logdet)
    assert allclose(x, xp)

    # xinv -> x -> xinv
    y, logdet, _ = s(x)
    xp = s(y, inverse=True)[0]
    assert y.shape == torch.Size([17,32,2,2])
    assert x.shape == xp.shape
    assert allclose(torch.zeros(17,1), logdet)
    assert allclose(x, xp)

def test_factor():
    x = torch.randn(17, 6, 5, 5)
    y, z = x.split(3, dim=1)
    f = Factor()

    # xinv -> x -> xinv
    yp, logdet, zp = f(x)
    xp, _, factored = f(yp, logdet, zp, inverse=True)
    assert x.shape == xp.shape
    assert y.shape == yp.shape
    assert len(zp) == 1 and zp[0].shape == z.shape
    assert factored == []
    assert allclose(torch.zeros(17,1), logdet)
    assert allclose(x, xp)
    assert allclose(y, yp)
    assert allclose(z, zp[0])

    # x -> xinv -> x
    xp, logdet, factored = f(y, None, [z], inverse=True)
    yp, _, zp = f(xp, logdet, factored)
    assert x.shape == xp.shape
    assert y.shape == yp.shape
    assert z.shape == zp[0].shape
    assert len(zp) == 1 and zp[0].shape == z.shape
    assert allclose(torch.zeros(17,1), logdet)
    assert allclose(x, xp)
    assert allclose(y, yp)
    assert allclose(z, zp[0])

def test_flow_sequential():
    x = torch.randn(17, 3, 32, 32)
    d_hid = 5
    fs = FlowSequential([
        # Scale 1
        CouplingLayer([3,32,32], d_hid, mask='checker0', num_blocks=2),
        CouplingLayer([3,32,32], d_hid, mask='checker1', num_blocks=2),
        Squeeze(2),
        CouplingLayer([12,16,16], d_hid, mask='channel0', num_blocks=2),
        CouplingLayer([12,16,16], d_hid, mask='channel1', num_blocks=2),
        Factor(),

        # Scale 2
        CouplingLayer([6,16,16], d_hid, mask='checker0', num_blocks=2),
        CouplingLayer([6,16,16], d_hid, mask='checker1', num_blocks=2),
        Squeeze(2),
        CouplingLayer([24,8,8], d_hid, mask='channel0', num_blocks=2),
        CouplingLayer([24,8,8], d_hid, mask='channel1', num_blocks=2),
        Factor(),

        # Scale 3
        CouplingLayer([12,8,8], d_hid, mask='checker0', num_blocks=2),
        CouplingLayer([12,8,8], d_hid, mask='checker1', num_blocks=2),
    ])

    # xinv -> x -> xinv
    y, logdet, z = fs(x)
    xp, logdet, zz = fs(y, logdet, z, inverse=True)
    assert len(zz) == 0 and len(z) == 2
    assert x.shape == xp.shape
    assert y.shape == z[-1].shape == torch.Size([17,12,8,8])
    assert z[0].shape == torch.Size([17,6,16,16])
    assert torch.allclose(x, xp, atol=1e-5, rtol=1e-4)
    # assert logdet.shape == (17, 1)  # TODO: Implement inverse logdet for CouplingLayer

def test_sigmoid_logit():
    from scipy.special import expit, logit
    s, l = Sigmoid(), Logit(eps=0)
    x = torch.rand(1, 100) * 0.98 + 0.01
    out_s, _ = s(x)
    out_l, _ = l(x)
    assert allclose(expit(x), out_s)
    assert allclose(logit(x), out_l)
    assert allclose(x, s(out_l)[0])
    assert allclose(x, s.backward(out_s))
    assert allclose(x, l(out_s)[0])
    assert allclose(x, l.backward(out_l))

def test_scale():
    scale = torch.exp(torch.rand(1, 100) * 10 - 5)
    x = torch.rand(3, 100)
    flow = Scale(scale)
    out, logdet = flow(x)
    assert allclose(x, flow.backward(out))
    assert allclose(scale.log().sum(-1, keepdims=True), logdet)

if __name__ == '__main__':
    for _ in range(100):
        test_get_mask()
        test_coupling_layer('checker0')
        test_coupling_layer('checker1')
        test_coupling_layer('channel0')
        test_coupling_layer('channel1')
        test_squeeze()
        test_factor()
        test_flow_sequential()
        # test_sigmoid_logit()
        # test_scale()
    print('All `layers` tests passed!')
