import sklearn
import sklearn.datasets
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as trfm

def get_dataset(args):
    if args.data == 'mnist':
        if hasattr(args, 'n'):
            return MNIST(args.bs, device=args.device, n=args.n)
        else:
            return MNIST(args.bs, device=args.device)

    elif args.data == 'fmnist':
        if hasattr(args, 'n'):
            return FMNIST(args.bs, device=args.device, n=args.n)
        else:
            return FMNIST(args.bs, device=args.device)

    elif args.data == 'svhn':
        if hasattr(args, 'n'):
            return SVHN(args.bs, device=args.device, n=args.n)
        else:
            return SVHN(args.bs, device=args.device)

    elif args.data == 'celeba':
        if hasattr(args, 'n'):
            return CelebA(args.bs, device=args.device, n=args.n)
        else:
            return CelebA(args.bs, device=args.device)

    elif args.data == 'cos':
        x = np.random.rand(n) * 6 - 3
        y = np.sin(x) * 2.5
        data = np.stack((x, y), 1)
        data = torch.from_numpy(data.astype('float32'))
        raise ValueError(f'Unsupported dataset {args.data}')
    else:
        raise ValueError(f'Unsupported dataset {args.data}')

class MNIST:
    def __init__(self, bs, device=None, n=None, flatten=False):
        self.img_shape = [1, 28, 28]
        tr_data = datasets.MNIST(root='./data', download=True, train=True).data
        te_data = datasets.MNIST(root='./data', download=True, train=False).data

        # Flatten
        if flatten:
            tr_data = tr_data.reshape(-1, 784)
            te_data = te_data.reshape(-1, 784)
        else:
            tr_data = tr_data.view(-1, 1, 28, 28)
            te_data = te_data.view(-1, 1, 28, 28)

        # Move to target device
        tr_data = tr_data.to(device)
        te_data = te_data.to(device)

        if n:
            tr_data = tr_data[:n]

        self.train_data = tr_data
        self.test_data = te_data

        self.train_loader = DataLoader(self.train_data, batch_size=bs, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=bs, shuffle=False)

class FMNIST:
    def __init__(self, bs, device=None, n=None, flatten=False):
        self.img_shape = [1, 28, 28]
        tr_data = datasets.FashionMNIST(root='./data', download=True, train=True).data
        te_data = datasets.FashionMNIST(root='./data', download=True, train=False).data

        # Flatten
        if flatten:
            tr_data = tr_data.reshape(-1, 784)
            te_data = te_data.reshape(-1, 784)
        else:
            tr_data = tr_data.view(-1, 1, 28, 28)
            te_data = te_data.view(-1, 1, 28, 28)

        # Move to target device
        tr_data = tr_data.to(device)
        te_data = te_data.to(device)

        if n:
            tr_data = tr_data[:n]

        self.train_data = tr_data
        self.test_data = te_data

        self.train_loader = DataLoader(self.train_data, batch_size=bs, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=bs, shuffle=False)

class CelebA:
    def __init__(self, bs, device=None, n=None, flatten=False):
        self.img_shape = [3, 64, 64]
        transforms = trfm.Compose([  # Same preprocessing as RealNVP
            trfm.CenterCrop(148),
            trfm.Resize(64),
            trfm.ToTensor(),
        ])

        tr_data = datasets.CelebA(root='./data', split='train', transform=transforms)
        va_data = datasets.CelebA(root='./data', split='valid', transform=transforms)
        te_data = datasets.CelebA(root='./data', split='test', transform=transforms)

        self.train_data = tr_data
        self.valid_data = va_data
        self.test_data = te_data

        self.train_loader = DataLoader(self.train_data, batch_size=bs, shuffle=True, pin_memory=True, num_workers=4)
        self.valid_loader = DataLoader(self.valid_data, batch_size=bs, shuffle=False, pin_memory=True, num_workers=4)
        self.test_loader = DataLoader(self.test_data, batch_size=bs, shuffle=False, pin_memory=True, num_workers=4)

class SVHN:
    def __init__(self, bs, device=None, n=None, flatten=False, extra=True):
        self.img_shape = [3, 32, 32]

        tr_data = torch.from_numpy(datasets.SVHN(root='./data', split='train', download=True).data)
        te_data = torch.from_numpy(datasets.SVHN(root='./data', split='test', download=True).data)
        if extra:
            ex_data = torch.from_numpy(datasets.SVHN(root='./data', split='extra', download=True).data)
            tr_data = torch.cat([tr_data, ex_data], 0)

        # Flatten
        if flatten:
            tr_data = tr_data.reshape(-1, 3*32*32)
            te_data = te_data.reshape(-1, 3*32*32)
        else:
            tr_data = tr_data.view(-1, 3, 32, 32)
            te_data = te_data.view(-1, 3, 32, 32)

        # Move to target device
        tr_data = tr_data.to(device)
        te_data = te_data.to(device)

        if n:
            tr_data = tr_data[:n]

        self.train_data = tr_data
        self.test_data = te_data

        self.train_loader = DataLoader(self.train_data, batch_size=bs, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=bs, shuffle=False)
