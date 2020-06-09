import torch
from torchvision import datasets, transforms


def mnist(seed=0, batch_size=1):
    torch.manual_seed(seed)

    # MNIST Test dataset and dataloader declaration
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=batch_size, shuffle=True)

    return test_loader
