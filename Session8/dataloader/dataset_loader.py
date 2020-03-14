import torch
from torchvision import datasets, transforms

# function to load train and test data
def get_dataloader(is_train, cuda):
    if is_train:
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=16)

    dataset = datasets.CIFAR10(root='./data', train=is_train, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_args)
    return dataloader
