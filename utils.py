# utils.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir='data', batch_size=32):
    """Retourne les DataLoaders pour train, val et test"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Charger les datasets
    train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=transform)
    val_dataset = datasets.ImageFolder(root=f'{data_dir}/val', transform=transform)
    test_dataset = datasets.ImageFolder(root=f'{data_dir}/test', transform=transform)

    # Créer les DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

