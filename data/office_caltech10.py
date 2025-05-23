from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class OfficeCaltechLoader(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.samples = [...]  # Implement based on dataset structure
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(os.path.join(self.root, path))
        return self.transform(image), label