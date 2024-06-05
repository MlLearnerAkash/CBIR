import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image
import os
class ImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.transform = transforms.Compose(
    [
        # We first resize the input image to 256x256 and then we take center crop.
        
        transforms.Resize((256,256)),
         transforms.ToTensor()
                                    
    ]
)
        self.image_pairs = self.load_image_pairs()

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        image1_path, image2_path, label = self.image_pairs[idx]
        image1 = Image.open(image1_path).convert("RGB")
        image2 = Image.open(image2_path).convert("RGB")
        
    
        # Convert the tensor to a PIL image
        # image1 = functional.to_pil_image(image1)
        # image2 = functional.to_pil_image(image2)
        
        image1 = self.transform(image1)
        image2 = self.transform(image2)
        # image1 = torch.clamp(image1, 0, 1)
        # image2 = torch.clamp(image2, 0, 1)
        return image1, image2, label

    def load_image_pairs(self):
        image_pairs = []
        # Assume the directory structure is as follows:
        # root_dir
        # ├── similar
        # │   ├── similar_image1.jpg
        # │   ├── similar_image2.jpg
        # │   └── ...
        # └── dissimilar
        #     ├── dissimilar_image1.jpg
        #     ├── dissimilar_image2.jpg
        #     └── ...
        similar_dir = os.path.join(self.root_dir, "airplane")
        dissimilar_dir = os.path.join(self.root_dir, "bridge")

        # Load similar image pairs with label 0
        
        similar_images = os.listdir(similar_dir)
        for i in range(0, len(similar_images), 2):
            image1_path = os.path.join(similar_dir, similar_images[i])
            image2_path = os.path.join(similar_dir, similar_images[i+1])
            image_pairs.append((image1_path, image2_path, 0))

        # Load dissimilar image pairs with label 1
        # dissimilar_images = os.listdir(dissimilar_dir)
        # for i in range(0, len(dissimilar_images), 2):
        #     image1_path = os.path.join(dissimilar_dir, similar_images[i])
        #     image2_path = os.path.join(dissimilar_dir, similar_images[i+1])
        #     image_pairs.append((image1_path, image2_path, 1))

        return image_pairs
      
dataset = ImagePairDataset(r"/home/pravaig-20/Downloads/Assignment_CVML_02_04_24/Assignment/datasets/RESISC45_partial")
print("The dataset is:", dataset)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
