from torchvision import transforms

def train_transform(base_transform):

    return transforms.Compose([

        transforms.RandomResizedCrop(
            224,
            scale=(0.85, 1.0)
        ),

        transforms.RandomHorizontalFlip(p=0.5),

        base_transform
    ])