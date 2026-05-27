from torchvision import transforms

def train_transform(base_transform):

    return transforms.Compose([

        transforms.RandomResizedCrop(
            224,
            scale=(0.85, 1.0)
        ),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            )
        ], p=0.3),

        base_transform
    ])