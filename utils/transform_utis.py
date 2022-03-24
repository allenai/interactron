import models.detr_models.util.transforms as T
import torchvision.transforms as TV


transform = T.Compose([
    T.RandomResize([300], max_size=300),
    T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
])

train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomResize([400, 500, 600]),
    T.RandomSizeCrop(300, 300),
    T.RandomResize([300], max_size=300),
    T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
])


inv_transform = TV.Compose([
    TV.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
    TV.Normalize([-0.485, -0.456, -0.406], [1., 1., 1.,]),
    TV.ToPILImage()
])
