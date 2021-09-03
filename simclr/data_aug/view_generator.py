import numpy as np
from torchvision import transforms 
np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        #self.identity_transform = transforms.ToTensor()
        self.identity_transform = transforms.Compose([transforms.Resize((256, 256)), #,interpolation=Image.NEAREST), 
        #[transforms.RandomResizedCrop(256), 
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.n_views = n_views

    def __call__(self, x):
        # the first transformation applies no augmentation (used for encoder4editing pipeline)
        #print('x ', x.size, self.identity_transform(x).size(), self.base_transform(x).size())
        return [self.identity_transform(x)] + [self.base_transform(x) for i in range(self.n_views-1)]

    #def __call__(self, x):
        # the first transformation applies no augmentation (used for encoder4editing pipeline)
        # print('x ', x.size, self.identity_transform(x).size(), self.base_transform(x).size())
        #return [self.base_transform(x) for i in range(self.n_views)]
