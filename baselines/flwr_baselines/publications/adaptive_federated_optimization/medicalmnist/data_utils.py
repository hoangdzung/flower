from medmnist import *
import numpy as np

class WrapBloodMNIST(BloodMNIST):
    def __init__(self, **kwargs):
        super(WrapBloodMNIST, self).__init__(**kwargs)

    def __getitem__(self, idx):
        image, label = super(WrapBloodMNIST, self).__getitem__(idx)
        image, label = np.array(image), int(label[0])
        if len(image.shape) != 3:
            image = np.expand_dims(image, 0)
        return image, label

class WrapOrganSMNIST(OrganSMNIST):
    def __init__(self, **kwargs):
        super(WrapOrganSMNIST, self).__init__(**kwargs)

    def __getitem__(self, idx):
        image, label = super(WrapOrganSMNIST, self).__getitem__(idx)
        image, label = np.array(image), int(label[0])
        if len(image.shape) != 3:
            image = np.expand_dims(image, 0)
        return image, label


class WrapDermaMNIST(DermaMNIST):
    def __init__(self, **kwargs):
        super(WrapDermaMNIST, self).__init__(**kwargs)

    def __getitem__(self, idx):
        image, label = super(WrapDermaMNIST, self).__getitem__(idx)
        image, label = np.array(image), int(label[0])
        if len(image.shape) != 3:
            image = np.expand_dims(image, 0)
        return image, label

class WrapTissueMNIST(TissueMNIST):
    def __init__(self, **kwargs):
        super(WrapTissueMNIST, self).__init__(**kwargs)

    def __getitem__(self, idx):
        image, label = super(WrapTissueMNIST, self).__getitem__(idx)
        image, label = np.array(image), int(label[0])
        if len(image.shape) != 3:
            image = np.expand_dims(image, 0)
        return image, label
    
class WrapOCTMNIST(OCTMNIST):
    def __init__(self, **kwargs):
        super(WrapOCTMNIST, self).__init__(**kwargs)

    def __getitem__(self, idx):
        image, label = super(WrapOCTMNIST, self).__getitem__(idx)
        image, label = np.array(image), int(label[0])
        if len(image.shape) != 3:
            image = np.expand_dims(image, 0)
        return image, label
    
def get_dataclass(dataset_name):
    if dataset_name == 'bloodmnist':
        return WrapBloodMNIST
    if dataset_name == 'organsmnist':
        return WrapOrganSMNIST
    if dataset_name == 'dermamnist':
        return WrapDermaMNIST
    if dataset_name == 'tissuemnist':
        return WrapTissueMNIST
    if dataset_name == 'octmnist':
        return WrapOCTMNIST
    
    raise NotImplementedError