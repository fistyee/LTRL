# From: https://github.com/kaidic/LDAM-DRW
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

# --- #
from PIL import Image

import pickle
from collections import Counter
import os

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False, reverse=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor, reverse)
        self.gen_imbalanced_data(img_num_list)
        self.reverse = reverse

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, reverse):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                if reverse:
                    num =  img_max * (imb_factor**((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))                    
                else:
                    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        if self.train:
            return img, target, index
        else:
            return img, target



class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100



# class GENCIFAR10(IMBALANCECIFAR10):
#     cls_num = 10

#     def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
#                  transform=None, target_transform=None,
#                  download=False, reverse=False):
#         super(GENCIFAR10, self).__init__(root, train, transform, target_transform, download)
#         # if self.train:
#             # self.img_num_per_cls = self.get_img_num_per_cls(self.cls_num, imb_type, imbalance_ratio)
#             # self.gen_imbalanced_data(self.img_num_per_cls)

#         self.get_gen_data()
#             # self.transform = transforms.Compose([
#             #     # transforms.RandomCrop(32, padding=4),
#             #     transforms.Resize((224, 224)),
#             #     transforms.RandomHorizontalFlip(),
#             #     CIFAR10Policy(),    # add AutoAug
#             #     transforms.ToTensor(),
#             #     Cutout(n_holes=1, length=16),
#             #     transforms.Normalize(
#             #         (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#             # ])
#         # else:
#         #     self.transform = transforms.Compose([
#         #         transforms.Resize((224, 224)),
#         #         transforms.ToTensor(),
#         #         transforms.Normalize(
#         #             (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         #     ])
        

#         # self.labels = self.targets


#         # pdb.set_trace()
#         print("Contain {} images".format(len(self.data)))

#     def get_gen_data(self):
#         count = Counter(self.targets)

#         for digit, frequency in count.items():
#             print(f"Unbalanced Data: Digit {digit} appears {frequency} times.")

#         gen_data = []
#         gen_list = ['custom_data_batch_1.pkl', 'custom_data_batch_2.pkl', 'custom_data_batch_3.pkl', 
#                     'custom_data_batch_4.pkl', 'custom_data_batch_5.pkl', 'custom_data_batch_6.pkl']
#         # now load the picked numpy arrays
#         for file_name in gen_list:
#             file_path = os.path.join(self.root, file_name)
#             with open(file_path, "rb") as f:
#                 entry = pickle.load(f, encoding="latin1")
#                 gen_data.append(entry["data"])
#                 if "labels" in entry:
#                     self.targets.extend(entry["labels"])
#                 else:
#                     self.targets.extend(entry["fine_labels"])
        
#         gen_data = np.vstack(gen_data).reshape(-1, 3, 32, 32)
#         gen_data = gen_data.transpose((0, 2, 3, 1))
#         self.data = np.vstack((self.data, gen_data))
#         count = Counter(self.targets)
#         for digit, frequency in count.items():
#             print(f"Digit {digit} appears {frequency} times.")

#         print("Successful loading Generated Data ...")

class GENCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False, reverse=False):
        super(GENCIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor, reverse)
        self.gen_imbalanced_data(img_num_list)
        self.get_gen_data()
        self.reverse = reverse

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, reverse):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                if reverse:
                    num =  img_max * (imb_factor**((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))                    
                else:
                    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
    
    def get_gen_data(self):
        count = Counter(self.targets)

        for digit, frequency in count.items():
            print(f"Unbalanced Data: Digit {digit} appears {frequency} times.")

        gen_data = []
        gen_list = ['custom_data_batch_1.pkl', 'custom_data_batch_2.pkl', 'custom_data_batch_3.pkl', 
                    'custom_data_batch_4.pkl', 'custom_data_batch_5.pkl', 'custom_data_batch_6.pkl']
        # now load the picked numpy arrays
        for file_name in gen_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                gen_data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])
        
        gen_data = np.vstack(gen_data).reshape(-1, 3, 32, 32)
        gen_data = gen_data.transpose((0, 2, 3, 1))
        self.data = np.vstack((self.data, gen_data))
        count = Counter(self.targets)
        for digit, frequency in count.items():
            print(f"Digit {digit} appears {frequency} times.")

        print("Successful loading Generated Data ...")
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        if self.train:
            return img, target, index
        else:
            return img, target


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = IMBALANCECIFAR100(root='./data', train=True,
                    download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb; pdb.set_trace()