import torch
import torch.utils.data as data
import cv2
import numpy as np
import os

class Dataset(data.Dataset):
    def __init__(self, dataset, subset, preproc=None):
        self.dataset = dataset
        self.imgs_path = []
        self.words = []
        self.subset = subset
        self.preproc = preproc
        txt_path = os.path.join("data", self.dataset, subset, 'label.txt')
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        self.img_names = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        for line in lines:
            if line.startswith("#"):
                line = line[2:-1]
                name = line
                self.img_names.append(name)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        img_name = self.img_names[index]

        if self.subset == 'train':
            annotations = np.zeros((0,15))
        else:
            annotations = np.zeros((0,16))

        if len(labels) == 0:
            return annotations

        for _, label in enumerate(labels):
            if self.subset == 'train':
                annotation = np.zeros((1, 15))
            else:
                annotation = np.zeros((1, 16))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            if len(label) >= 19: # or just sefl.dataset == 'widerface'
                # landmarks
                annotation[0, 4] = label[4]    # l0_x
                annotation[0, 5] = label[5]    # l0_y
                annotation[0, 6] = label[7]    # l1_x
                annotation[0, 7] = label[8]    # l1_y
                annotation[0, 8] = label[10]   # l2_x
                annotation[0, 9] = label[11]   # l2_y
                annotation[0, 10] = label[13]  # l3_x
                annotation[0, 11] = label[14]  # l3_y
                annotation[0, 12] = label[16]  # l4_x
                annotation[0, 13] = label[17]  # l4_y
            
            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            if self.subset == 'test' or self.subset == 'val':
                annotation[0, 15] = label[-1]

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        
        img, target = self.preproc(img, target, self.subset)

        return torch.from_numpy(img), target, img_name


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    img_names = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)
            elif isinstance(tup, str):
                img_names.append(tup)

    return (torch.stack(imgs, 0), targets, img_names)