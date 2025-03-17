import cv2
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

def polygons_to_segmentation_mask(polygons, image):
    for polygon, val in polygons:
        poly = np.array([polygon], dtype=np.int32)
        cv2.fillPoly(image, [poly], val)
    return image

class SupervisedDataset(Dataset):
    def __init__(self, data_path, dataframe, transforms):
        self.root_path = data_path
        self.transforms = transforms
        self.data = dataframe

        self.images = []
        self.targets = []
        self.vertex_boil = []
        self.vertex_pan = []
        self.nr_boild = []
        self.nr_pan = []
        self.id = []
        self.hows_polygon = []
        self.img_origin = []
        self.placement = []

        for idx in self.data.index:
            self.images.append(os.path.join(self.root_path, self.data['ID'].iloc[idx] + '.jpg'))
            self.vertex_boil.append(self.data['boil_polygon'].iloc[idx])
            self.vertex_pan.append(self.data['pan_polygon'].iloc[idx])
            self.nr_boild.append(self.data['boil_nbr'].iloc[idx])
            self.nr_pan.append(self.data['pan_nbr'].iloc[idx])
            self.id.append(self.data['ID'].iloc[idx])
            self.hows_polygon.append(self.data['hows_polygon'].iloc[idx])
            self.img_origin.append(self.data['img_origin'].iloc[idx])
            self.placement.append(self.data['placement'].iloc[idx])

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        size = img.size
        img = self.transforms(img)
        vertices_boil = self.vertex_boil[idx]
        vertex_pan = self.vertex_pan[idx]
        hows_polygon = self.hows_polygon[idx]

        mask = np.zeros((img.shape[1],img.shape[2]),dtype = np.int32)
        all_polygons = []
        polygons_boiler = []
        polygons_pan = []
        for polygon in vertices_boil:
            lst2 = []
            lst1 = []
            for i in range(len(polygon)):
                v1 = list(polygon[i])
                v1[0] = (v1[0]*512)/size[0]
                v1[1] = (v1[1]*512)/size[1]
                if hows_polygon == 'a esquerda':
                    v1[0] += 64
                    v1[1] += 64
                elif hows_polygon == 'a direita':
                    v1[0] -= 83
                    v1[1] -= 83
                lst2.append((v1[0],v1[1]))
            lst1.append(lst2)
            lst1.append(1) #boiler
            all_polygons.append(lst1)
            polygons_boiler.append(lst2)
        for polygon in vertex_pan:
            lst2 = []
            lst1 = []
            for i in range(len(polygon)):
                v1 = list(polygon[i])
                v1[0] = (v1[0]*512)/size[0]
                v1[1] = (v1[1]*512)/size[1]
                if hows_polygon == 'a esquerda':
                    v1[0] += 64
                    v1[1] += 64
                elif hows_polygon == 'a direita':
                    v1[0] -= 83
                    v1[1] -= 83
                lst2.append((v1[0],v1[1]))
            lst1.append(lst2)
            lst1.append(2) #pan
            all_polygons.append(lst1)
            polygons_pan.append(lst2)
        mask = polygons_to_segmentation_mask(all_polygons, mask)
        return img, mask.astype(int), polygons_pan, polygons_boiler, self.nr_boild[idx], self.nr_pan[idx], self.id[idx], self.img_origin[idx], self.placement[idx]