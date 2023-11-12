import pickle, numpy as np, torch
from copy import deepcopy
from PIL import Image
import torchvision.transforms as transforms


class nuScenesDataset:
    def __init__(
            self, 
            imageset='train'):

        with open(imageset, 'rb') as f:
            data = pickle.load(f)
        self.nusc_infos = data['infos']
        self.transforms = transforms.Compose([
            transforms.Resize(512, interpolation=Image.BICUBIC)
        ])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)
    
    def __getitem__(self, index):
        info = deepcopy(self.nusc_infos[index])
        imgs_info = self.get_data_info(info)
        curr_imgs, _, ori_imgs = self.read_surround_imgs(imgs_info['img_filename'])
        return curr_imgs, ori_imgs, imgs_info
    
    def read_surround_imgs(self, img_paths):
        imgs = []
        ori_imgs = []
        img_sizes = []
        for filename in img_paths:
            ori_img = Image.open(filename).convert("RGB")
            width = ori_img.size[0]
            height = ori_img.size[1]
            img = self.transforms(ori_img)
            img = np.asarray(img)
            ori_img = torch.from_numpy(np.asarray(ori_img).copy())
            img = torch.from_numpy(img.copy()).permute(2, 0, 1)
            imgs.append(img)
            ori_imgs.append(ori_img)
            img_sizes.append([height, width])
        imgs = torch.stack(imgs, dim=0)
        ori_imgs = torch.stack(ori_imgs, dim=0)
        return imgs, img_sizes, ori_imgs

    def get_data_info(self, info):
        image_paths = []
        cam_types = []
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            cam_types.append(cam_type)

        input_dict =dict(
            img_filename=image_paths,
            cam_types=cam_types,
            token=info['token'])
        return input_dict

def custom_collate_fn(data):
    data_tuple = []
    for i, item in enumerate(data[0]):
        if isinstance(item, torch.Tensor):
            data_tuple.append(torch.stack([d[i] for d in data]))
        elif isinstance(item, (dict, str)):
            data_tuple.append([d[i] for d in data])
        elif item is None:
            data_tuple.append(None)
        else:
            raise NotImplementedError
    return data_tuple



if __name__ == '__main__':
    pass