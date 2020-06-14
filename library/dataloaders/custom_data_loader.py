from imports.imports_eva import *

class OfficeDataset(object):

    def resize_transforms (self, img):
      resize_transform = transforms.Compose([
        transforms.Resize((98, 98), 2),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

      return resize_transform (img)


    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "fg_bg_images"))))[0:1000]
        self.img_bgs = list(sorted(os.listdir(os.path.join(root, "bg_images_all"))))[0:1000]
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask_bg_images"))))[0:1000]

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "fg_bg_images", self.imgs[idx])
        bg_path = os.path.join(self.root, "bg_images_all", self.img_bgs[idx])
        mask_path = os.path.join(self.root, "mask_bg_images", self.masks[idx])

        img = Image.open(img_path)
        img_bg = Image.open(bg_path)
        mask = Image.open(mask_path)

        if self.transforms is not None:
            img = self.transforms(img)
            img_bg = self.transforms (img_bg)
            mask = self.resize_transforms (mask)

        img_data = {"Image": img, "Image_Bg": img_bg, "Mask": mask}

        return img_data

    def __len__(self):
        return len(self.imgs)
        
class custom_data_loader:
    def __init__():
        super().__init__()
    
    def get_def_data_transform():
        data_transform = transforms.Compose([
                                                    #transforms.Resize((64, 64), 2),
                                                    transforms.ToTensor()

                                                    
        ])
        return data_transform

    def custom_data_set (root_path, folder, transform=None):
        return OfficeDataset(root = root_path, transforms = transform)
                                           
    def custom_data_loader (dataset, batch_size, num_workers, shuffle=False):
        return torch.utils.data.DataLoader (dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)