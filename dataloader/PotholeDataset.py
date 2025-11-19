import os
import glob
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as F


class PotholeDataset(Dataset):
    """
    Pothole object-detection dataset with Pascal VOC-style XML annotations.

    Each XML file looks like:
      <annotation>
        <filename>potholes0.png</filename>
        <object>
          <name>pothole</name>
          <bndbox>
            <xmin>...</xmin> ...
          </bndbox>
        </object>
        ...
      </annotation>
    """

    def __init__(self, img_dir, ann_dir, transforms=None):
        """
        Args:
            img_dir (str): directory with images.
            ann_dir (str): directory with XML annotation files.
            transforms (callable, optional): function taking PIL image and
                target dict, and returning transformed (image, target).
        """
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms

        # collect all xml files
        self.ann_paths = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))
        if len(self.ann_paths) == 0:
            raise RuntimeError(f"No annotation files found in {ann_dir}")

        # one class: pothole -> label 1 (0 is reserved for background)
        self.class_to_idx = {"pothole": 1}

    def __len__(self):
        return len(self.ann_paths)

    def __getitem__(self, idx):
        ann_path = self.ann_paths[idx]

        # --- parse XML ---
        tree = ET.parse(ann_path)
        root = tree.getroot()

        filename = root.find("filename").text
        img_path = os.path.join(self.img_dir, filename)

        # read image
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        difficults = []
        areas = []

        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in self.class_to_idx:
                # skip unknown classes (or raise, if you prefer)
                continue

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[name])

            difficult = obj.find("difficult")
            difficult = int(difficult.text) if difficult is not None else 0
            difficults.append(difficult)  # 0 or 1

            areas.append((xmax - xmin) * (ymax - ymin))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        difficults = torch.as_tensor(difficults, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)

        image_id = torch.tensor([idx])

        target = {
            "filename": filename,
            "boxes": boxes,          # (N, 4)
            "labels": labels,        # (N,)
            "image_id": image_id,
            "area": areas,           # (N,)
            "difficults": difficults,      # (N,)
        }

        # default: convert PIL image to tensor [0,1]
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = F.to_tensor(img)

        return img, target


def collate_fn(batch):
    """
    Custom collate function for variable number of objects per image.
   """
    return tuple(zip(*batch))


# ---------- Example usage ----------

if __name__ == "__main__":
    # Example directories (change to your real paths):
    img_dir = r"C:\Users\lucas\PycharmProjects\02516_Assignment_04\data\potholes\images"          # where potholes0.png lives
    ann_dir = r"C:\Users\lucas\PycharmProjects\02516_Assignment_04\data\potholes\annotations"          # where potholes0.xml lives

    dataset = PotholeDataset(img_dir=img_dir, ann_dir=ann_dir)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,     # set >0 if you want multiprocessing
        collate_fn=collate_fn,
    )

    # iterate once to see shapes
    for images, targets in data_loader:
        print(len(images), "images in batch")
        print("image[0] shape:", images[0].shape)    # C x H x W
        print("targets[0]:", targets[0])

        #plot image
        import matplotlib.pyplot as plt
        plt.imshow(images[0].permute(1, 2, 0))
        # plot bounding boxes
        for box in targets[0]["boxes"]:
            xmin, ymin, xmax, ymax = box.tolist()
            plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], "r")



        plt.show()
        break
