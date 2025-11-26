import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Sequence

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

    def __init__(
        self,
        img_dir,
        ann_dir,
        transforms=None,
        allowed_ids: Optional[Sequence[str]] = None,
        to_tensor: bool = True,
    ):
        """
        Args:
            img_dir (str): directory with images.
            ann_dir (str): directory with XML annotation files.
            transforms (callable, optional): function taking PIL image and
                target dict, and returning transformed (image, target).
            allowed_ids (Sequence[str], optional): restrict the dataset to the
                file stems present in this iterable.
            to_tensor (bool): if True and no custom transform is supplied, the
                PIL image is converted to a torch.FloatTensor in [0,1].
        """
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.transforms = transforms
        self.to_tensor = to_tensor
        self.allowed_ids = (
            {Path(item).stem for item in allowed_ids} if allowed_ids is not None else None
        )

        # collect all xml files
        self.ann_paths = sorted(self.ann_dir.glob("*.xml"))
        if self.allowed_ids:
            self.ann_paths = [path for path in self.ann_paths if path.stem in self.allowed_ids]
        if len(self.ann_paths) == 0:
            raise RuntimeError(f"No annotation files found in {self.ann_dir}")

        # one class: pothole -> label 1 (0 is reserved for background)
        self.class_to_idx = {"pothole": 1}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.ann_paths)

    def __getitem__(self, idx):
        ann_path = self.ann_paths[idx]

        # --- parse XML ---
        tree = ET.parse(str(ann_path))
        root = tree.getroot()

        filename_node = root.find("filename")
        filename = (
            filename_node.text if filename_node is not None else f"{Path(ann_path).stem}.jpg"
        )
        img_path = self.img_dir / filename

        # read image
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

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
            "image_path": str(img_path),
            "size": (width, height),
        }

        # default: convert PIL image to tensor [0,1]
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        elif self.to_tensor:
            img = F.to_tensor(img)

        return img, target


def collate_fn(batch):
    """
    Custom collate function for variable number of objects per image.
   """
    return tuple(zip(*batch))


# ---------- Example usage ----------

if __name__ == "__main__":

    base_dir = Path(r"/home/arian-sumak/Documents/DTU/computer vision/potholes_local_copy")
    img_dir = base_dir / "images"
    ann_dir = base_dir / "annotations"

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
