import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from typing import Tuple, Optional, List, Dict
import csv
import random
#import ujson
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection

class ImageNetQuadDataset(Dataset):
    """Dataset for loading and combining 4 ImageNet validation images into one"""
    def __init__(
        self,
        root_dir: str,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        final_size: int = 224,
        seed: int = 42,
        normalize: bool = True,
        vit: bool = False
    ):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.final_size = final_size
        
        # Set the random seed
        self.seed = seed
        random.seed(self.seed)
        
        if vit:
            self.individual_transform = transforms.Compose([
                transforms.Resize((final_size // 2, final_size // 2)),  # Force exact size
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])
        # Define fixed transform to ensure consistent sizes
        elif normalize:
            self.individual_transform = transforms.Compose([
                transforms.Resize((final_size // 2, final_size // 2)),  # Force exact size
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.individual_transform = transforms.Compose([
                transforms.Resize((final_size // 2, final_size // 2)),  # Force exact size
                transforms.ToTensor()
            ])
        # Optional additional transforms
        self.extra_transform = transform
        
        # Get all image files
        self.image_files = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.JPEG')]
        self.num_quads = len(self.image_files) // 4
        self.quad_combinations = self._generate_fixed_combinations()

        # Load class mapping and validation labels
        self.class_map_file = os.path.join(root_dir, 'LOC_synset_mapping.txt')
        self.class_file = os.path.join(root_dir, 'LOC_val_solution.csv')
        self.validation_labels = {}
        self.idx_to_class = {}
        self.class_to_name = {}
        self.idx_to_name = {}
        
        self.load_class_map()
        self.load_validation_labels()

    def __len__(self) -> int:
        return self.num_quads
    
    def load_validation_labels(self) -> None:
        """Load validation labels from CSV file"""
        with open(self.class_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                image_id, prediction_string = row
                idx = prediction_string.split(' ')[0]
                if idx == "PredictionString":
                    continue
                self.validation_labels[image_id] = int(self.idx_to_class[idx])

    def load_class_map(self) -> Dict[str, str]:

        """
        Load mapping between synset IDs, class indices, and class names
        """
        with open(self.class_map_file, 'r') as f:
            for label, line in enumerate(f):
                class_id, class_name = line.strip().split(' ', 1)
                self.idx_to_class[class_id] = label
                self.class_to_name[label] = class_name
                self.idx_to_name[class_id] = class_name

    def _generate_fixed_combinations(self) -> List[List[int]]:
        """Generate fixed random combinations of image indices"""
        all_indices = list(range(len(self.image_files)))
        random.shuffle(all_indices)
        combinations = [
            all_indices[i:i + 4] 
            for i in range(0, len(all_indices) - len(all_indices) % 4, 4)
        ]
        return combinations
    
    def convert(self, label: int) -> str:
        """
        Convert class index to class name
        Args:
            label: Class index
        Returns:
            str: Class name
        """
        return self.class_to_name[label]
    
    def combine_images(self, images: List[torch.Tensor]) -> torch.Tensor:
        """Combine four images into a 2x2 grid"""
        # Ensure all images have the same size
        assert all(img.shape == images[0].shape for img in images), "Images must have the same shape"
        
        # Combine images
        top_row = torch.cat([images[0], images[1]], dim=2)  # Concatenate horizontally
        bottom_row = torch.cat([images[2], images[3]], dim=2)
        combined = torch.cat([top_row, bottom_row], dim=1)  # Concatenate vertically
        return combined

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[int]]:
        """
        Returns:
            tuple: (combined_image, labels) where combined_image is a tensor containing
                  four images in a 2x2 grid and labels is a list of their class indices
        """
        selected_indices = self.quad_combinations[idx]
        images = []
        labels = []
        
        for index in selected_indices:
            img_id = self.image_files[index]
            img_path = os.path.join(self.data_dir, img_id + '.JPEG')
            
            # Load and convert image
            img = Image.open(img_path).convert('RGB')
            label = self.validation_labels[img_id]
            
            # Apply the fixed transform to ensure consistent size
            img = self.individual_transform(img)
            
            # Apply any additional transforms if specified
            if self.extra_transform:
                img = self.extra_transform(img)
                
            images.append(img)
            labels.append(label)
        
        # Combine the four images
        combined_image = self.combine_images(images)
            
        return combined_image.unsqueeze(0), labels
    
class ImageNetValidationDataset(Dataset):
    """Dataset for loading ImageNet validation images"""
    def __init__(
        self,
        root_dir: str,
        data_dir: str,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            root_dir: Path to ImageNet dataset root containing metadata files
            data_dir: Path to validation images directory
            transform: Torchvision transforms to apply
        """
        self.root_dir = root_dir
        self.data_dir = data_dir
        
        # Set default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

        # Load class mapping and validation labels
        self.class_map_file = os.path.join(root_dir, 'LOC_synset_mapping.txt')
        self.class_file = os.path.join(root_dir, 'LOC_val_solution.csv')
        self.validation_labels = {}
        
        # Initialize mapping dictionaries
        self.idx_to_class = {}
        self.class_to_name = {}
        self.idx_to_name = {}
        
        # Get all image files from validation directory
        self.image_files = sorted([f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.JPEG')])
        
        self.load_class_map()
        self.load_validation_labels()

    def load_class_map(self) -> Dict[str, str]:
        """
        Load mapping between synset IDs, class indices, and class names
        """
        with open(self.class_map_file, 'r') as f:
            for label, line in enumerate(f):
                class_id, class_name = line.strip().split(' ', 1)
                self.idx_to_class[class_id] = label
                self.class_to_name[label] = class_name
                self.idx_to_name[class_id] = class_name

    def convert(self, label: int) -> str:
        """
        Convert class index to class name
        Args:
            label: Class index
        Returns:
            str: Class name
        """
        return self.class_to_name[label]

    def load_validation_labels(self) -> None:
        """Load validation labels from CSV file"""
        with open(self.class_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                image_id, prediction_string = row
                idx = prediction_string.split(' ')[0]
                if idx == "PredictionString":
                    continue
                self.validation_labels[image_id] = int(self.idx_to_class[idx])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            tuple: (image, label) where image is a transformed PIL Image and 
                  label is the class index
        """
        img_id = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_id + '.JPEG')
        
        # Load and convert image
        img = Image.open(img_path).convert('RGB')
        
        # Get label
        label = self.validation_labels[img_id]
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
            
        return img, label
  #4913  
class ImageNetPairDataset(Dataset):
    """Dataset for loading pairs of ImageNet images"""
    def __init__(
        self,
        root_dir: str,
        data_dir: str,
        image_pairs: List[Tuple[str, str]],
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            root_dir: Path to ImageNet dataset
            image_pairs: List of tuples containing pairs of image paths
            transform: Torchvision transforms to apply
        """
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.image_pairs = image_pairs
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        self.class_map_file = os.path.join(root_dir, 'LOC_synset_mapping.txt')
        self.class_file = os.path.join(root_dir, 'LOC_val_solution.csv')
        self.validation_labels = {}

        self.load_class_map()
        self.load_validation_labels()
    
    def apply_transform(self, img):
        return self.transform(img)
    
    def load_validation_labels(self) -> None:
        """Load validation labels from CSV file"""
        with open(self.class_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                image_id, prediction_string = row
                idx = prediction_string.split(' ')[0]
                if idx == "PredictionString":
                    continue
                self.validation_labels[image_id] = int(self.idx_to_class[idx])

    def __len__(self) -> int:
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img1_path = os.path.join(self.data_dir, self.image_pairs[idx][0] + ".JPEG")
        img2_path = os.path.join(self.data_dir, self.image_pairs[idx][1]+ ".JPEG")
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        label1 = self.validation_labels[self.image_pairs[idx][0]]
        label2 = self.validation_labels[self.image_pairs[idx][1]]
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, label1, label2
    
    def convert(self, label):
        return self.class_to_name[label].split(',')[0]  
    
    def load_class_map(self) -> Dict[str, str]:
        idx_to_class = {}
        class_to_name = {}
        idx_to_name = {}

        with open(self.class_map_file, 'r') as f:
            for label, line in enumerate(f):
                class_id, class_name = line.strip().split(' ', 1)
                idx_to_class[class_id] = label
                class_to_name[label] = class_name
                idx_to_name[class_id] = class_name

        self.idx_to_class = idx_to_class
        self.class_to_name = class_to_name
        self.idx_to_name = idx_to_name

def setup_model(device, model, pretrained=True) -> torch.nn.Module:
    if "bcos" in model.lower():
        model = torch.hub.load('B-cos/B-cos-v2', model.split("-")[1], pretrained=True).to("cuda")
    elif model == "vgg16":
        model = torchvision.models.vgg16(pretrained=pretrained).to(device)
    elif model == "densenet121":
        model = torchvision.models.densenet121(pretrained=pretrained).to(device)
    elif model == "resnext50_32x4d":
        model = torchvision.models.resnext50_32x4d(weights=torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1).to(device)
    elif model == "wide_resnet502":
        model = torchvision.models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device)
    elif model == "convnext":
        model = torchvision.models.convnext_base(pretrained=pretrained).to(device)
    elif model == "vit":
        from models.ViT.ViT_new import vit_base_patch16_224
        model = vit_base_patch16_224(pretrained=pretrained).to(device)
    else:
        model = torchvision.models.resnet50(pretrained=pretrained).to(device)
    model.eval()
    return model

def get_coco_dataset(root_dir='../data/coco', ann_file='annotations/instances_val2017.json', image_dir='val2017'):
    """
    Load the COCO dataset and return original image sizes.

    Args:
        root_dir: Root directory for COCO data
        ann_file: Annotation file path (relative to root_dir)
        image_dir: Directory containing images (relative to root_dir)
    """
    # Define transformations for inference
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize COCO API for annotations
    coco = COCO(os.path.join(root_dir, ann_file))

    # Custom dataset class to capture original image size
    class CustomCocoDataset(CocoDetection):
        def __getitem__(self, index):
            # Load image and annotations
            img, target = super().__getitem__(index)
            
            # Open the image again to get its original size
            img_path = self.coco.loadImgs(self.ids[index])[0]['file_name']
            orig_img = Image.open(os.path.join(self.root, img_path))
            orig_size = orig_img.size  # (width, height)
            
            return img, target, orig_size

    # Load dataset
    dataset = CustomCocoDataset(
        root=os.path.join(root_dir, image_dir),
        annFile=os.path.join(root_dir, ann_file),
        transform=transform
    )

    return dataset, coco

coco_to_imagenet = {
        1: 401,      # person -> person
        2: 444,      # bicycle -> bicycle
        3: 436,      # car -> passenger car
        4: 670,      # motorcycle -> motorcycle
        5: 404,      # airplane -> airliner
        6: 779,      # bus -> school bus
        7: 820,      # train -> passenger car
        8: 867,      # truck -> tractor-trailer
        9: 472,      # boat -> pontoon
        10: 920,     # traffic light -> traffic light
        13: 919,     # stop sign -> street sign
        14: 697,     # parking meter -> parking meter
        15: 703,     # bench -> park bench
        16: 13,      # bird -> common house bird
        17: 283,     # cat -> tabby cat
        18: 151,     # dog -> chihuahua (first dog class)
        19: 304,     # horse -> horse
        20: 349,     # sheep -> sheep
        21: 347,     # cow -> ox
        22: 386,     # elephant -> elephant
        23: 294,     # bear -> bear
        24: 340,     # zebra -> zebra
        25: 292,     # giraffe -> giraffe
        27: 414,     # backpack -> backpack
        28: 879,     # umbrella -> umbrella
        31: 617,     # handbag -> handbag
        32: 459,     # tie -> tie
        33: 485,     # suitcase -> suitcase
        34: 495,     # frisbee -> frisbee
        35: 795,     # skis -> ski
        36: 800,     # snowboard -> snowboard
        37: 722,     # sports ball -> basketball
        38: 829,     # kite -> kite
        39: 429,     # baseball bat -> baseball bat
        40: 582,     # baseball glove -> baseball glove
        41: 781,     # skateboard -> skateboard
        42: 776,     # surfboard -> surfboard
        43: 852,     # tennis racket -> tennis racket
        44: 898,     # bottle -> water bottle
        46: 918,     # wine glass -> wine glass
        47: 965,     # cup -> coffee mug
        48: 561,     # fork -> fork
        49: 625,     # knife -> knife
        50: 809,     # spoon -> spoon
        51: 927,     # bowl -> bowl
        52: 954,     # banana -> banana
        53: 948,     # apple -> apple
        54: 933,     # sandwich -> sandwich
        55: 950,     # orange -> orange
        56: 937,     # broccoli -> broccoli
        57: 924,     # carrot -> carrot
        58: 934,     # hot dog -> hot dog
        59: 963,     # pizza -> pizza
        60: 938,     # donut -> donut
        61: 925,     # cake -> cake
        62: 423,     # chair -> barber chair
        63: 678,     # couch -> couch
        64: 973,     # potted plant -> potted plant
        65: 424,     # bed -> bed
        67: 650,     # dining table -> dining table
        70: 861,     # toilet -> toilet
        72: 664,     # tv -> monitor
        73: 620,     # laptop -> laptop
        74: 673,     # mouse -> computer mouse
        75: 758,     # remote -> remote control
        76: 508,     # keyboard -> computer keyboard
        77: 487,     # cell phone -> cellular telephone
        78: 651,     # microwave -> microwave
        79: 685,     # oven -> oven
        80: 859,     # toaster -> toaster
        81: 747,     # sink -> sink
        82: 760,     # refrigerator -> refrigerator
        84: 521,     # book -> book
        85: 409,     # clock -> analog clock
        86: 883,     # vase -> vase
        87: 794,     # scissors -> scissors
        88: 850,     # teddy bear -> teddy bear
        89: 589,     # hair drier -> hair dryer
        90: 878,     # toothbrush -> toothbrush
    }

def filter_large_objects(annotations, orig_size, min_area_percentage=2.5):
    """
    Filter annotations to keep only objects that occupy at least a certain percentage of image area
    
    Parameters:
        annotations: List of annotation objects with 'bbox' field
        orig_size: Original image size (height, width)
        min_area_percentage: Minimum area threshold (default 2.5%)
    
    Returns:
        List of filtered annotations
    """
    # Get image dimensions
    height, width = orig_size
    image_area = height * width
    
    # Filter annotations
    filtered = []
    
    for anno in annotations:
        # Get bounding box
        x_min, y_min, box_width, box_height = anno["bbox"]
        box_area = box_width * box_height
        area_percentage = (box_area / image_area) * 100
        
        if area_percentage >= min_area_percentage:
            filtered.append(anno)
    
    return filtered

