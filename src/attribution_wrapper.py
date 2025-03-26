import torch
import numpy as np
from src.utils import attr_functions

class ContrastiveAttributionWrapper:
    """
    Wrapper for attribution methods that implements contrastive attribution.
    This wrapper forwards the input through the model, gets top classes,
    computes attributions for each, and applies contrastive weighting.
    """
    def __init__(self, base_method, model, device="cuda", tau=0.01, num_classes=2, 
                baseline=None, use_abs=True):
        """
        Initialize the contrastive attribution wrapper.
        
        Parameters:
            base_method: Base attribution method name (e.g., 'integrated_gradients')
            model: PyTorch model
            device: Device for computation
            tau: Threshold parameter for significance filtering
            num_classes: Number of top classes to consider
            baseline: Baseline tensor for attribution methods
            use_abs: Whether to use absolute values for attribution 
        """
        self.model = model
        self.device = device
        self.tau = tau
        self.num_classes = num_classes
        self.use_abs = use_abs
        self.baseline = baseline
        
        # Create base attribution function
        self.base_attribute_fn = attr_functions(
            base_method, 
            model,
            baseline=baseline
        )
    
    def attribute(self, img, target=None):
        """
        Compute contrastive attribution for target class.
        
        Parameters:
            img: Input image tensor (should include batch dimension)
            target: Optional target class index. If None, uses predicted class.
            
        Returns:
            contrastive_attr: Contrastive attribution tensor for target class
        """
        # Ensure input is on the correct device
        img = img.to(self.device)
        
        # Get model predictions if target not specified
        if target is None:
            with torch.no_grad():
                if "visiontransformer" in type(self.model).__name__.lower():
                    outputs = self.model(img,False)
                else:
                    outputs = self.model(img)
                probs = torch.softmax(outputs, dim=1)
                target = torch.argmax(probs, dim=1).item()
        
        # Get top-k classes 
        with torch.no_grad():
            if "visiontrans" in self.model.__class__.__name__.lower():
                outputs = self.model(img.to(self.device), False)
            else:
                outputs = self.model(img.to(self.device))
            probs = torch.softmax(outputs, dim=1).squeeze()
            
            # Get classes with probability at least 0.01
            #high_prob_indices = torch.where(probs >= 0.01)[0]
            
            # If less than 4 classes have prob >= 0.01, take the top-4 classes
            high_prob_indices = torch.topk(probs, k=min( self.num_classes, len(probs))).indices
            

            min_prob_idx = torch.argmin(probs).unsqueeze(0)
            high_prob_indices = torch.cat([high_prob_indices, min_prob_idx])

            if target not in high_prob_indices:
                high_prob_indices = torch.cat([torch.tensor([target], device=self.device), high_prob_indices[:3]])
            
            # Convert to list and limit to maximum 4 classes
            top_classes = high_prob_indices[: self.num_classes+1].tolist()
            # Convert to list (it's fine if there are fewer classes than num_classes)
            #top_classes = multi_labels
        
        # Compute attributions for each top class
        attributions = []
        for class_idx in top_classes:
            attribution = self.base_attribute_fn(img, target=class_idx).detach()
            attributions.append(attribution)
        
        # Stack attributions and compute significance mask
        stacked = torch.stack(attributions, dim=0)
        if self.use_abs:
            stacked = stacked.abs()
        
        # Apply softmax to get importance weighting across classes
        temperature = 10.0  # Controls sharpness of the softmax
        mask = torch.nn.functional.softmax(temperature * stacked, dim=0)

        try:
        # Find index of target class in top_classes
            target_idx = top_classes.index(target)
        except:
            print("index not there")
        
        # Compute contrastive attribution for target class
        # Apply threshold to filter out non-significant attributions
        contrastive_attr = attributions[target_idx] * mask[target_idx] * (
            (mask[target_idx] - (1.0 / len(top_classes))) > self.tau
        )
        return contrastive_attr
    
def normalize_channel_wise(image):
    # Assume image is tensor with shape [..., C] where C is number of channels
    channels = image.shape[-1]
    result = torch.zeros_like(image, dtype=torch.float)
    
    for c in range(channels):
        # Select this channel across all other dimensions
        channel = image[..., c]
        c_min, c_max = channel.min(), channel.max()
        if c_max > c_min:
            # Normalize and store back
            result[..., c] = 2 * (channel - c_min) / (c_max - c_min) - 1
    
    return result

class ClassSpecificAttributionWrapper:
    """
    Wrapper for attribution methods that implements contrastive attribution
    specifically for a provided set of classes.
    This wrapper computes attributions for each provided class and applies 
    contrastive weighting without automatically identifying top classes.
    """
    def __init__(self, base_method, model, device="cuda", tau=0.1, num_classes=4, 
                baseline=None, use_abs=True, class_indices=None):
        """
        Initialize the class-specific attribution wrapper.
        
        Parameters:
            base_method: Base attribution method name (e.g., 'integrated_gradients')
            model: PyTorch model
            device: Device for computation
            tau: Threshold parameter for significance filtering
            num_classes: Number of top classes to consider (not used but kept for API compatibility)
            baseline: Baseline tensor for attribution methods
            use_abs: Whether to use absolute values for attribution 
        """
        self.model = model
        self.device = device
        self.tau = tau
        self.num_classes = num_classes  # Kept for API compatibility but not used
        self.use_abs = use_abs
        self.baseline = baseline
        self.class_indices = class_indices  # Will be set during attribute method call
        
        # Create base attribution function
        self.base_attribute_fn = attr_functions(
            base_method, 
            model,
            baseline=baseline
        )
    
    def attribute(self, img, target=None):
        """
        Compute contrastive attribution for target class using only the provided classes.
        
        Parameters:
            img: Input image tensor (should include batch dimension)
            target: Optional target class index. If None, uses predicted class.
            class_indices: List of class indices to consider for contrastive attribution.
                           If None, will use previously set class_indices or throw an error.
            
        Returns:
            contrastive_attr: Contrastive attribution tensor for target class
        """
        # Ensure input is on the correct device
        img = img.to(self.device)
        

            
        # Check if we have class indices to work with
        if self.class_indices is None:
            raise ValueError("No class_indices provided. Set class_indices either during initialization or when calling attribute.")
        
        # Get model predictions if target not specified
        if target is None:
            with torch.no_grad():
                outputs = self.model(img)
                probs = torch.softmax(outputs, dim=1)
                target = torch.argmax(probs, dim=1).item()
        
        # Make sure target is in class_indices
        top_classes = self.class_indices.copy()
        #if target not in top_classes:
        #    top_classes = [target] + top_classes
        
        # Compute attributions for each provided class
        attributions = []
        normed_attributions = []
        for class_idx in top_classes:
            attribution = self.base_attribute_fn(img, target=class_idx).detach()
            #normed_attribution = normalize_channel_wise(attribution)
            attributions.append(attribution)
            #normed_attributions.append(normed_attribution)
        
        # Stack attributions and compute significance mask
        stacked = torch.stack(attributions, dim=0)
        if self.use_abs:
            stacked = stacked.abs()
        
        # Apply softmax to get importance weighting across classes
        temperature = 10.0  # Controls sharpness of the softmax
        mask = torch.nn.functional.softmax(temperature * stacked, dim=0)
        
        # Find index of target class in top_classes
        target_idx = top_classes.index(target)
        
        # Compute contrastive attribution for target class
        # Apply threshold to filter out non-significant attributions
        contrastive_attr = attributions[target_idx] * mask[target_idx] * (
            (mask[target_idx] - (1.0 / len(top_classes))) > self.tau
        )
        return contrastive_attr