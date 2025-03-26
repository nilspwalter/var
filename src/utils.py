import torch
import src.ig as ig
import torchvision
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import GuidedBackprop, GuidedGradCam,InputXGradient,GradientAttribution, LRP,LayerLRP,LayerGradCam
from functools import partial
import torch.functional as F
import cv2
import matplotlib.pyplot as plt

default_cmap = LinearSegmentedColormap.from_list('blue_red',
                                                 [(0, '#0000ff'),
                                                  (0.5, '#ffffff'),
                                                  (1, '#ff0000')], N=256)

def get_layer(model):
    if isinstance(model, torchvision.models.DenseNet):
        return model.features.denseblock4.denselayer16
    elif isinstance(model, torchvision.models.ResNet):
        return model.layer4[-1]
    elif isinstance(model, torchvision.models.VisionTransformer):
        return model.encoder.layers.encoder_layer_11
    elif isinstance(model, torchvision.models.VGG):
        return model.features[28]
    elif isinstance(model, torchvision.models.ConvNeXt):
        return model.features[7][2]
    else:
        raise ValueError("Model is not an instance of DenseNet121")
    
def rgba_to_rgb_batch_with_background(rgba_tensor, background_color=(1.0, 1.0, 1.0)):
    """
    Convert an RGBA tensor with shape (1, 4, 224, 224) to RGB with shape (1, 3, 224, 224)
    by compositing against a background color.
    
    Parameters:
        rgba_tensor: PyTorch tensor with shape (1, 4, 224, 224)
        background_color: RGB tuple for background color, default is white (255, 255, 255)
    
    Returns:
        rgb_tensor: PyTorch tensor with shape (1, 3, 224, 224)
    """
    # Extract RGB and alpha channels
    rgb = rgba_tensor[:, :3, :, :]
    alpha = rgba_tensor[:, 3:4, :, :] / 1.0
    
    # Create background tensor with same shape as RGB channels
    dtype = rgba_tensor.dtype
    device = rgba_tensor.device
    background = torch.tensor(background_color, dtype=dtype, device=device).reshape(1, 3, 1, 1)
    
    # Composite foreground and background using alpha blending
    rgb_tensor = alpha * rgb + (1 - alpha) * background
    
    return rgb_tensor

class VitGuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.handles = []
        
        # Register hooks on all GELU activations in MLP
        for block in model.blocks:
            # Hook for GELU activation in MLP
            self.handles.append(block.mlp.act.register_backward_hook(self._relu_hook))
            
            # Hooks for LayerNorm components
            self.handles.append(block.norm1.register_backward_hook(self._layernorm_hook))
            self.handles.append(block.norm2.register_backward_hook(self._layernorm_hook))
        
        # Hook for final LayerNorm
        self.handles.append(model.norm.register_backward_hook(self._layernorm_hook))
    
    def _relu_hook(self, module, grad_in, grad_out):
        # For guided backprop: zero out negative gradients (ReLU-like behavior)
        if isinstance(grad_in, tuple) and len(grad_in) > 0:
            return (torch.clamp(grad_in[0], min=0),)
    
    def _layernorm_hook(self, module, grad_in, grad_out):
        # For LayerNorm: preserve gradient flow but stabilize if needed
        return grad_in
    
    def attribute(self, input, target=None):
        # Ensure input requires gradients
        input_tensor = input.clone()
        input_tensor.requires_grad = True
        
        # Forward pass with hook registration enabled
        output = self.model(input_tensor.cuda(), register_hook=True)
        
        # Get target class
        if target is None:
            target = torch.argmax(output, dim=1).item()
        
        # Create one-hot encoding of target class
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        
        # Backward pass
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        # Return gradients as attribution
        return input_tensor.grad.clone()
    
    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class VitInputXGradient:
    def __init__(self, model):
        self.model = model
    
    def attribute(self, input, target=None):
        # Ensure input requires gradients
        input_tensor = input.clone()
        input_tensor.requires_grad = True
        
        # Forward pass with hook registration
        output = self.model(input_tensor.cuda(), register_hook=True)
        
        # Get target class
        if target is None:
            target = torch.argmax(output, dim=1).item()
        
        # Create one-hot encoding of target class
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        
        # Backward pass
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        # Input × Gradient
        return input_tensor * input_tensor.grad


def attr_functions(att, model=None, baseline=None):
    assert model is not None
    if baseline is not None:
        if "bcos" in type(model).__name__.lower():
            inner_baseline = torch.concatenate([baseline, 1-baseline], dim=1)
        else:
            inner_baseline = baseline
            
    # Check if model is the custom ViT
    is_vit = hasattr(model, 'blocks') and hasattr(model, 'cls_token')
    if att == "integrated_gradients":
        def attr(inp, target=0):
            return ig.IG(inp, model, 128, 64, 1, inner_baseline, "cuda", target).unsqueeze(0)
        return attr
        
    elif att == "guided_backprop":
        if is_vit:
            gb = VitGuidedBackprop(model)
            return gb.attribute
        else:
            gb = GuidedBackprop(model)
            return gb.attribute
            
            
    elif att == "input_x_gradient":
        if is_vit:
            ixg = VitInputXGradient(model)
            return ixg.attribute
        else:
            ixg = InputXGradient(model)
            return ixg.attribute
            
    elif att == "guided_grad_cam":
        if is_vit:
            # For ViT, we need to handle this specially
            # Reusing your GradCAM implementation logic
            class VitGuidedGradCam:
                def __init__(self, model):
                    self.model = model
                    self.gb = VitGuidedBackprop(model).attribute
                
                def __call__(self, input, target=None):
                    index = target
                    # First, compute GuidedBackprop attribution
                    gb_attr = self.gb(input, index)
                    
                    # Now compute Grad-CAM
                    input_tensor = input.clone()
                    output = self.model(input_tensor.cuda(), register_hook=True)
                    
                    if index is None:
                        index = np.argmax(output.cpu().data.numpy())
                    
                    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                    one_hot[0][index] = 1
                    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
                    one_hot = torch.sum(one_hot.cuda() * output)
                    
                    self.model.zero_grad()
                    one_hot.backward(retain_graph=True)
                    
                    # Get attention gradients and maps
                    grad = self.model.blocks[-1].attn.get_attn_gradients()
                    cam = self.model.blocks[-1].attn.get_attention_map()
                    
                    # Process attention maps as in your GradCAM implementation
                    cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
                    grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
                    grad = grad.mean(dim=[1, 2], keepdim=True)
                    cam = (cam * grad).mean(0).clamp(min=0)
                    cam = (cam - cam.min()) / (cam.max() - cam.min())
                    
                    # Resize CAM to input size
                    cam = cam.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                    cam = torch.nn.functional.interpolate(
                        cam, 
                        size=(input.shape[2], input.shape[3]),
                        mode='bilinear',
                        align_corners=False
                    )
                    cam = cam.squeeze(0)  # Remove batch dim
                    
                    # Guided GradCAM is GuidedBackprop * GradCAM
                    return gb_attr * cam
            
            ggc = VitGuidedGradCam(model)
            return ggc
        else:
            layer = get_layer(model)
            ggc = GuidedGradCam(model, layer)
            return ggc.attribute
            
    elif att == "gradcam":
        if is_vit:
            # Custom implementation for ViT using your existing GradCAM logic
            class VitGradCam:
                def __init__(self, model):
                    self.model = model
                
                def attribute(self, input, target=None):
                    input_tensor = input.clone()
                    output = self.model(input_tensor.cuda(), register_hook=True)
                    
                    if target is None:
                        target = torch.argmax(output, dim=1).item()
                    
                    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                    one_hot[0][target] = 1
                    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
                    one_hot = torch.sum(one_hot.cuda() * output)
                    
                    self.model.zero_grad()
                    one_hot.backward(retain_graph=True)
                    
                    # Get attention gradients and maps
                    grad = self.model.blocks[-1].attn.get_attn_gradients()
                    cam = self.model.blocks[-1].attn.get_attention_map()
                    cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
                    grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
                    grad = grad.mean(dim=[1, 2], keepdim=True)
                    cam = (cam * grad).mean(0).clamp(min=0)
                    cam = (cam - cam.min()) / (cam.max() - cam.min())
                    
                    # Resize to match input dimensions
                    # First convert to tensor with batch and channel dimensions
                    cam_tensor = torch.tensor(cam.detach().cpu().numpy(), device=input.device).unsqueeze(0).unsqueeze(0)
                    
                    # Use interpolate to resize
                    resized_cam = torch.nn.functional.interpolate(
                        cam_tensor, 
                        size=(input.shape[2], input.shape[3]),
                        mode='bilinear',
                        align_corners=False
                    )
        
                    # Remove batch dimension and return
                    return resized_cam#.squeeze(0).squeeze(0)
            
            gradcam = VitGradCam(model)
            return gradcam.attribute
        else:
  
            layer = get_layer(model)
            gradcam = LayerGradCam(model, layer)
            
            def attr(inp, target=None):
                cam = gradcam.attribute(inp, target)
                #upsampled_cam = LayerAttribution.interpolate(cam, inp.shape[-2:])
                upsampled_cam = F.interpolate(cam, 
                             size=inp.shape[-2:], 
                             mode='bilinear', 
                             align_corners=False)
                return upsampled_cam
                
            return attr
            
    elif att == "lrp":
        lrp = LRP(model)
        def attr(inp, target=0):
            return lrp.attribute(inp, target=target)
        return attr
        
    elif att == "bcos":
        assert "bcos" in type(model).__name__.lower(), "Only available for BcosModels"
        def attr(bcos_im, target):
            expl_out = model.explain(bcos_im, target)
            return expl_out["contribution_map"].unsqueeze(0)
        return attr
        
    else:
        raise ValueError(f"Invalid attribution method: {att}")
    
def normalize_for_display(attribution, percentile=99, use_abs=True, colormap="jet", smooth=False):
    """
    Normalize attribution maps for better visualization with specialized handling for different colormaps.
    
    Parameters:
        attribution: numpy array of attribution scores
        percentile: clip values above this percentile to reduce outlier influence (default: 99)
        use_abs: whether to take the absolute value (default: False)
        colormap: matplotlib colormap to apply
    
    Returns:
        normalized attribution map ready for display
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    attribution = attribution * (attribution > 0)
    # Reshape if needed (handle different input formats)
    if attribution.ndim == 3 and attribution.shape[2] == 3:
        # Handle RGB attributions by converting to grayscale
        if use_abs:
            attr = np.abs(attribution).mean(axis=2)
        else:
            attr = attribution.mean(axis=2)
    elif attribution.ndim == 3 and attribution.shape[0] == 3:
        # Handle channel-first format (convert to channel-last)
        transposed = np.transpose(attribution, (1, 2, 0))
        if use_abs:
            attr = np.abs(transposed).mean(axis=2)
        else:
            attr = transposed.mean(axis=2)
    else:
        # For single-channel attributions
        attr = np.abs(attribution) if use_abs else attribution
    
    # Clip outliers for better visualization
    if percentile < 100:
        if use_abs:
            # For absolute values, just clip the upper percentile
            vmax = np.percentile(attr, percentile)
            attr = np.clip(attr, a_min=attr.min(), a_max=vmax)
        else:
            # For signed values, apply symmetric percentile clipping
            pos_max = np.percentile(attr[attr > 0], percentile) if np.any(attr > 0) else 0
            neg_min = np.percentile(attr[attr < 0], 100-percentile) if np.any(attr < 0) else 0
            attr = np.clip(attr, a_min=neg_min, a_max=pos_max)
    
    # Apply different normalization strategies based on colormap type
    diverging_cmaps = ['RdBu', 'coolwarm', 'RdBu_r', 'bwr', 'seismic', 'PiYG', 'PRGn', 'BrBG']
    
    if not use_abs:
        if colormap == "jet":
            # For jet with signed values: Center at zero
            vmax = max(abs(attr.max()), abs(attr.min()))
            if vmax > 0:
                attr = (attr + vmax) / (2 * vmax)  # Map from [-vmax, vmax] to [0, 1]
        elif colormap in diverging_cmaps:
            # For diverging colormaps: Properly center at zero
            vmax = max(abs(attr.max()), abs(attr.min()))
            if vmax > 0:
                attr = (attr + vmax) / (2 * vmax)  # Map from [-vmax, vmax] to [0, 1]
        else:
            # Standard normalization for other colormaps
            if attr.max() > attr.min():
                attr = (attr - attr.min()) / (attr.max() - attr.min())
    else:
        # Standard normalization for absolute values
        if attr.max() > attr.min():
            attr = (attr - attr.min()) / (attr.max() - attr.min())
    #print(f"Normalized range: {attr.min():.4f} to {attr.max():.4f}")
    if len(attr.shape) == 3:
        attr = np.squeeze(attr)
    if smooth:
        attr = cv2.GaussianBlur(attr, (11, 11), 0)
        attr = (attr - attr.min()) / ((attr.max() - attr.min())+1e-8)
    # Apply colormap
    if colormap is not None:
        cmap = plt.get_cmap(colormap)
        attr = cmap(attr)
    return attr

def denormalize_image(image_tensor):
    """Denormalize image from ImageNet normalization"""
    # Convert from tensor to numpy if needed
    if hasattr(image_tensor, 'numpy'):
        image = image_tensor.numpy()
    else:
        image = image_tensor
        
    # Denormalize from ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    
    if image.shape[0] == 3 and len(image.shape) == 3:  # CHW format
        image = np.transpose(image, (1, 2, 0))
        
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    return image

def plot_pointing_game(img, attributions):
    fig, axes = plt.subplots(1,5,figsize=(20,4))
    axes[0].imshow(denormalize_image(img[0]))
    axes[0].axis("off")
    axes[0].set_title("Image", fontsize=18)#, fontweight='bold')


    for i, attr in enumerate(attributions):
        axes[i+1].imshow(attr)
        axes[i+1].axis("off")
        axes[i+1].set_title("Quadrant %d"%(i+1), fontsize=18)#, fontweight='bold')
