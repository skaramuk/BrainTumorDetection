import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Grad-CAM implementasyonu.
        Args:
            model: PyTorch modeli (ResNet18).
            target_layer: Gradient ve aktivasyonlarin alinmali hedeflenen katman.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook'lari kaydet
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate_cam(self, input_tensor, target_class=None):
        """
        Grad-CAM isisi (heatmap) uretir.
        Args:
            input_tensor: Modele verilecek olan (1, C, H, W) boyutundaki tensor.
            target_class: Hedef sinif indeksi. None ise en yuksek tahmin secilir.
        Returns:
            Normalize edilmis (0-1 arasi) 2D numpy matrisi (heatmap).
        """
        self.model.eval()
        
        # Gradient hesaplamasina ihtiyacimiz var, bu yuzden eval modunda bile enable ediyoruz
        with torch.enable_grad():
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
                
            self.model.zero_grad()
            
            # Sadece hedef sinifa gore geriye yayilim (backprop)
            one_hot = torch.zeros_like(output)
            one_hot[0][target_class] = 1
            output.backward(gradient=one_hot, retain_graph=True)
            
            # Global Average Pooling (kanal bazinda ortalama gradient)
            weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
            
            # Agirliklandirilmis aktivasyon haritalarinin toplami
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = F.relu(cam) # Sadece pozitif etkileri al
            
            cam = cam.squeeze().detach().cpu().numpy()
            
            # Normalize (0 ile 1 arasinda)
            cam = cam - np.min(cam)
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
                
            return cam

def apply_colormap_on_image(original_image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Uretilen Grad-CAM heatmap'ini orijinal goruntu uzerine bindirir (overlay).
    Args:
        original_image: PIL Image RGB.
        cam: 2D numpy array [0-1].
        alpha: Bindirme orani.
        colormap: OpenCV colormap.
    Returns:
        heatmap_img: Sadece isil harita goruntusu (PIL Image).
        overlay_img: Bindirilmis goruntu (PIL Image).
    """
    img_np = np.array(original_image)
    
    # Heatmap'i orijinal boyuta getir
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    
    # Heatmap renklendirme (OpenCV)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Bindirme (Overlay)
    overlay = np.uint8(alpha * heatmap + (1 - alpha) * img_np)
    
    return Image.fromarray(heatmap), Image.fromarray(overlay)
