import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *
from models.classifier import create_model
from utils.data_loader import get_transforms


class MedicalImagePredictor:
    """Inference class for medical image classification"""
    
    def __init__(self, dataset_type, model_name='efficientnet'):
        self.dataset_type = dataset_type
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = create_model(dataset_type, model_name)
        checkpoint_path = MODEL_DIR / f"{dataset_type}_{model_name}_best.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get transform
        self.transform = get_transforms(augment=False)
        
        # Get class names
        if dataset_type == 'aptos':
            self.class_names = APTOS_CLASSES
        elif dataset_type == 'ham10000':
            self.class_names = HAM10000_CLASSES
        elif dataset_type == 'mura':
            self.class_names = MURA_CLASSES
    
    def predict(self, image_path):
        """
        Make prediction on a single image
        
        Returns:
            dict: {
                'predicted_class': str,
                'confidence': float,
                'all_probabilities': dict
            }
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
        
        # Get predicted class
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
        
        # Map to class name
        if self.dataset_type == 'ham10000':
            class_keys = list(self.class_names.keys())
            predicted_class = self.class_names[class_keys[predicted_idx]]
        else:
            predicted_class = self.class_names[predicted_idx]
        
        # Get all probabilities
        all_probs = {}
        if self.dataset_type == 'ham10000':
            class_keys = list(self.class_names.keys())
            for idx, prob in enumerate(probabilities.cpu().numpy()):
                all_probs[self.class_names[class_keys[idx]]] = float(prob)
        else:
            for idx, prob in enumerate(probabilities.cpu().numpy()):
                all_probs[self.class_names[idx]] = float(prob)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probs
        }
    
    def generate_gradcam(self, image_path, target_layer=None):
        """Generate Grad-CAM visualization"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get feature maps and prediction
        feature_maps = self.model.get_feature_maps(image_tensor)
        outputs = self.model(image_tensor)
        
        # Get predicted class
        predicted_idx = torch.argmax(outputs).item()
        
        # Compute gradients
        self.model.zero_grad()
        outputs[0, predicted_idx].backward()
        
        # Get gradients and compute weights
        gradients = feature_maps.grad
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Compute Grad-CAM
        cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Apply colormap
        cam_colored = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        original_image = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
        overlayed = cv2.addWeighted(original_image, 0.6, cam_colored, 0.4, 0)
        
        return overlayed


def load_predictor(dataset_type, model_name='efficientnet'):
    """Load predictor for specific dataset"""
    return MedicalImagePredictor(dataset_type, model_name)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test predictor
    print("Testing predictor...")
    
    # Example: Test APTOS
    try:
        predictor = load_predictor('aptos')
        # Use a sample image from validation set
        sample_image = list((APTOS_DIR / 'val_images').glob('*.png'))[0]
        result = predictor.predict(sample_image)
        
        print(f"\nPrediction for {sample_image.name}:")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nAll Probabilities:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name}: {prob:.2%}")
    except Exception as e:
        print(f"Error: {e}")
