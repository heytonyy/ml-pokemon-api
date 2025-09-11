import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import vgg16, VGG16_Weights
from transformers import PretrainedConfig, PreTrainedModel
from PIL import Image
from tqdm import tqdm
import requests
from io import BytesIO
from typing import Dict, List, Optional, Union, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PokemonModel(nn.Module):
    """Core Pokemon classification model using VGG16 backbone."""
    
    def __init__(self, num_classes: int = 151):
        super(PokemonModel, self).__init__()
        self.vgg16 = vgg16(weights=VGG16_Weights.DEFAULT)
        num_ftrs = self.vgg16.classifier[-1].in_features
        self.vgg16.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vgg16(x)


class PokemonConfig(PretrainedConfig):
    """Configuration class for Hugging Face integration."""
    
    model_type = "pokemon_classifier"

    def __init__(self, num_labels: int = 151, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels


class HFPokemonModel(PreTrainedModel):
    """Hugging Face compatible wrapper for the Pokemon classifier."""
    
    config_class = PokemonConfig

    def __init__(self, config: PokemonConfig):
        super().__init__(config)
        self.pokemon_classifier = PokemonModel(num_classes=config.num_labels)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.pokemon_classifier(pixel_values)


class PokemonClassifier:
    """Main class for Pokemon classification with training, evaluation, and prediction capabilities."""
    
    def __init__(self, 
                 num_classes: int = 151,
                 device: Optional[str] = None,
                 model_path: Optional[str] = None,
                 class_mapping_path: Optional[str] = None):
        """
        Initialize the Pokemon classification model.
        
        Args:
            num_classes: Number of Pokemon classes (default: 151 for Gen 1)
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
            model_path: Path to pre-trained model weights
            class_mapping_path: Path to class mapping JSON file
        """
        self.num_classes = num_classes
        self.device = self._get_device(device)
        self.model = PokemonModel(num_classes=num_classes).to(self.device)
        
        # Initialize transforms
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()
        
        # Initialize class mapping
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Load pre-trained model and class mapping if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        if class_mapping_path and os.path.exists(class_mapping_path):
            self.load_class_mapping(class_mapping_path)
    
    def _get_device(self, device: Optional[str]) -> torch.device:
        """Get the appropriate device for computation."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        device_obj = torch.device(device)
        logger.info(f"Using device: {device_obj}")
        return device_obj
    
    def _get_train_transforms(self) -> transforms.Compose:
        """Get data transformations for training."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, fill=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_val_transforms(self) -> transforms.Compose:
        """Get data transformations for validation/inference."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_datasets(self, train_dir: str, val_dir: str, batch_size: int = 32, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
        """
        Load training and validation datasets.
        
        Args:
            train_dir: Path to training data directory
            val_dir: Path to validation data directory
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes for data loading
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_dataset = datasets.ImageFolder(train_dir, transform=self.train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=self.val_transform)
        
        # Store class mappings
        self.class_to_idx = train_dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        logger.info(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
        logger.info(f"Found {len(self.class_to_idx)} classes")
        
        return train_loader, val_loader
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 5,
              learning_rate: float = 0.001,
              momentum: float = 0.9,
              save_path: Optional[str] = None,
              save_best_only: bool = True) -> Dict[str, List[float]]:
        """
        Train the Pokemon classifier.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            momentum: Momentum for SGD optimizer
            save_path: Path to save the best model (optional)
            save_best_only: Whether to save only the best model based on validation accuracy
        
        Returns:
            Dictionary containing training history
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = running_loss / len(train_loader)
            
            # Validation phase
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Store history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {avg_train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val Accuracy: {val_acc:.2f}%")
            
            # Save best model
            if save_path and (not save_best_only or val_acc > best_val_acc):
                best_val_acc = val_acc
                self.save_model(save_path)
                logger.info(f"Model saved to {save_path}")
        
        return history
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model on validation data.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Tuple of (validation_loss, validation_accuracy)
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * correct / total
        
        return avg_val_loss, val_accuracy
    
    def predict(self, 
                image_input: Union[str, Image.Image],
                return_top_k: int = 1,
                return_probabilities: bool = False) -> Dict:
        """
        Predict Pokemon from an image.
        
        Args:
            image_input: Path to image file, URL, or PIL Image object
            return_top_k: Number of top predictions to return
            return_probabilities: Whether to return probability scores
        
        Returns:
            Dictionary containing prediction results
        """
        # Load and preprocess image
        if isinstance(image_input, str):
            if image_input.startswith(('http://', 'https://')):
                # URL input
                try:
                    response = requests.get(image_input)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                except Exception as e:
                    raise ValueError(f"Failed to load image from URL: {e}")
            else:
                # File path input
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            raise ValueError("image_input must be a file path, URL, or PIL Image object")
        
        # Transform image
        image_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, return_top_k)
        
        # Format results
        results = []
        for i in range(return_top_k):
            idx = int(top_indices[i].item())
            prob = float(top_probs[i].item())
            
            # Get class name if mapping is available
            class_name = self.idx_to_class.get(str(idx), f"Class_{idx}")
            
            result = {
                'class_id': idx,
                'class_name': class_name,
                'confidence': prob
            }
            
            if return_probabilities:
                result['probability'] = prob
            
            results.append(result)
        
        response = {
            'predictions': results,
            'top_prediction': results[0] if results else None
        }
        
        if return_probabilities:
            response['all_probabilities'] = probabilities.cpu().numpy().tolist()
        
        return response
    
    def save_model(self, save_path: str) -> None:
        """
        Save the model state dict.
        
        Args:
            save_path: Path to save the model
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load model state dict.
        
        Args:
            model_path: Path to the saved model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")
    
    def save_class_mapping(self, save_path: str) -> None:
        """
        Save class mapping to JSON file.
        
        Args:
            save_path: Path to save the class mapping
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.idx_to_class, f, indent=2)
        logger.info(f"Class mapping saved to {save_path}")
    
    def load_class_mapping(self, mapping_path: str) -> None:
        """
        Load class mapping from JSON file.
        
        Args:
            mapping_path: Path to the class mapping file
        """
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Class mapping file not found: {mapping_path}")
        
        with open(mapping_path, 'r') as f:
            self.idx_to_class = json.load(f)
        
        # Recreate class_to_idx mapping
        self.class_to_idx = {v: int(k) for k, v in self.idx_to_class.items()}
        logger.info(f"Class mapping loaded from {mapping_path}")
    
    def save_for_huggingface(self, save_directory: str) -> None:
        """
        Save model in Hugging Face format.
        
        Args:
            save_directory: Directory to save the Hugging Face model
        """
        config = PokemonConfig(num_labels=self.num_classes)
        hf_model = HFPokemonModel(config)
        hf_model.pokemon_classifier.load_state_dict(self.model.state_dict())
        
        os.makedirs(save_directory, exist_ok=True)
        hf_model.save_pretrained(save_directory)
        logger.info(f"Hugging Face model saved to {save_directory}")


# Convenience functions for quick usage
def load_pretrained_model(model_path: str, 
                         class_mapping_path: str,
                         device: Optional[str] = None) -> PokemonClassifier:
    """
    Load a pre-trained Pokemon classifier model.
    
    Args:
        model_path: Path to the saved model weights
        class_mapping_path: Path to the class mapping JSON file
        device: Device to run model on
    
    Returns:
        Loaded PokemonClassifier instance
    """
    model = PokemonClassifier(device=device)
    model.load_model(model_path)
    model.load_class_mapping(class_mapping_path)
    return model


def quick_predict(model_path: str,
                 class_mapping_path: str,
                 image_input: Union[str, Image.Image],
                 device: Optional[str] = None) -> str:
    """
    Quick prediction function for single image.
    
    Args:
        model_path: Path to the saved model weights
        class_mapping_path: Path to the class mapping JSON file
        image_input: Path to image file, URL, or PIL Image object
        device: Device to run model on
    
    Returns:
        Predicted Pokemon name
    """
    model = load_pretrained_model(model_path, class_mapping_path, device)
    result = model.predict(image_input)
    return result['top_prediction']['class_name']


if __name__ == "__main__":
    # Example usage
    print("Pokemon Classifier Model")
    print("========================")
    
    # Initialize model
    model = PokemonClassifier()
    
    # Example prediction (uncomment and modify paths as needed)
    # model.load_model("path/to/your/model.pth")
    # model.load_class_mapping("path/to/your/class_mapping.json")
    # result = model.predict("path/to/pokemon/image.jpg")
    # print(f"Predicted Pokemon: {result['top_prediction']['class_name']}")