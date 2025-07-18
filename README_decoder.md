# Image Reconstruction Decoder for Federated Learning

This module provides a comprehensive decoder implementation for reconstructing original images from backbone outputs in federated learning scenarios.

## Features

- **Multi-Architecture Support**: Compatible with ResNet18, ResNet50, VGG11, VGG19, DenseNet121, and ViT-B/16 architectures
- **Flexible Cut Layers**: Supports different cut layers (0-4 for ResNet18/ResNet50/VGG11/VGG19/DenseNet121/ViT-B/16)
- **High-Quality Reconstruction**: Uses transposed convolutions with batch normalization
- **Quality Metrics**: Includes PSNR and SSIM calculations
- **Training Integration**: Seamlessly integrates with existing federated learning pipeline
- **Visualization**: Saves reconstructed images for visual inspection

## Quick Start

### 1. Basic Usage

```python
from architectures import Decoder, create_decoder_for_experiment

# Create decoder for DenseNet121 with cut layer 2
decoder = create_decoder_for_experiment(
    model_name='densenet121',  # or 'vit_b16' for Vision Transformer
    cut_layer=2,
    device='cuda'
)

# Forward pass
reconstructed_images = decoder(backbone_output)

# Calculate reconstruction loss
total_loss, mse_loss, l1_loss = decoder.compute_reconstruction_loss(reconstructed_images, original_images)
```

### 2. Complete Training Pipeline

```python
from architectures import train_federated_learning_with_decoder

# Run federated learning with image reconstruction
classification_losses, reconstruction_losses, accuracies = train_federated_learning_with_decoder(
    clients=clients,
    server=server,
    gated_fusion=gated_fusion,
    decoder=decoder,
    num_rounds=5,
    epochs_per_round=2,
    decoder_training_epochs=3,
    save_reconstructions=True,
    reconstruction_save_dir="./reconstruction_results"
)
```

### 3. Evaluation

```python
from architectures import evaluate_reconstruction_quality

# Evaluate reconstruction quality
results = evaluate_reconstruction_quality(
    decoder=decoder,
    test_dataloader=test_dataloader,
    server=server,
    clients=clients,
    device=device
)

print(f"PSNR: {results['psnr']:.2f} dB")
print(f"SSIM: {results['ssim']:.4f}")
```

## Architecture Details

### Decoder Structure

The decoder uses a series of transposed convolutions to progressively upsample the backbone output back to the original image size:

- **Input**: Backbone output (varies by cut layer)
- **Upsampling**: Transposed convolutions with stride=2
- **Normalization**: Batch normalization after each layer
- **Activation**: ReLU for hidden layers, Tanh for output
- **Output**: RGB images in [0, 1] range

### Supported Configurations

| Model | Cut Layer | Input Size | Output Size |
|-------|-----------|------------|-------------|
| ResNet18 | 0 | [B, 64, 112, 112] | [B, 3, 224, 224] |
| ResNet18 | 1 | [B, 64, 56, 56] | [B, 3, 224, 224] |
| ResNet18 | 2 | [B, 128, 28, 28] | [B, 3, 224, 224] |
| ResNet18 | 3 | [B, 256, 14, 14] | [B, 3, 224, 224] |
| ResNet18 | 4 | [B, 512, 7, 7] | [B, 3, 224, 224] |
| ResNet50 | 0 | [B, 64, 112, 112] | [B, 3, 224, 224] |
| ResNet50 | 1 | [B, 256, 56, 56] | [B, 3, 224, 224] |
| ResNet50 | 2 | [B, 512, 28, 28] | [B, 3, 224, 224] |
| ResNet50 | 3 | [B, 1024, 14, 14] | [B, 3, 224, 224] |
| ResNet50 | 4 | [B, 2048, 7, 7] | [B, 3, 224, 224] |
| VGG11 | 0 | [B, 64, 112, 112] | [B, 3, 224, 224] |
| VGG11 | 1 | [B, 128, 56, 56] | [B, 3, 224, 224] |
| VGG11 | 2 | [B, 256, 28, 28] | [B, 3, 224, 224] |
| VGG11 | 3 | [B, 512, 14, 14] | [B, 3, 224, 224] |
| VGG11 | 4 | [B, 512, 7, 7] | [B, 3, 224, 224] |
| VGG19 | 0 | [B, 64, 112, 112] | [B, 3, 224, 224] |
| VGG19 | 1 | [B, 128, 56, 56] | [B, 3, 224, 224] |
| VGG19 | 2 | [B, 256, 28, 28] | [B, 3, 224, 224] |
| VGG19 | 3 | [B, 512, 14, 14] | [B, 3, 224, 224] |
| VGG19 | 4 | [B, 512, 7, 7] | [B, 3, 224, 224] |
| DenseNet121 | 0 | [B, 64, 56, 56] | [B, 3, 224, 224] |
| DenseNet121 | 1 | [B, 128, 28, 28] | [B, 3, 224, 224] |
| DenseNet121 | 2 | [B, 256, 14, 14] | [B, 3, 224, 224] |
| DenseNet121 | 3 | [B, 512, 7, 7] | [B, 3, 224, 224] |
| DenseNet121 | 4 | [B, 1024, 7, 7] | [B, 3, 224, 224] |

## Loss Functions

The decoder uses a combination of losses for optimal reconstruction:

- **MSE Loss**: Pixel-wise mean squared error
- **L1 Loss**: Perceptual quality enhancement
- **Combined Loss**: `total_loss = mse_loss + 0.1 * l1_loss`

## Quality Metrics

- **PSNR**: Peak Signal-to-Noise Ratio in dB
- **SSIM**: Structural Similarity Index (simplified version)

## Example Results

Expected reconstruction quality (varies by cut layer):

| Cut Layer | PSNR (dB) | SSIM |
|-----------|-----------|------|
| 0 | 25-30 | 0.7-0.8 |
| 1 | 20-25 | 0.6-0.7 |
| 2 | 15-20 | 0.5-0.6 |
| 3 | 10-15 | 0.4-0.5 |
| 4 | 5-10 | 0.3-0.4 |

*Note: Earlier cut layers generally produce better reconstructions due to more spatial information.*

## Running the Example

```bash
# Run the complete example
python example_decoder_usage.py
```

This will:
1. Download CIFAR-10 dataset
2. Train federated learning with image reconstruction
3. Save reconstructed images to `./reconstruction_results/`
4. Generate training curves and evaluation metrics

## Key Components

### Decoder Class

```python
class Decoder(nn.Module):
    def __init__(self, model_name, cut_layer, input_size=(224, 224), device='cpu')
    def forward(self, backbone_output)
    def compute_reconstruction_loss(self, reconstructed, original)
    def train_step(self, backbone_output, original_images)
    def evaluate(self, backbone_output, original_images)
    def save_model(self) / load_model()
    def get_reconstruction_quality_metrics(self, reconstructed, original)
```

### Training Functions

- `train_federated_learning_with_decoder()`: Main training loop
- `train_decoder_reconstruction()`: Decoder-specific training
- `evaluate_reconstruction_quality()`: Evaluation on test set
- `save_reconstruction_samples()`: Save images for inspection

## Tips for Best Results

1. **Cut Layer Selection**: Earlier cut layers (0-2) generally produce better reconstructions
2. **Training Epochs**: Use more decoder training epochs for better quality
3. **Batch Size**: Larger batch sizes can improve training stability
4. **Learning Rate**: The default 0.001 works well, but can be tuned
5. **Loss Weight**: Adjust L1 loss weight (default 0.1) based on your needs

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use smaller cut layers
2. **Poor Reconstruction Quality**: Try earlier cut layers or more training epochs
3. **Model Not Loading**: Check checkpoint directory and file permissions

### Performance Tips

- Use GPU for faster training
- Enable mixed precision training for large models
- Use DataLoader with multiple workers for faster data loading

## Integration with Existing Code

The decoder integrates seamlessly with your existing federated learning pipeline:

1. Create decoder alongside other components
2. Use `train_federated_learning_with_decoder()` instead of regular training
3. Decoder training happens after each federated learning round
4. All models are saved/loaded automatically

## Future Enhancements

- Support for more architectures (MobileNet, EfficientNet)
- Advanced loss functions (perceptual loss, adversarial loss)
- Skip connections for better reconstruction
- Attention mechanisms for feature preservation
- Support for different image sizes and datasets

## Dependencies

```
torch >= 1.9.0
torchvision >= 0.10.0
matplotlib >= 3.3.0
numpy >= 1.20.0
```

## License

This module is part of the BDSL project and follows the same licensing terms. 