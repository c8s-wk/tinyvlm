# tinyvlm
Dual-encoder, shared embedding space, based on CLIP

## Set up environment + Install dependencies
### Create venv (optional)
```bash
python -m venv .venv

# Activate
.venv/Scripts/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

If using GPU, download PyTorch(GPU):
```bash
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

## Logic
### Dataset
- Uses Flickr8k from Hugging Face (datasets library).
- Each image comes with 5 human-written captions.
- Only a small subset (e.g., 2,000 samples) is used for quick training.

### Model
- Image Encoder: ResNet18 (pretrained on ImageNet, classification head removed).
- Text Encoder: BERT-base uncased, CLS token embedding as text representation.
- Projection layers: Linear mappings align both encoders into the same latent space.

### Loss function
Contrastive loss: Loss = average of image→text and text→image cross-entropy
