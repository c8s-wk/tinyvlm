import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from prepare_data import get_datasets, get_tokenizer
from model import VLM

def collate_fn(batch, tokenizer, transform):
    images, captions = [], []
    for b in batch:
        images.append(transform(b["image"].convert("RGB")))
        captions.append(b["captions"][0])  # pick the first caption

    enc = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
    return torch.stack(images), enc["input_ids"], enc["attention_mask"]

def contrastive_loss(img_emb, txt_emb, temperature=0.07):
    logits = img_emb @ txt_emb.T / temperature
    labels = torch.arange(len(logits)).to(logits.device)
    loss_i = nn.CrossEntropyLoss()(logits, labels)       # image → text
    loss_t = nn.CrossEntropyLoss()(logits.T, labels)     # text → image
    return (loss_i + loss_t) / 2

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset and tokenizer
    train_ds, val_ds, _ = get_datasets()
    tokenizer = get_tokenizer()
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer, transform))
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, tokenizer, transform))

    # Model
    model = VLM(embed_dim=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Training
    model.train()
    for epoch in range(5):
        total_loss = 0
        for images, input_ids, attention_mask in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, input_ids, attention_mask = images.to(device), input_ids.to(device), attention_mask.to(device)

            img_emb, txt_emb = model(images, input_ids, attention_mask)
            loss = contrastive_loss(img_emb, txt_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Train] Epoch {epoch+1}, Loss={avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, input_ids, attention_mask in val_loader:
                images, input_ids, attention_mask = images.to(device), input_ids.to(device), attention_mask.to(device)
                img_emb, txt_emb = model(images, input_ids, attention_mask)
                val_loss += contrastive_loss(img_emb, txt_emb).item()
        print(f"[Val]   Epoch {epoch+1}, Loss={val_loss/len(val_loader):.4f}")
        model.train()

    torch.save(model.state_dict(), "files/vlm.pt")
    print("Model saved to files/vlm.pt")

if __name__ == "__main__":
    main()
