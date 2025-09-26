import torch
from torchvision import transforms
from prepare_data import get_datasets, get_tokenizer
from model import VLM
import numpy as np

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset (use test split) and tokenizer
    _, _, test_ds = get_datasets()
    tokenizer = get_tokenizer()
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    # Model
    model = VLM(embed_dim=128).to(device)
    model.load_state_dict(torch.load("files/vlm.pt", map_location=device))
    model.eval()

    # Encode embeddings
    img_embeds, txt_embeds, captions = [], [], []
    with torch.no_grad():
        for i in range(min(50, len(test_ds))):  # evaluate on first 50 samples
            image = transform(test_ds[i]["image"].convert("RGB")).unsqueeze(0).to(device)
            caption = test_ds[i]["captions"][0]
            enc = tokenizer([caption], return_tensors="pt", truncation=True, padding=True).to(device)

            img_emb, txt_emb = model(image, enc["input_ids"], enc["attention_mask"])
            img_embeds.append(img_emb.cpu().numpy())
            txt_embeds.append(txt_emb.cpu().numpy())
            captions.append(caption)

    img_embeds = np.vstack(img_embeds)
    txt_embeds = np.vstack(txt_embeds)

    # Retrieval demo: text → image
    query = "a dog is running in the grass"
    enc = tokenizer([query], return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        q_emb = model.text_encoder(**enc).pooler_output
        q_emb = model.txt_fc(q_emb)
        q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=-1).cpu().numpy()

    sims = img_embeds @ q_emb.T
    best_idx = np.argmax(sims)
    print("Query:", query)
    print("Retrieved Caption:", captions[best_idx])

if __name__ == "__main__":
    main()
