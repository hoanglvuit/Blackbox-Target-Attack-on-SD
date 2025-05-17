import torch

def CLIP_score(image, prompt, model, tokenizer, device):
    # Tokenize prompt và đưa lên GPU
    text = tokenizer([prompt]).to(device)

    # Tính đặc trưng và điểm tương đồng
    with torch.no_grad(), torch.autocast(device_type=device):
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # Chuẩn hóa đặc trưng
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Tính điểm tương đồng cosine
        similarity = (text_features @ image_features.T).cpu().item()

    return similarity
