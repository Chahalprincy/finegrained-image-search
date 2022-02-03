import argparse
import torch
import clip
import os
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--text")
parser.add_argument("--image_path",help = "path to image folder")
parser.add_argument("--output_path")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_name = os.listdir(args.image_path)
possible_file_ext = [".jpeg",".jpg",".png"]

image_file = []
for i in image_name:
    for k in possible_file_ext:
        if i.lower().endswith(k):
            image_file.append(i)

a = len(image_file)
b = torch.zeros(a,512)

text = clip.tokenize([args.text]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text)

j = 0
for i in tqdm(image_file):  
    image = preprocess(Image.open(args.image_path+"/"+i)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    b[j] = image_features
    j = j+1
            
image_features = b
image_features = image_features/image_features.norm(dim=-1,keepdim = True)
text_features = text_features/text_features.norm(dim=-1,keepdim = True)
text_features = torch.t(text_features)

final = torch.mm(image_features,text_features)
res = torch.topk(final.flatten(),5).indices


for i in res:
    iml = Image.open(args.image_path+"/"+image_file[i])
    iml = iml.save(args.output_path+"/"+image_file[i])

