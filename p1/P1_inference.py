import clip
import torch
import p1_parser
import os, sys
import pandas as pd
from PIL import Image
import json

args = p1_parser.arg_parse()
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

print("test image at : ", args.p1_test_dir)
print("test json at : ", args.p1_table)
print("result save at :", args.save_path)

# Download the dataset
classes = []
with open(os.path.join(args.p1_table)) as f:
    data = json.load(f)
    for i in range(len(data)):
        # print(data[str(i)])
        classes.append(data[str(i)])
# Prepare the inputs
# print(cifar100[3637], ":)")
filenames = []
labels = []
acc = 0
total = 0
for filename in os.listdir(os.path.join(args.p1_test_dir)):
    # refer to : https://pythonexamples.org/python-pillow-read-image/
    image = Image.open(os.path.join(args.p1_test_dir, filename))

    label = ""
    for i in range(len(filename)):
        if filename[i] != "_":
            label += filename[i]
        else:
            break


    # labels.append(label)
    # print(class_id)
    # print(cifar100.classes[77], cifar100.classes[78], cifar100.classes[79])
    image_input = preprocess(image).unsqueeze(0).to(device)
    # print(cifar100.classes)

    # text_inputs = torch.cat([clip.tokenize(f"No {c}, no score.") for c in classes]).to(device) # 0.5628
    # text_inputs = torch.cat([clip.tokenize(f"This is a photo of {c}") for c in classes]).to(device) # 0.6076
    text_inputs = torch.cat([clip.tokenize(f"This is a {c} image.") for c in classes]).to(device)  # 0.682

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)
    # Print the result
    # print("\nTop predictions:\n")
    filenames.append(filename)
    # refer to : https://www.runoob.com/python/python-func-zip.html
    # for value, index in zip(values, indices):
    #     print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
    # print("--------------------------------")
        # break
    labels.append(int(indices[0]))
    if int(indices[0]) == int(label):
        acc += 1
    total += 1

print("acc is : ", acc / total)
df = pd.DataFrame({'filename': filenames, 'label': labels})
df.to_csv(os.path.join(args.save_path), index=False)



