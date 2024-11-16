# import time

import torch

# from Token.models.bert.tokenization_bert import BertTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE

from PIL import Image
import argparse

from models import caption
from datasets import coco, utils
from configuration import Config
import os

import json


parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--path', type=str, default="../coco/val2017/", help='path to image')
parser.add_argument('--checkpoint', type=str, default="./hw3_p2_best.pth", help='checkpoint path')
parser.add_argument('--save_path', type=str, default="demochange.json", help='save path')

args = parser.parse_args()
image_path = args.path
checkpoint_path = args.checkpoint
# print(checkpoint_path)
config = Config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Checking for checkpoint.")
if checkpoint_path is None:
    raise NotImplementedError('No model to chose from!')
else:
    if not os.path.exists(checkpoint_path):
        raise NotImplementedError('Give valid checkpoint path')
    print("Found checkpoint! Loading!")
    model, _ = caption.build_model(config)
    # print(model)
    print("Loading Checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

model = model.to(device)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
# tokenizer = Tokenizer(BPE())
# tokenizer.to(device)
start_token = tokenizer.encode("[CLS]")
end_token = tokenizer.encode("[SEP]")

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template



@torch.no_grad()
def evaluate():
    model.eval()
    for i in range(config.max_position_embeddings - 1):
        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102:
            return caption

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption
ans = {}
# start_time = time.time()



# count = 1
print("strat_caption")
for filename in os.listdir(image_path):
    # tokenizer = tokenizer.to(device)
    start_token = 101
    end_token = 102
    caption, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)

    target = os.path.join(image_path, filename)
    image = Image.open(target)
    # image = image.to(device)
    image = coco.val_transform(image).to(device)
    image = image.unsqueeze(0).to(device)
    image, caption, cap_mask = image.to(device), caption.to(device), cap_mask.to(device)
    output = evaluate().to(device)
    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    result = result.capitalize()
    #result = tokenizer.decode(output[0], skip_special_tokens=True)
    ans[filename[:len(filename) - 4]] = result
    if len(result) >= 150: result = result[:150]
    # count += 1
    # if count == 4: break
    # print(filename, result)
    # print(ans)
print("end_caption")
output_dir = os.path.join(args.save_path)
# if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

jsonFile = open(output_dir, 'w')
json.dump(ans, jsonFile)
# end_time = time.time()
# print("Excute time = %f sec." % (end_time - start_time))