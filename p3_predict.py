import torch
import numpy as np
from ./p2/Token.models.bert.tokenization_bert import BertTokenizer
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from ./p2/models import transformer
from ./p2/models import caption
from ./p2/datasets import coco, utils
from p3_configuration import Config
import cv2
import os

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--folder', type=str, default="../hw3_data/p3_data/images", help='path to images')
# parser.add_argument('--folder', type=str, default="./BestWorst", help='path to images')
parser.add_argument('--output', type=str, default="./resultOfHw3P3/", help='path to images')
parser.add_argument('--checkpoint_path', type=str, default="./hw3_p2_best.pth", help='checkpoint path')

args = parser.parse_args()
version = ''
config = Config()

print("Checking for checkpoint.")
if args.checkpoint_path is None:
    raise NotImplementedError('No model to chose from!')
else:
    if not os.path.exists(args.checkpoint_path):
        raise NotImplementedError('Give valid checkpoint path')
    print("Found checkpoint! Loading!")
    model, _ = caption.build_model(config)
    print("Loading Checkpoint...")
    checkpoint = torch.load(args.checkpoint_path, )
    model.load_state_dict(checkpoint['model'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device {}".format(device))
model = model.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


features_in_hook = []
features_out_hook = []

def get_attention(self, input, output):
        attn_output, attn_output_weights = output[0], output[1]
        features_in_hook.append(input)
        features_out_hook.append(output[1])


@torch.no_grad()
def evaluate():
    model.eval()
    caption_len = 0
    for i in range(config.max_position_embeddings - 1):    
        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)
    
        if predicted_id[0] == 102:
            print(caption_len)
            break
        caption_len += 1
        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    model.transformer.decoder.layers[5].multihead_attn.register_forward_hook(get_attention)
    output = model(image, caption, cap_mask)
    attn_map = get_attn_map(caption_len, features_out_hook[0])
    
    return caption, attn_map, caption_len

def get_attn_map(caption_len, attn_output_weights):
    return attn_output_weights[0, :caption_len, :]
    return attn_map

def visualize(orig_img, attn_map, caption, caption_len, filename, output_folder):
    attn_map = attn_map.cpu().numpy()
    captions = caption.split(' ')
    
    results = []
    results.append(orig_img)

    emb_h, emb_w = 19, 19
    for i in range(len(captions)+1):
        cur_map = attn_map[i]
        new_attn_map = cur_map.reshape(emb_h, emb_w)
        new_attn_map = cv2.resize(new_attn_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        final_map = np.stack((new_attn_map, new_attn_map, new_attn_map), axis=-1)
        final_map = normalize(final_map, orig_img.shape[0], orig_img.shape[1])
        final_map = final_map.astype(np.uint8)
        final_map = cv2.applyColorMap(final_map, cv2.COLORMAP_JET)
        result = cv2.addWeighted(orig_img, 0.3, final_map, 0.7, 0)
        cv2.imwrite("results/{}_{}.jpg".format(filename,i), result)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        results.append(result)
        # plt.imshow(result)
    captions.append("<EOS>")
    captions.insert(0, "<BOS>")
    
    assert len(captions) == len(results)
    
    rows, cols = int(np.ceil(len(results)/7)), 7
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(16,9))

    # set axis to off
    for row in range(rows):
        for col in range(cols):
            ax[row, col].axis("off")
    
    i = 0
    for row in range(rows):
        if i >= len(results):
            break
        for col in range(cols):
            if i < len(results):
                ax[row, col].imshow(results[i])
                ax[row, col].set_title(captions[i], fontsize=26)
                i += 1
            else:
                break
                
                
                
                
    name = "{}.png".format(filename)
    plt.savefig(os.path.join(output_folder, name))

    
        
def normalize(arr, h, w):
    new = np.zeros((h, w, 3))
    for j in range(3):
        channel = arr[:,:,j]
        a = 255 / (np.max(channel) - np.min(channel))
        b = -a * (np.min(channel))
        normalized = channel * a + b
        new[:,:, j] = normalized
    return new
    
if __name__ == '__main__':
    images = [file for file in os.listdir(args.folder)]
    
    for i in range(len(images)):
        image_path = os.path.join(args.folder, images[i])
        

        print("=============Reading image {}...=============".format(images[i]))
        image = Image.open(image_path)
        orig_img = np.array(image)
        orig_w, orig_h = image.size[0], image.size[1]
        image = coco.val_transform(image)
        image = image.unsqueeze(0)
        caption, cap_mask = create_caption_and_mask(
        start_token, config.max_position_embeddings)
        image, caption, cap_mask = image.to(device), caption.to(device), cap_mask.to(device)
        output, attn_map, caption_len = evaluate()
        
        result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        result = result.capitalize()
        visualize(orig_img=orig_img, attn_map=attn_map, caption=result, caption_len=caption_len, filename=images[i][:-4], output_folder=args.output)

