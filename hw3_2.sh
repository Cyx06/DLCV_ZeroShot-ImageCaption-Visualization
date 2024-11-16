# TODO: create shell script for running your ViT testing code

# Example
# wget -O hw3_p2_best.pth 'https://www.dropbox.com/s/dgc5ilfzo8rbayy/hw3_p2_best.pth?dl=0'
python3 ./p2/p2_inference.py --path=$1 --checkpoint=hw3_p2_best.pth --save_path=$2
