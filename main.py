import sys
from train import train
from inference import inference

if __name__ == '__main__':
    if not sys.argv[1] in ["train", "inference"]:
        raise Exception("Prompt Unknown")

    if sys.argv[1] == "train":
        train()

    if sys.argv[1] == "inference":
        inference()

   