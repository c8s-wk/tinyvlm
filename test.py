from datasets import load_dataset

ds = load_dataset("tsystems/flickr8k", split="train")
print(ds[0])
