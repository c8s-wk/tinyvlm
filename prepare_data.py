from datasets import load_dataset
from transformers import BertTokenizer

def get_datasets(split_ratio=(0.8, 0.1, 0.1), seed=42):
    """
    Load Flickr8k dataset and split into train/val/test.
    :param split_ratio: tuple, e.g. (0.8, 0.1, 0.1)
    :param seed: random seed for reproducibility
    :return: train_ds, val_ds, test_ds
    """
    assert sum(split_ratio) == 1.0, "Split ratios must sum to 1."

    # Load the whole Flickr8k dataset
    ds = load_dataset("tsystems/flickr8k", split="train")

    ds = ds.shuffle(seed=seed)

    n_total = len(ds)
    n_train = int(split_ratio[0] * n_total)
    n_val   = int(split_ratio[1] * n_total)

    train_ds = ds.select(range(0, n_train))
    val_ds   = ds.select(range(n_train, n_train + n_val))
    test_ds  = ds.select(range(n_train + n_val, n_total))

    return train_ds, val_ds, test_ds


def get_tokenizer():
    """
    Load a pretrained BERT tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_datasets()
    tokenizer = get_tokenizer()

    print(f"Train size: {len(train_ds)}")
    print(f"Val size: {len(val_ds)}")
    print(f"Test size: {len(test_ds)}")

    # Show an example
    print(train_ds[0]["image"])
    print(train_ds[0]["caption"])
