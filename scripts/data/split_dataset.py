import json
import os.path
import random
import pandas as pd

train_percentage = 0.7
dataset_root = "/home/gr-agv-lx91/Downloads/pretraining_dataset"
scenes = ["hospital", "office"]


def split():
    dataset = []
    for scene in scenes:
        path = f"{dataset_root}/{scene}/dataset_info.json"
        with open(path, "r") as f:
            data_index = json.load(f)
            assert isinstance(data_index, dict)
            for i, trajectory in enumerate(data_index.values()):
                for j in range(len(trajectory["data"])):
                    dataset.append((scene, i, j))
    random.shuffle(dataset)
    train_data = dataset[:int(train_percentage * len(dataset))]
    eval_data = dataset[int(train_percentage * len(dataset)):]
    train_df = pd.DataFrame(train_data, columns=["scene", "trajectory", "step"])
    eval_df = pd.DataFrame(eval_data, columns=["scene", "trajectory", "step"])
    train_df.to_csv(os.path.join(dataset_root, "train_dataset.csv"), index=False)
    eval_df.to_csv(os.path.join(dataset_root, "eval_dataset.csv"), index=False)


if __name__ == "__main__":
    split()
