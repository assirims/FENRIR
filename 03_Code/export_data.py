"""Export the processed dataset, client partition and splits used in all experiments (04_Data)."""
import json, os
import numpy as np
import pandas as pd
import data

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "processed")
os.makedirs(OUT, exist_ok=True)


def main():
    fd = data.prepare(K=10, alpha=0.5, seed=42)
    split = np.array(["unused"] * len(fd.y), dtype=object)
    client = np.full(len(fd.y), -1)
    split[fd.test] = "test"
    for k, c in enumerate(fd.clients):
        split[c["tr"]] = "train"; client[c["tr"]] = k + 1
        split[c["va"]] = "val"; client[c["va"]] = k + 1
    df = pd.DataFrame(fd.X, columns=fd.features)
    df.insert(0, "label", [fd.classes[i] for i in fd.y])
    df.insert(1, "label_id", fd.y)
    df.insert(2, "split", split)
    df.insert(3, "client", client)
    df.to_csv(os.path.join(OUT, "rtiot2022_processed_all92_standardised.csv"), index=False)
    sel = json.load(open(os.path.join(HERE, "..", "results", "fs_selected.json")))["names"]
    df[["label", "label_id", "split", "client"] + sel].to_csv(
        os.path.join(OUT, "rtiot2022_processed_selected24.csv"), index=False)
    pd.DataFrame(fd.corr, index=fd.features, columns=fd.features).to_csv(os.path.join(OUT, "federated_correlation_matrix.csv"))
    summary = pd.crosstab(df["client"], df["label"])
    summary.to_csv(os.path.join(OUT, "client_partition_counts.csv"))
    print(df.shape, summary.shape)


if __name__ == "__main__":
    main()
