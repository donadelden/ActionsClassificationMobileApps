import pandas as pd
import numpy as np
import tensorflow as tf

data_dir = "data"
data_file = "apps_total_plus_filtered"


def read_dataset(data_dir=data_dir, data_file=data_file):
    ds = pd.read_csv(f"{data_dir}/{data_file}.csv")
    sequence = ds["sequence"].map(lambda l: int(l[9:]))
    ds["sequence"] = sequence
    # packets_length_total column is in string format, converting to numpy array
    pkt_len = ds["packets_length_total"].map(
        lambda l: tuple(np.fromstring(l[1:-1], sep=", ", dtype=np.float))
    )
    ds["packets_length_total"] = pkt_len
    return ds


def dataset_mean_variance(data_dir=data_dir, data_file=data_file):
    ds = read_dataset(data_dir, data_file)
    # group together flows related to the same instance of action
    # and aggregate packet sequences
    ds = (
        ds[["app", "action", "sequence", "action_start", "packets_length_total"]]
        .groupby(
            ["app", "action", "sequence", "action_start"], sort=False, as_index=False
        )
        .agg(
            {
                "packets_length_total": lambda group: tuple(
                    np.concatenate(group.tolist())
                )
            }
        )
    )

    app = ds["app"]
    action = ds["action"]
    sequence = ds["sequence"]
    mean = (
        ds["packets_length_total"]
        .map(lambda l: np.mean(np.abs(l)))
        .rename("packets_length_mean")
    )
    variance = (
        ds["packets_length_total"]
        .map(lambda l: np.std(np.abs(l)))
        .rename("packets_length_std")
    )
    return pd.concat([app, action, sequence, mean, variance], axis=1)
