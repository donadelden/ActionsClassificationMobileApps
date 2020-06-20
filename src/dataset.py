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


def aggregate_flows_by_action(ds):
    ds = (
        ds[["app", "action", "sequence", "action_start", "packets_length_total"]]
        # group together flows related to the same instance of action
        .groupby(
            ["app", "action", "sequence", "action_start"], sort=False, as_index=False
        )
        # and aggregate packet sequences
        .agg(
            {
                "packets_length_total": lambda group: tuple(
                    np.concatenate(group.tolist())
                )
            }
        )
    )
    return ds


def dataset_mean_variance(data_dir=data_dir, data_file=data_file):
    ds = read_dataset(data_dir, data_file)
    ds = aggregate_flows_by_action(ds)

    app = ds["app"]
    action = ds["action"]
    sequence = ds["sequence"]
    mean = (
        ds["packets_length_total"]
        .map(np.abs)
        .map(np.mean)
        .rename("packets_length_mean")
    )
    variance = (
        ds["packets_length_total"].map(np.abs).map(np.std).rename("packets_length_std")
    )
    return pd.concat([app, action, sequence, mean, variance], axis=1)


def dataset_mean_variance_ingress(data_dir=data_dir, data_file=data_file):
    ds = read_dataset(data_dir, data_file)
    ds = aggregate_flows_by_action(ds)

    app = ds["app"]
    action = ds["action"]
    sequence = ds["sequence"]
    filtered = (
        ds["packets_length_total"]
        .map(np.array)
        .map(lambda l: l[l > 0])
        .where(lambda s: s.map(lambda l: l.size) > 0)
    )
    mean = filtered.map(np.mean).rename("packets_length_mean")
    variance = filtered.map(np.std).rename("packets_length_std")
    return pd.concat([app, action, sequence, mean, variance], axis=1)


def dataset_mean_variance_egress(data_dir=data_dir, data_file=data_file):
    ds = read_dataset(data_dir, data_file)
    ds = aggregate_flows_by_action(ds)

    app = ds["app"]
    action = ds["action"]
    sequence = ds["sequence"]
    filtered = (
        ds["packets_length_total"]
        .map(np.array)
        .map(lambda l: -l[l < 0])
        .where(lambda s: s.map(lambda l: l.size) > 0)
    )
    mean = filtered.map(np.mean).rename("packets_length_mean")
    variance = filtered.map(np.std).rename("packets_length_std")
    return pd.concat([app, action, sequence, mean, variance], axis=1)


def dataset_mean_variance2(data_dir=data_dir, data_file=data_file):
    """ mean and std for ingress and egress packets, total 4 features """
    ds = read_dataset(data_dir, data_file)
    ds = aggregate_flows_by_action(ds)

    app = ds["app"]
    action = ds["action"]
    sequence = ds["sequence"]
    filtered_ingress = (
        ds["packets_length_total"]
        .map(np.array)
        .map(lambda l: l[l > 0])
        .where(lambda s: s.map(lambda l: l.size) > 0)
    )
    mean_ingress = filtered_ingress.map(np.mean).rename("packets_length_mean_ingress")
    variance_ingress = filtered_ingress.map(np.std).rename("packets_length_std_ingress")
    filtered_egress = (
        ds["packets_length_total"]
        .map(np.array)
        .map(lambda l: -l[l < 0])
        .where(lambda s: s.map(lambda l: l.size) > 0)
    )
    mean_egress = filtered_egress.map(np.mean).rename("packets_length_mean_egress")
    variance_egress = filtered_egress.map(np.std).rename("packets_length_std_egress")
    return pd.concat(
        [
            app,
            action,
            sequence,
            mean_ingress,
            variance_ingress,
            mean_egress,
            variance_egress,
        ],
        axis=1,
    )
