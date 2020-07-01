import pandas as pd
import numpy as np

data_dir = "data"
data_file = "apps_total_plus_filtered"


def read_dataset(data_dir=data_dir, data_file=data_file):
    ds = pd.read_csv(f"{data_dir}/{data_file}.csv", index_col="flow_number")
    sequence = ds["sequence"].map(lambda l: int(l[9:]))
    ds["sequence"] = sequence.ne(sequence.shift()).cumsum()
    # packets_length_total column is in string format, converting to numpy array
    pkt_len = ds["packets_length_total"].map(
        lambda l: tuple(np.fromstring(l[1:-1], sep=", ", dtype=np.float))
    )
    ds["packet_length"] = pkt_len
    ds["app"] = ds["app"].astype("category")
    ds["action"] = ds["action"].astype("category")
    ds = ds.drop(columns="packets_length_total").explode("packet_length").reset_index()
    return ds


def dataset_mean_variance(
    data_dir=data_dir,
    data_file=data_file,
    agg_by="sequence",
    filter=None,
    na=None,
    na_value=0,
):
    """ Get a mean/variance dataset in a Pandas DataFrame
    Parameters
    ----------
    data_dir : str, optional
        relative directory path (from project root) to dataset location
    
    data_file : str, optional
        dataset (csv) filename excl. extension

    agg_by : str, optional
        parameter by which aggregate flows, can be sequence or action

    filter : str, optional
        filter packets in the flow, by default (`None`) returns the grand total, other options are `ingress`, `egress`, `both` (4 features)

    na : str, optional
        how to treat NA/NaN values, by default (`None`) NA/NaN are left, `fill` fills with value provided in `na_value`, `drop` drops rows with any NA/NaN
    
    na_value : number, optional
        only if `na` is `fill`, value the dataset is filled with
    
    Returns
    -------
    ds
        the dataset in a Pandas DataFrame
    """
    ds = read_dataset(data_dir=data_dir, data_file=data_file)

    if agg_by == "sequence":
        grouped = ds[["app", "sequence", "packet_length"]].groupby(["app", "sequence"])
    elif agg_by == "action":
        grouped = (
            ds[["app", "action", "sequence", "action_start", "packet_length"]]
            # group together flows related to the same instance of action
            .groupby(["app", "action", "sequence", "action_start"])
        )
    else:
        raise ValueError(f"Can only aggregate by action or sequence, not {agg_by}")

    app = ds["app"]
    action = ds["action"]
    # sequence = ds["sequence"]

    if filter is None:
        ds = grouped["packet_length"].agg(
            packets_length_mean=lambda l: np.mean(np.abs(l)),
            packets_length_std=lambda l: np.std(np.abs(l)),
        )
    elif filter == "ingress":
        ds = grouped["packet_length"].agg(
            packets_length_mean=lambda l: np.mean(-l[l < 0]),
            packets_length_std=lambda l: np.std(-l[l < 0]),
        )
    elif filter == "egress":
        ds = grouped["packet_length"].agg(
            packets_length_mean=lambda l: np.mean(l[l > 0]),
            packets_length_std=lambda l: np.std(l[l > 0]),
        )
    elif filter == "both":
        ds = grouped["packet_length"].agg(
            ingress_packets_length_mean=lambda l: np.mean(-l[l < 0]),
            ingress_packets_length_std=lambda l: np.std(-l[l < 0]),
            egress_packets_length_mean=lambda l: np.mean(l[l > 0]),
            egress_packets_length_std=lambda l: np.std(l[l > 0]),
        )
    else:
        raise ValueError(f"Cannot filter by {filter}")

    ds = ds.reset_index()

    if na == "fill":
        ds.fillna(value=na_value, inplace=True)
    elif na == "drop":
        ds.dropna(inplace=True)
    elif na != None:
        raise ValueError(f"cannot use {na} method to treat NA/NaN")

    if agg_by == "action":
        ds = ds.reset_index().drop(columns="action_start")

    ds.drop(columns="sequence")

    return ds


def dataset_windowed_random(
    data_dir=data_dir,
    data_file=data_file,
    agg_by="sequence",
    N=10000,
    K=100,
    filter=None,
    random_state=None,
):
    """ Generate dataset of N arrays with K subsequent packet lengths.

    The original dataset is aggregated by sequence or action, then randomly sampled with replacement, and cropped randomly to match the desired length.

    Parameters
    ----------
    data_dir : str, optional
        relative directory path (from project root) to dataset location
    
    data_file : str, optional
        dataset (csv) filename excl. extension

    agg_by : str, optional
        parameter by which aggregate flows, can be sequence or action

    N : int
        the number of output samples

    K : int
        the size of each sample, number of subsequent packet lengths

    filter : str, optional
        filter by `"ingress"`, `"egress"` or `None` (default)

    random_state : int
        initial seed for the RNG

    Returns
    -------
    ds
        the dataset
    """
    orig = read_dataset(data_dir=data_dir, data_file=data_file)

    if agg_by == "sequence":
        aggregated = aggregate_flows_by_sequence(orig)
    elif agg_by == "action":
        aggregated = aggregate_flows_by_action(orig)
    else:
        raise ValueError(f"cannot aggregate by {agg_by}")

    if filter == "ingress":
        aggregated["packets_length_total"] = (
            aggregated["packets_length_total"]
            .map(np.array)
            .map(lambda l: l[l > 0])
            .where(lambda s: s.map(lambda l: l.size) > 0)
            .dropna()
            .map(tuple)
        )
    elif filter == "egress":
        aggregated["packets_length_total"] = (
            aggregated["packets_length_total"]
            .map(np.array)
            .map(lambda l: -l[l < 0])
            .where(lambda s: s.map(lambda l: l.size) > 0)
            .dropna()
            .map(tuple)
        )
    elif filter != None:
        raise ValueError(f"cannot filter by {filter}")

    aggregated = aggregated.mask(
        lambda l: l["packets_length_total"].map(len) < K + 1
    ).dropna()

    # work in chunks for memory performance
    sampled = pd.DataFrame(data=None, columns=aggregated.columns)
    np.random.seed(random_state)
    for _ in range(int(np.ceil(N / 100))):
        n = min((N - sampled.shape[0], 100))
        temp = aggregated.sample(n=n, replace=True)

        temp["packets_length_total"] = (
            temp["packets_length_total"]
            .map(lambda l: l[np.random.randint(len(l) - K) :])
            .map(lambda l: l[:K])
            .map(np.array)
        )
        sampled = sampled.append(temp)

    return sampled.reset_index(drop=True)


def dataset_windowed_mean_variance(
    data_dir=data_dir, data_file=data_file, K=100, stride=0.2, filter=None,
):
    """ Generate dataset of arrays with K subsequent packet lengths using a sliding window approach.

    The original dataset is aggregated by sequence and cropped with a sliding window with desired size and stride.


    Parameters
    ----------
    data_dir : str, optional
        relative directory path (from project root) to dataset location
    
    data_file : str, optional
        dataset (csv) filename excl. extension

    K : int
        the size of each sample, number of subsequent packet lengths

    stride : float or int or None, optional
        the stride between windows, if `stride < 1` then `int(K*stride)` is used, if `stride == None` then `stride = K`

    filter : str, optional
        filter by `"ingress"`, `"egress"` or `None` (default)

    Returns
    -------
    ds
        the dataset
    """
    if stride is None:
        stride = K
    elif stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    elif stride < 1:
        stride = int(K * stride)

    ds = read_dataset(data_dir=data_dir, data_file=data_file)

    if filter is None:
        ds["packet_length"] = ds["packet_length"].abs()
    elif filter == "ingress":
        ds = ds.query("packet_length < 0")
        ds["packet_length"] = -ds["packet_length"]
    elif filter == "egress":
        ds = ds.query("packet_length > 0")
    else:
        raise ValueError(f"Cannot filter by {filter}")

    rolling = ds.groupby(["app", "sequence"])["packet_length"].rolling(window=K)

    ds = (
        ds.groupby(["app", "sequence"])["packet_length"]
        .rolling(window=K, min_periods=int(K / 2))
        .agg(dict(packets_length_mean="mean", packets_length_std="std",))
        .dropna()
        .iloc[::stride]
    )
    ds = ds.reset_index()
    return ds


def sliding_window(packets_length, K, stride):
    return [
        packets_length[i : i + K] for i in range(0, len(packets_length) - K + 1, stride)
    ]
