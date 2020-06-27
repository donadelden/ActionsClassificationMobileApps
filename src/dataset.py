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
        ).reset_index(drop=True)
    )
    return ds


def aggregate_flows_by_sequence(ds):
    ds = (
        ds[["app", "sequence", "packets_length_total"]]
        .groupby(["app", "sequence"], sort=False, as_index=False)
        .agg(
            {
                "packets_length_total": lambda group: tuple(
                    np.concatenate(group.tolist())
                )
            }
        )
        .set_index("sequence")
    )
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
        ds = aggregate_flows_by_sequence(ds)
    elif agg_by == "action":
        ds = aggregate_flows_by_action(ds)
    else:
        raise ValueError(f"Can only aggregate by action or sequence, not {agg_by}")

    app = ds["app"]
    # action = ds["action"]
    # sequence = ds["sequence"]

    if filter == "both":
        filtered_ingress = (
            ds["packets_length_total"]
            .map(np.array)
            .map(lambda l: l[l > 0])
            .where(lambda s: s.map(lambda l: l.size) > 0)
            .dropna()
        )
        filtered_egress = (
            ds["packets_length_total"]
            .map(np.array)
            .map(lambda l: -l[l < 0])
            .where(lambda s: s.map(lambda l: l.size) > 0)
            .dropna()
        )
        mean_ingress = filtered_ingress.map(np.mean).rename("packets_length_mean")
        variance_ingress = filtered_ingress.map(np.std).rename("packets_length_std")
        mean_egress = filtered_egress.map(np.mean).rename("packets_length_mean")
        variance_egress = filtered_egress.map(np.std).rename("packets_length_std")

        ds = pd.concat(
            [app, mean_ingress, variance_ingress, mean_egress, variance_egress], axis=1
        )

    else:
        if filter == None:
            filtered = ds["packets_length_total"].map(np.abs)
        elif filter == "ingress":
            filtered = (
                ds["packets_length_total"]
                .map(np.array)
                .map(lambda l: l[l > 0])
                .where(lambda s: s.map(lambda l: l.size) > 0)
                .dropna()
            )
        elif filter == "egress":
            filtered = (
                ds["packets_length_total"]
                .map(np.array)
                .map(lambda l: -l[l < 0])
                .where(lambda s: s.map(lambda l: l.size) > 0)
                .dropna()
            )
        else:
            raise ValueError(f"Cannot filter by {filter}")

        mean = filtered.map(np.mean).rename("packets_length_mean")
        variance = filtered.map(np.std).rename("packets_length_std")

        ds = pd.concat([app, mean, variance], axis=1)

    if na == "fill":
        ds.fillna(value=na_value, inplace=True)
    elif na == "drop":
        ds.dropna(inplace=True)
    elif na != None:
        raise ValueError(f"cannot use {na} method to treat NA/NaN")

    return ds


def dataset_windowed(
    data_dir=data_dir,
    data_file=data_file,
    agg_by="sequence",
    N=10000,
    K=100,
    filter=None,
    random_state=None,
):
    """ Generate dataset of N arrays with K subsequent packet lengths.

    The original dataset is aggregated by sequence, then randomly sampled with replacement, and cropped randomly to match the desired length.

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
