import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pandas as pd
from dataset import (
    dataset_mean_variance,
    read_dataset,
    dataset_mean_variance_ingress,
    dataset_mean_variance_egress,
    dataset_mean_variance2,
)
import matplotlib.pyplot as plt

import numpy as np


if __name__ == "__main__":
    ds = dataset_mean_variance2()

    c = {
        "facebook": "blue",
        "twitter": "cyan",
        "gmail": "red",
        "gplus": "magenta",
        "tumblr": "purple",
        "dropbox": "yellow",
        "evernote": "green",
    }

    # ds = ds.query('app == "facebook" | app == "twitter"')
    # ds = ds.query('sequence == 1')

    plt.figure()
    for u in ds["app"].unique():
        plt.scatter(
            ds["packets_length_mean_ingress"].where(ds["app"] == u),
            ds["packets_length_mean_egress"].where(ds["app"] == u),
            c=c[u],
            s=1.0,
            label=u,
            alpha=0.3,
        )

    plt.legend()
    plt.xlabel("mean pkt lenght")
    plt.ylabel("variance")
    plt.show()
