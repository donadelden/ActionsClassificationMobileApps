import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pandas as pd
from dataset import dataset_mean_variance, dataset_windowed, read_dataset

import matplotlib.pyplot as plt

import numpy as np


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-d", "--ds_type", dest="dataset_type", default="mean_variance")
    p.add_argument("-f", "--filter")
    args = p.parse_args()

    if args.dataset_type == "mean_variance":
        ds = dataset_mean_variance(agg_by="sequence", filter=args.filter, na="drop")

        if args.filter != "both":

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
                    ds["packets_length_mean"].where(ds["app"] == u),
                    ds["packets_length_std"].where(ds["app"] == u),
                    c=c[u],
                    s=1.0,
                    label=u,
                    alpha=0.3,
                )

            plt.legend()
            plt.xlabel("mean pkt lenght")
            plt.ylabel("variance")
            plt.show()
        else:
            print(ds)

    elif args.dataset_type == "windowed":
        ds = dataset_windowed(N=5, K=100, filter=args.filter)
        print(ds)
        plt.figure()
        for _, row in ds.iterrows():
            plt.plot(row["packets_length_total"], label=row["app"])

        plt.legend()
        plt.show()
