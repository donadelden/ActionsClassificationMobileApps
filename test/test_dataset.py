import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pandas as pd
from dataset import (
    dataset_mean_variance,
    dataset_windowed_mean_variance,
    # dataset_windowed,
    dataset_windowed_random,
    read_dataset,
)

import matplotlib.pyplot as plt

import numpy as np


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-d", "--ds_type", dest="dataset_type", default="mean_variance")
    p.add_argument("-f", "--filter")
    args = p.parse_args()

    if args.dataset_type == "mean_variance":
        # ds = dataset_mean_variance(agg_by="sequence", filter=args.filter, na="drop")
        ds = dataset_windowed_mean_variance(filter=args.filter)

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
            labels = tuple()
            for app, group in ds.groupby("app"):
                group.plot.scatter(
                    x="packets_length_mean",
                    y="packets_length_std",
                    c=c[app],
                    s=2,
                    ax=plt.gca(),
                )
                labels += (app,)

            plt.legend(labels)
            plt.xlabel("mean pkt lenght")
            plt.ylabel("variance")
            plt.show()
        else:
            print(ds)

    elif args.dataset_type == "windowed_random":
        ds = dataset_windowed_random(N=5, K=100, filter=args.filter)
        print(ds)
        plt.figure()
        for _, row in ds.iterrows():
            plt.plot(row["packets_length_total"], label=row["app"])

        plt.legend()
        plt.show()

    elif args.dataset_type == "windowed":
        ds = dataset_windowed(K=10, stride=None, filter=args.filter)
        print(ds)
        print(ds["app"].value_counts())

        plt.figure()
        labels = tuple()
        for app, group in (
            ds.explode("packets_length_total").sample(100000).groupby("app")
        ):
            group.plot.kde(ax=plt.gca())
            labels += (app,)
        plt.legend(labels)
        plt.xlim(xmin=-2000, xmax=2000)
        plt.show()
