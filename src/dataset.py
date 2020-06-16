import pandas as pd
import numpy as np
import tensorflow as tf
import re

data_dir = "data"
data_file = "apps_total_plus_filtered"


def dataset_mean_variance(data_dir=data_dir, data_file=data_file):
    ds = pd.read_csv(f"{data_dir}/{data_file}.csv")
    app = ds["app"]
    action = ds["action"]
    sequence = ds["sequence"].map(lambda l: int(re.sub(r"sequence_([0-9]+)", r"\1", l)))
    mean = (
        ds["packets_length_total"]
        .map(
            lambda l: np.mean(np.abs(np.fromstring(l[1:-1], sep=", ", dtype=np.float)))
        )
        .rename("packets_length_mean")
    )
    variance = (
        ds["packets_length_total"]
        .map(lambda l: np.var(np.abs(np.fromstring(l[1:-1], sep=", ", dtype=np.float))))
        .rename("packets_length_var")
    )
    return pd.concat([app, action, sequence, mean, variance], axis=1)


# ds = dataset_mean_variance()

# import matplotlib.pyplot as plt

# c = {
#   'facebook': 'blue',
#   'twitter': 'cyan',
#   'gmail': 'red',
#   'gplus': 'magenta',
#   'tumblr': 'purple',
#   'dropbox': 'yellow',
#   'evernote': 'green'
# }

# plt.figure()
# for u in ds['app'].unique():
#   plt.scatter(ds['packets_length_mean'].where(ds['app'] == u), ds['packets_length_var'].where(ds['app'] == u), c=c[u], s=1., label=u)

# plt.legend()
# plt.xlabel('mean pkt lenght')
# plt.ylabel('variance')
# plt.show()
