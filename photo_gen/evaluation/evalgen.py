import os
from tqdm import tqdm
from math import log2
import yaml

import torch
import numpy as np
from sklearn.cluster import KMeans
# from kymatio.torch import Scattering2D
# from icecream import ic
import matplotlib.pyplot as plt
import infomeasure as im
import hydra
from omegaconf import OmegaConf

from eval_utils import tonumpy, normalise


device = torch.device('cuda')
dtype = torch.float32
OmegaConf.register_new_resolver("env", lambda k: os.environ.get(k, ""))

TRAINSETDIR = "/home/mila/l/letournv/scratch/nanophoto/topoptim/fulloptim/"


def twoplots_ax(x, y1, y2, xlabel, label1, label2, title, savepath):
    if type(y1) is torch.Tensor:
        y1 = tonumpy(y1)
        y2 = tonumpy(y2)

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.plot(x, y1, color=color, label=label1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(label1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.plot(x, y2, color=color, label=label2)
    ax2.set_ylabel(label2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(title)
    fig.tight_layout()
    plt.savefig(savepath)
    plt.clf()


def id_embedding(data):
    return data


def get_dist_matrice(data, distance):
    N = data.shape[0]
    distmat = torch.zeros(N, N)
    for i in range(data.shape[0]):
        for j in range(i):
            distmat[i, j] = distance(data[i], data[j])
    distmat += distmat.clone().transpose(0, 1)
    return distmat


def get_images_and_fom(path: str | os.PathLike):
    """
    path is the dir containing images and fom files
    """
    imagespath = os.path.join(path, 'images.npy')
    fompath = os.path.join(path, 'fom.npy')
    images = np.load(imagespath)
    fom = np.load(fompath)
    return images, fom


def compute_pairwise_distances(A, B=None):
    if B is None:
        B = A
    if B.ndim == 2:
        B = B.unsqueeze(0)
    if A.ndim == 2:
        A = A.unsqueeze(0)
    assert A.shape[1:] == B.shape[1:]
    shape = (A.shape[0], B.shape[0])+A.shape[1:]
    expandedA = A.unsqueeze(1).expand(shape)
    expandedB = B.unsqueeze(0).expand(shape)
    diffnorm = torch.norm(expandedA-expandedB, dim=(-2, -1))
    assert diffnorm.shape == (A.shape[0], B.shape[0])
    return diffnorm


def get_average_distances(data, savepath):
    N = data.shape[0]
    dist = compute_pairwise_distances(data, data)
    avg_dist = torch.sum(dist)/N**2
    plt.hist(tonumpy(dist), bins=100, density=True)
    plt.title('Distances histogram')
    plt.savefig(os.path.join(savepath, 'disthist.png'))
    plt.clf()
    return avg_dist


def get_avg_dissim(data, savepath):
    N = data.shape[0]
    sim = torch.zeros(N, N)
    for i in range(N):
        for j in range(i):
            zi = torch.norm(data[i])
            zj = torch.norm(data[j])
            sim[i, j] = torch.tensordot(data[i]/zi, data[j]/zj)
    sim += sim.transpose(-2, -1).clone()
    dissim = 1-sim
    avg_dissim = dissim.sum()/(N**2)
    dissim = tonumpy(dissim).flatten()
    plt.hist(dissim, bins=100, density=True)
    plt.savefig(os.path.join(savepath, 'dissim.png'))
    plt.clf()
    return avg_dissim


def elbo_km_clusters(data):
    if type(data) is torch.Tensor:
        data = tonumpy(data)
    if data.ndim >= 3:
        data = data.reshape(data.shape[0], -1)
    N = data.shape[0]
    wcss = []
    for i in tqdm(range(1, N)):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300,
                        n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        with open('kmeans.py', 'w') as file:
            file.write(str(wcss))
        if i > 2:
            # compute the second derivative and compare to eps
            if np.abs(wcss[-1] - 2*wcss[-2] + wcss[-3]) < 5e3:
                return i - 1, kmeans.predict(data)
    return None, None


def clusters_analysis(data, cluster_idx):
    n = np.unique(cluster_idx).shape[0]
    mean = []
    std = []
    for i in range(n):
        cluster_data = data[cluster_idx == i]
        distances = pair_dist_list(cluster_data)
        mean.append(distances.mean())
        std.append(distances.std())
    return mean, std


def smooth(x):
    x.append(x[-1])
    x.prepend(x[0])
    smoothx = [(x[i+2] + x[i+1] + x[i])/3 for i in range(len(x)-3)]
    return smoothx


def pair_dist_list(data):
    if type(data) is not np.ndarray:
        data = tonumpy(data)
    n = data.shape[0]
    distances = []
    for i in range(n):
        for j in range(i):
            distance = np.linalg.norm(data[i] - data[j])
            distances.append(distance)
    return torch.tensor(distances)


def box_plot(x, data, statname, datasetname, metricname, savepath, label=None):
    if type(data) is torch.Tensor:
        data = tonumpy(data)

    title = statname + ' ' + datasetname + ' ' + metricname
    if type(data) is torch.Tensor:
        data = tonumpy(data)
    fig, ax = plt.subplots()
    bp = ax.boxplot(data.transpose(), positions=x, widths=5, patch_artist=True)
    for box in bp['boxes']:
        box.set_facecolor('c')
    ax.set_ylabel(label)
    plt.xticks(rotation=90)
    plt.ylim(0, 80)

    plt.title(title)
    fig.tight_layout()
    filename = statname + datasetname + metricname + 'boxplot.png'
    plotpath = os.path.join(savepath, filename)
    plt.savefig(plotpath)
    plt.clf()


def mean_and_std_plot(steps, data, savepath, statname, datasetname,
                      metricname):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if type(data) is torch.Tensor:
        data = tonumpy(data)
    n = data.shape[0]
    mean = []
    std = []
    for k in range(n):
        kdist = tonumpy(data[:, k]).flatten()
        mean.append(kdist.mean())
        std.append(kdist.std())
    filename = statname + datasetname + metricname + 'meanstd.png'
    plotpath = os.path.join(savepath, filename)
    title = f'all up to k-th nn {datasetname} stats'
    twoplots_ax(steps, mean, std, 'k', 'mean', 'std', title, plotpath)


def accumulate_data(data, steps):
    device = data.device
    n = data.shape[0]
    accum_data = torch.empty(0, n).to(device)
    for k in steps:
        subsample = data[:, :k]
        means = subsample.mean(dim=1).unsqueeze(0)
        accum_data = torch.concat([accum_data, means], dim=0)
    return accum_data


def is_sorted(tensor):
    for row in tensor:
        if torch.all(row[:-1] <= row[1:]):
            return True
        else:
            return False


def single_analysis(data, metricname, savepath):
    n = data.shape[0]
    sorted_data = torch.sort(data, dim=-1, descending=False)[0]
    sorted_data = sorted_data[:, 1:]  # getting rid of 0
    steps = np.array(list(range(0, n, 10)))
    data_subsample = sorted_data[:, steps].transpose(0, 1)
    statname = 'cst_steps'
    analysis(steps, data_subsample, statname, metricname, savepath)


def all_analysis(data, metricname, savepath):
    n = data.shape[0]
    sorted_data = torch.sort(data, dim=-1, descending=False)[0]
    sorted_data = sorted_data[:, 1:]  # getting rid of 0
    # logscale
    steps = list(range(0, int(log2(n))))
    steps = np.array([2**k for k in steps])
    accum_data = accumulate_data(sorted_data, steps)
    statname = 'exp_steps'
    analysis(steps, accum_data, statname, metricname, savepath)
    # linear
    steps = np.array(list(range(0, n, 10)))
    accum_data = accumulate_data(sorted_data, steps)
    statname = 'cst_steps'
    analysis(steps, accum_data, statname, metricname, savepath)


def analysis(steps, data, statname, metricname, savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    data = tonumpy(data)
    datasetname = savepath.split('/')[-3]
    box_plot(steps, data, statname, datasetname, metricname, savepath)
    plottitle = "_".join([statname, metricname])
    savepath = os.path.join(savepath, plottitle + '.png')
    twoplots_ax(steps, np.median(data, axis=-1), data.std(axis=-1), 'k',
                'median', 'std', plottitle, savepath)


def knn_elbo_cluster_stats(data, savepath):
    num_clusters, cluster_idx = elbo_km_clusters(data)
    mean, std = clusters_analysis(data, cluster_idx)
    x = list(range(len(mean)))
    graphpath = os.path.join(savepath, 'clusteranalysis.png')
    twoplots_ax(x, mean, std, 'cluster idx', 'mean', 'std',
                'Elbow clusters stats', graphpath)
    return


def get_dissim(data):
    N = data.shape[0]
    sim = torch.zeros(N, N)
    for i in range(N):
        for j in range(i):
            zi = torch.norm(data[i])
            zj = torch.norm(data[j])
            sim[i, j] = torch.tensordot(data[i]/zi, data[j]/zj)
    sim += sim.transpose(-2, -1).clone() + torch.eye(N)
    dissim = 1-sim
    return dissim


def knear_neighbor_analysis(data, savepath):
    dist = compute_pairwise_distances(data, data)
    single_analysis(dist, 'k-th nn dist', savepath)


def all_knn_analysis(data, savepath):
    dist = compute_pairwise_distances(data, data)
    all_analysis(dist, 'cumulative k-th nn dist', savepath)


def single_k_dissim_analysis(data, savepath):
    dissim = get_dissim(data)
    single_analysis(dissim, 'dissim', savepath)


def all_k_dissim_analysis(data, savepath):
    dissim = get_dissim(data)
    all_analysis(dissim, 'cumulative dissim', savepath)


def basic_stats(data, fom, savepath):
    path = os.path.join(savepath, 'basicstats.txt')
    with open(path, 'w') as file:
        file.write(f'''Mean {data.mean()}, Median {data.median()},
                    Std {data.std()}''')




def test__nn_distance_to_train_ds():
    gen_path = "~/scratch/nanophoto/evalgen/topoptimGAN/data/images.npy"
    gen_data = np.load(os.path.expanduser(gen_path))
    gen_data = torch.tensor(gen_data)
    train_path = "~/scratch/nanophoto/evalgen/topoptim/data/images.npy"
    train_data = np.load(os.path.expanduser(train_path))
    train_data = torch.tensor(train_data)
    nn_distance_to_train_ds(gen_data, train_data, "test")


def eval_metrics(datasets: dict["name":str, "path":str],
                 trainset_path: str = TRAINSETDIR,
                 n_samples: int | None = None):
    """
    For each dataset, compute the metrics from the metrics list
    inputs:
        datasets: name, path dict, the path is the dir containing images.npy
        and fom.npy
        trainset_path: the path of the training set images.npy
        n_samples:  how many images to use in the dataset at evaluation of the
        metrics
    """
    trainset_images, _ = get_images_and_fom(trainset_path)
    if n_samples is None:
        n_samples = trainset_images.shape[0]
    trainset_images = torch.tensor(trainset_images)
    eval_stats_dict = {
        "train-set-nn": lambda name, x, y: nn_distance_to_train_ds(name, x,
                                                       trainset_images, y),
        "pca_dim_reduction_entropy": pca_dim_reduction_entropy,
        # "entropy": lambda n, x, p: compute_entropy(x, p)
        # "knn-distance": lambda data, savepath: single_analysis(data,
        # "knn-similarity":
    }

    N = n_samples
    for ds_cfg in datasets:
        path = os.path.expanduser(ds_cfg.path)
        images, fom = get_images_and_fom(path)
        images = normalise(images[:N])
        fom = fom[:N]
        savepath = ds_cfg.path
        savepath = os.path.expanduser(savepath)

        for stat_name, stat_fn in eval_stats_dict.items():
            os.makedirs(savepath, exist_ok=True)
            stat_fn(ds_cfg.name, images, savepath)


@hydra.main(version_base=None, config_path='config', config_name='evalgen')
def main(cfg):
    """
    Read the cfg from hydra.
    Modifier config/evalgen.yaml to change the datasets and metrics.
    """
    trainset_path = os.path.expanduser(cfg.trainset.path)
    eval_metrics(cfg.datasets, trainset_path, cfg.n_samples)
    print("DONE")


if __name__ == '__main__':
    main()
    # nn_distances = nn_distance_to_train_ds(gen_ds, train_ds)
