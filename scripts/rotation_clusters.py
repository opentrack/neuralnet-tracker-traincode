import h5py
from scipy.spatial.transform.rotation import Rotation
import numpy as np
from matplotlib import pyplot
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import argparse
import time

import vis
from datasets.preprocessing import imdecode, labels_to_lists


def compute_clustering_by_quats(quats, n_clusters=128, return_model=False):
    # Make real component positive to make clustering easier. Can do that because q and -q are the same rotation. Real component is last.
    quats = np.array(quats, copy=True)
    mask = quats[:,3]<0
    quats[mask] *= -1
    
    N = len(quats)
    batch_size = 1024 if N > 1024*8 else N
    
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size,
                          n_init=10, max_no_improvement=10, verbose=0)
    mbk.fit(quats)
    if return_model:
        return  mbk
    else:
        return mbk.labels_


def print_info(clusters):
    count = sum(c.shape[0] for c in clusters)
    # Number of times the samples in the largest cluster will be selected per sweep
    num_passes = count / (max(c.shape[0] for c in clusters)*len(clusters))
    # Number of times the samples in the smallest cluster will be selected per sweep
    oversampling = count / (min(c.shape[0] for c in clusters)*len(clusters))
    ratio = (max(c.shape[0] for c in clusters)) / (min(c.shape[0] for c in clusters))
    print (f"Number of data points = {count}\n"
           f"Times sampled / sweep | largest cluster  = {num_passes:.2f}\n"
           f"Times sampled / sweep | smallest cluster = {oversampling:.2f}\n"
           f"Ratio = {ratio:.1f}\n")


def visualize_frequencies(clusters):
    N = len(clusters)
    counts = np.sort([c.shape[0] for c in clusters])
    pyplot.bar(np.arange(N), counts)
    pyplot.show(block=True)



def render_data(data, idx):
    # TODO: handle JPEG and uncompressed array
    #img = data['images'][idx]
    # if img.ndim == 2:
    #     img = np.tile(img[...,None], (1,1,3))
    img = imdecode(data['images'][idx], color=True)
    vis.draw_axis(img, Rotation.from_quat(data['quats'][idx]), data['coords'][idx,0], data['coords'][idx,1], data['coords'][idx,2]*0.5)
    vis.draw_points3d(img, data['pt3d_68'][idx], labels=False)
    vis.draw_roi(img, data['rois'][idx], (255,0,255), 1)
    vis.draw_roi(img, data['head_rois'][idx], (255,255,255), 1)
    return img

def render_cluster(cluster, datafile, max_count):
    items = np.sort(cluster)[:max_count]
    imgs = [ render_data(datafile, i) for i in items ]
    imgs = np.concatenate(imgs, axis=0)
    return imgs


def visualize_faces(clusters, filename):
    with h5py.File(filename, 'r') as datafile:
        N = 16
        cols = []
        clusters = sorted(clusters, key=lambda x: len(x))
        for c in [ 0, 2, 20, 50, 80, len(clusters)-10, len(clusters)-1 ]:
            imgs = render_cluster(clusters[c], datafile, max_count=N)
            cols.append(imgs)
        cols = np.concatenate(cols, axis=1)
        Image.fromarray(cols).show()


def main():
    parser = argparse.ArgumentParser(description="Add clusters for uniformly distributed sampling of orientations")
    parser.add_argument('filename', help="File to process", type=str)
    parser.add_argument('--add-clusters', help="Add a dataset with cluster labels to the file", action='store_true', default=False)
    args = parser.parse_args()

    t = time.time()
    with h5py.File(args.filename, 'r') as datafile:
        quats = np.asarray(datafile['quats'])
    labels = compute_clustering_by_quats(quats)
    clusters = labels_to_lists(labels)
    print (f"Clustering time {time.time()-t:.1f}")
    assert all(np.all(labels[c]==i) for i,c in enumerate(clusters))

    print_info(clusters)
    visualize_frequencies(clusters)
    visualize_faces(clusters, args.filename)

    if args.add_clusters:
        with h5py.File(args.filename, 'a') as datafile:
            assert (np.all(labels <= 255))
            ds = datafile.require_dataset('cluster-labels', (len(labels),), np.uint8, exact=True)
            ds[...] = labels


if __name__ == "__main__":
    main()