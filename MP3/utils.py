from torchvision import datasets, transforms
import torch
from torchvision.utils import save_image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import pdb

def prepare_dataset():
    ds = datasets.CIFAR10(root='data/cifar10', train=True, transform=transforms.ToTensor(), download=True)
    dns_ds = []
    dns_label = set()
    counter = 0
    while counter < 10:
        idx = torch.randint(0, len(ds), (1,)).item()
        im, label = ds[idx]
        if label not in dns_label:
            dns_label.add(label)
            dns_ds.append(im)
            counter += 1

    dns_ds = torch.stack(dns_ds, dim=0)
    if not os.path.exists('out'):
        os.mkdir('out')
    save_image(dns_ds, 'out/data.pdf')

    return dns_ds

def cluster_noise(im):
    '''
        Args:
            im: tensor with shape 3 x 32 x 32
    '''
    model = KMeans(n_clusters=32, random_state=0)
    X = im.numpy().reshape(3, -1).T
    model = model.fit(X)
    centers = model.cluster_centers_ 
    labels = model.predict(X)

    perm = np.random.rand(len(labels)) <= 1/32
    rand_labels = np.random.randint(0, 32, size=len(perm))
    labels_noise = np.array([rand_labels[i] if perm[i] else labels[i] for i in range(len(labels))])

    X_km = np.array([centers[label] for label in labels])
    im_km = X_km.reshape(32, 32, 3)

    X_noise = np.array([centers[label] for label in labels_noise])
    im_noise = X_noise.reshape(32, 32, 3)


    #  if not os.path.exists('out'):
    #      os.mkdir('out')
    #  plt.figure(figsize=(20, 10))
    #  plt.subplot(131)
    #  plt.imshow(im.transpose(0, 1).transpose(1, 2).numpy())
    #  plt.title('Original')
    #  plt.subplot(132)
    #  plt.imshow(im_km)
    #  plt.title('Transformed')
    #  plt.subplot(133)
    #  plt.imshow(im_noise)
    #  plt.title('Noised')
    #  plt.savefig('out/im.pdf')
    #  plt.close()

    labels = labels.reshape(32, 32)
    labels_noise = labels_noise.reshape(32, 32)

    return labels, labels_noise, centers 

def visualize_result(y_km, y_ns, y_re, centers, save_name='res.pdf'):
    '''

    '''
    im_km = np.zeros(y_km.shape + (3,))
    im_ns = np.zeros_like(im_km)
    im_re = np.zeros_like(im_km)
    for i in range(im_km.shape[0]):
        for j in range(im_km.shape[1]):
            im_km[i, j] = centers[y_km[i, j]]
            im_ns[i, j] = centers[y_ns[i, j]]
            im_re[i, j] = centers[y_re[i, j]]

    if not os.path.exists('out'):
        os.mkdir('out')
    plt.figure(figsize=(20, 10))
    plt.subplot(131)
    plt.imshow(im_km)
    plt.title('K-means')
    plt.subplot(132)
    plt.imshow(im_ns)
    plt.title('Noisy')
    plt.subplot(133)
    plt.imshow(im_re)
    plt.title('Denoised')
    out_path = os.path.join('out/', save_name)
    plt.savefig(out_path)
    plt.close()



if __name__ == '__main__':
    ds = prepare_dataset()
    X_km, X_noise, centers = cluster_noise(ds[0])

