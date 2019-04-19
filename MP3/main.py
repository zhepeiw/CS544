from torchvision import datasets, transforms
import torch
from torchvision.utils import save_image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
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
    save_image(dns_ds, './data.pdf')

    return dns_ds

def cluster_dataset(im):
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
    labels_noise = [rand_labels[i] if perm[i] else labels[i] for i in range(len(labels))]

    X_km = np.array([centers[label] for label in labels])
    im_km = X_km.reshape(32, 32, 3)

    X_noise = np.array([centers[label] for label in labels_noise])
    im_noise = X_noise.reshape(32, 32, 3)


    plt.figure(figsize=(20, 10))
    plt.subplot(131)
    plt.imshow(im.transpose(0, 1).transpose(1, 2).numpy())
    plt.title('Original')
    plt.subplot(132)
    plt.imshow(im_km)
    plt.title('Transformed')
    plt.subplot(133)
    plt.imshow(im_noise)
    plt.title('Noised')
    plt.savefig('im_0.pdf')
    plt.close()

    return X_km, X_noise, centers 
    

if __name__ == '__main__':
    ds = prepare_dataset()
    X_km, X_noise, centers = cluster_dataset(ds[0])

