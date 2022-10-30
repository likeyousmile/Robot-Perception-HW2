import torch
import os
from data import data_set, view, transform_all


def cache(filename: str):
    def deco(func):
        def wrapper(*args, **kwargs):
            if os.path.exists(filename):
                return torch.load(filename)
            else:
                embedded = func(*args, **kwargs)
                torch.save(embedded, filename)
                return embedded

        return wrapper

    return deco


@cache("./Fashion-Embedding-CAE-M-16.pt")
def embed_cae(data):
    from cae import CAEMedium, EMBEDDING_DIM
    network = CAEMedium(EMBEDDING_DIM)
    network.load_state_dict(torch.load("./ConvAE-Medium-16.pt"))
    network.eval()
    with torch.no_grad():
        encoded, _decoded = network(data.resize(10000, 1, 28, 28))
        return encoded.numpy()


@cache("./Fashion-Embedding-tSEN-2.pt")
def embed_tsne(data):
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, learning_rate="auto", init="random").fit_transform(data)


@cache("./Fashion-Embedding-PCA-2.pt")
def embed_pca(data):
    from sklearn.decomposition import PCA
    return PCA(n_components=2).fit_transform(data)


@cache("./Fashion-Embedding-PyMDE-2.pt")
def embed_pymde(data):
    import pymde
    return pymde.preserve_neighbors(data, embedding_dim=2).embed()


@cache("./Fashion-Embedding-UMAP-2.pt")
def embed_umap(data):
    import umap
    return umap.UMAP().fit_transform(data)


@cache("./Fashion-Embedding-CAE-M-2.pt")
def embed_cae_m_2(data):
    from cae import CAEMedium
    network = CAEMedium(2)
    network.load_state_dict(torch.load("./ConvAE-Medium-2.pt"))
    network.eval()
    with torch.no_grad():
        encoded, _decoded = network(data.resize(10000, 1, 28, 28))
        return encoded.numpy()


@cache("./Fashion-Embedding-CAE-S-2.pt")
def embed_cae_s_2(data):
    from cae import CAESmall
    network = CAESmall(2)
    network.load_state_dict(torch.load("./ConvAE-Small-2.pt"))
    network.eval()
    with torch.no_grad():
        encoded, _decoded = network(data.resize(10000, 1, 28, 28))
        return encoded.numpy()


if __name__ == "__main__":
    images = transform_all()
    embed_16 = embed_cae(images)

    embed_2 = {
        "t-SNE": embed_tsne(embed_16),
        "PCA": embed_pca(embed_16),
        "PyMDE": embed_pymde(embed_16),
        "UMAP": embed_umap(embed_16)
    }
    view.projection(embed_2, data_set.targets)

    embed_2_cae = {
        "ConvAE-M": embed_cae_m_2(images),
        "ConvAE-S": embed_cae_s_2(images),
    }
    view.projection(embed_2_cae, data_set.targets)
