import numpy as np
import scipy.io as sio
from sklearn import preprocessing
from sklearn.decomposition import PCA

from keras import backend as K
from keras.models import model_from_json
from keras.utils import plot_model

from core import util
from core import pairs


def get_data(params):
    """
    Convenience function: preprocesses all data in the manner specified in params, and returns it
    as a nested dict with the following keys:

    the permutations (if any) used to shuffle the training and validation sets
    'p_train'                           - p_train
    'p_val'                             - p_val

    the data used for spectral net
    'spectral'
        'train_and_test'                - (x_train, y_train, x_val, y_val, x_test, y_test)
        'train_unlabeled_and_labeled'   - (x_train_unlabeled, y_train_unlabeled, x_train_labeled, y_train_labeled)
        'val_unlabeled_and_labeled'     - (x_val_unlabeled, y_val_unlabeled, x_val_labeled, y_val_labeled)

    the data used for siamese net, if the architecture uses the siamese net
    'siamese'
        'train_and_test'                - (pairs_train, dist_train, pairs_val, dist_val)
        'train_unlabeled_and_labeled'   - (pairs_train_unlabeled, dist_train_unlabeled, pairs_train_labeled, dist_train_labeled)
        'val_unlabeled_and_labeled'     - (pairs_val_unlabeled, dist_val_unlabeled, pairs_val_labeled, dist_val_labeled)
    """
    # data list (views)
    data_list = []

    if params["views"] is None:
        params["views"] = range(1, params["view_size"] + 1)

    for i in params["views"]:
        view_name = "view" + str(i)
        print("********", "load", params["dset"], view_name, "********")
        ret = {}
        x_train, y_train, x_test, y_test = load_data(params, i)
        print("data size (training, testing)", x_train.shape, x_test.shape)
        print("train data (max, min)", np.max(x_train), np.min(x_test))

        # Using all data to train
        if params["use_all_data"]:
            x_train = np.concatenate((x_train, x_test))
            y_train = np.concatenate((y_train, y_test))
            x_test = x_train.copy()
            y_test = y_train.copy()

        # split x training, validation, and test subsets
        if params["val_set_fraction"] == 0:
            x_val = None
            y_val = None
            p_val = None
        elif params["val_set_fraction"] > 0 and params["val_set_fraction"] <= 1:
            train_val_split = (
                1 - params["val_set_fraction"],
                params["val_set_fraction"],
            )
            (x_train, y_train, p_train), (x_val, y_val, p_val) = split_data(
                x_train, y_train, train_val_split
            )
        else:
            raise ValueError("val_set_fraction must be in range [0, 1]")

        # Using the low-dimension data via AE
        if params["use_code_space"]:
            all_data = [x_train, x_val, x_test]
            for j, d in enumerate(all_data):
                all_data[j] = embed_data(d, dset=params["dset"], view_name=view_name)
            x_train, x_val, x_test = all_data

            # save the low-dimension data for comparison
            if params["val_set_fraction"] == 0:
                dic = {
                    "x_train": x_train,
                    "x_test": x_test,
                    "y_train": y_train,
                    "y_test": y_test,
                }
            else:
                dic = {
                    "x_train": np.concatenate((x_train, x_val)),
                    "x_test": x_test,
                    "y_train": np.concatenate((y_train, y_val)),
                    "y_test": y_test,
                }
        elif params["use_code_space"] == "pca":
            # PCA
            pca = PCA(n_components=10)
            x_train = pca.fit_transform(x_train)
            if x_val is not None:
                pca = PCA(n_components=10)
                x_val = pca.fit_transform(x_val)
            pca = PCA(n_components=10)
            x_test = pca.fit_transform(x_test)

        # data for MvLNet
        ret["spectral"] = (x_train, y_train, x_val, y_val, x_test, y_test)

        # prepare the training pairs for SiameseNet
        if "siamese" in params["affinity"]:
            ret["siamese"] = {}
            pairs_train, dist_train = pairs.create_pairs_from_unlabeled_data(
                x1=x_train,
                k=params["siam_k"],
                tot_pairs=params["siamese_tot_pairs"],
                use_approx=params["use_approx"],
            )
            if x_val is not None:
                pairs_val, dist_val = pairs.create_pairs_from_unlabeled_data(
                    x1=x_val,
                    k=params["siam_k"],
                    tot_pairs=params["siamese_tot_pairs"],
                    use_approx=params["use_approx"],
                )
            else:
                pairs_val = None
                dist_val = None

            # data for SiameseNet
            ret["siamese"] = (pairs_train, dist_train, pairs_val, dist_val)

        data_list.append(ret)

    return data_list


def load_data(params, view):
    """
    Convenience function: reads from disk, downloads, or generates the data specified in params
    """
    #  multiple view
    if params["dset"] == "noisymnist" or params["dset"] == "noisymnist_10000":
        data = util.load_data(
            "noisymnist_view" + str(view) + ".gz",
            "https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view"
            + str(view)
            + ".gz",
        )
        train_set_x, train_set_y = data[0]
        valid_set_x, valid_set_y = data[1]
        test_set_x, test_set_y = data[2]

        # split it into two partitions
        if params["dset"] == "noisymnist_10000":
            # Use the first 10000 samples
            test_set_x = train_set_x[5000:10000]
            test_set_y = train_set_y[5000:10000]
            train_set_x = train_set_x[:5000]
            train_set_y = train_set_y[:5000]
        elif params["dset"] == "noisymnist":
            train_set_x, train_set_y = np.concatenate(
                (train_set_x, valid_set_x), axis=0
            ), np.concatenate((train_set_y, valid_set_y), axis=0)
    elif params["dset"] in ["Caltech101-all", "Caltech101-7", "Caltech101-20"]:
        mat = sio.loadmat("./data/" + params["dset"] + ".mat")
        X = mat["X"][0]
        x = X[view - 1]
        x = util.normalize(x)
        y = np.squeeze(mat["Y"])

        # split it into two partitions
        data_size = x.shape[0]
        train_index, test_index = util.random_index(data_size, int(data_size * 0.5), 1)
        test_set_x = x[test_index]
        test_set_y = y[test_index]
        train_set_x = x[train_index]
        train_set_y = y[train_index]
    elif params["dset"] == "wiki":
        # from pyx
        file_name = "./data/wiki_4096_3000.mat"
        mat = sio.loadmat(file_name)
        if view == 1:
            train_set_x = mat["I_tr"]
            test_set_x = mat["I_te"]
            train_set_y = np.squeeze(mat["trCatAll"])
            test_set_y = np.squeeze(mat["teCatAll"])
        elif view == 2:
            train_set_x = mat["T_tr"]
            test_set_x = mat["T_te"]
            train_set_y = np.squeeze(mat["trCatAll"])
            test_set_y = np.squeeze(mat["teCatAll"])
    else:
        raise ValueError("Dataset provided ({}) is invalid!".format(params["dset"]))

    return train_set_x, train_set_y, test_set_x, test_set_y


def embed_data(x, dset, view_name):
    """
    Convenience function: embeds x into the code space using the corresponding
    autoencoder (specified by dset).
    """
    if x is None:
        return None
    if not x.shape[0]:
        return np.zeros(shape=(0, 10))

    if dset == "noisymnist_10000":
        dset = "noisymnist"

    json_path = "./pretrain/ae/" + dset + "/ae_{}.json".format(dset + "_" + view_name)
    weights_path = (
        "./pretrain/ae/" + dset + "/ae_{}_weights.h5".format(dset + "_" + view_name)
    )

    with open(json_path) as f:
        pt_ae = model_from_json(f.read())
    pt_ae.load_weights(weights_path)
    x = x.reshape(-1, np.prod(x.shape[1:]))

    get_embeddings = K.function([pt_ae.input], [pt_ae.layers[4].output])

    get_reconstruction = K.function([pt_ae.layers[5].input], [pt_ae.output])
    x_embedded = predict_with_K_fn(get_embeddings, x)[0]
    x_recon = predict_with_K_fn(get_reconstruction, x_embedded)[0]
    reconstruction_mse = np.mean(np.square(x - x_recon))
    print(
        "using pretrained embeddings; sanity check, total reconstruction error:",
        np.mean(reconstruction_mse),
    )

    del pt_ae

    return x_embedded


def predict_with_K_fn(K_fn, x, bs=512):
    """
    Convenience function: evaluates x by K_fn(x), where K_fn is
    a Keras function, by batches of size 1000.
    """
    if not isinstance(x, list):
        x = [x]
    num_outs = len(K_fn.outputs)
    y = [np.empty((x[0].shape[0], output_.get_shape()[1])) for output_ in K_fn.outputs]
    recon_means = []
    for i in range(int(x[0].shape[0] / bs + 1)):
        x_batch = []
        for x_ in x:
            x_batch.append(x_[i * bs : (i + 1) * bs])
        temp = K_fn(x_batch)
        for j in range(num_outs):
            y[j][i * bs : (i + 1) * bs] = temp[j]

    return y


def split_data(x, y, split, permute=None):
    """
    Splits arrays x and y, of dimensionality n x d1 and n x d2, into
    k pairs of arrays (x1, y1), (x2, y2), ..., (xk, yk), where both
    arrays in the ith pair is of shape split[i-1]*n x (d1, d2)

    x, y:       two matrices of shape n x d1 and n x d2
    split:      a list of floats of length k (e.g. [a1, a2,..., ak])
                where a, b > 0, a, b < 1, and a + b == 1
    permute:    a list or array of length n that can be used to
                shuffle x and y identically before splitting it

    returns:    a tuple of tuples, where the outer tuple is of length k
                and each of the k inner tuples are of length 3, of
                the format (x_i, y_i, p_i) for the corresponding elements
                from x, y, and the permutation used to shuffle them
                (in the case permute == None, p_i would simply be
                range(split[0]+...+split[i-1], split[0]+...+split[i]),
                i.e. a list of consecutive numbers corresponding to the
                indices of x_i, y_i in x, y respectively)
    """
    n = x.shape[0]
    if permute is not None:
        if not isinstance(permute, np.ndarray):
            raise ValueError(
                "Provided permute array should be an np.ndarray, not {}!".format(
                    type(permute)
                )
            )
        if len(permute.shape) != 1:
            raise ValueError(
                "Provided permute array should be of dimension 1, not {}".format(
                    len(permute.shape)
                )
            )
        if len(permute) != n:
            raise ValueError(
                "Provided permute should be the same length as x! (len(permute) = {}, len(x) = {}".format(
                    len(permute), n
                )
            )
    else:
        permute = np.arange(n)

    if np.sum(split) != 1:
        raise ValueError("Split elements must sum to 1!")

    ret_x_y_p = []
    prev_idx = 0
    for s in split:
        idx = prev_idx + np.round(s * n).astype(np.int)
        p_ = permute[prev_idx:idx]
        x_ = x[p_]
        y_ = y[p_]
        prev_idx = idx
        ret_x_y_p.append((x_, y_, p_))

    return tuple(ret_x_y_p)


def pre_process(x_train, x_test, standardize):
    """
    Convenience function: uses the sklearn StandardScaler on x_train
    and x_test if standardize == True
    """
    # if we are going to standardize
    if standardize:
        # standardize the train data set
        preprocessor = preprocessing.StandardScaler().fit(x_train)
        x_train = preprocessor.transform(x_train)
        # if we have test data
        if x_test.shape[0] > 0:
            # standardize the test data set
            preprocessor = preprocessing.StandardScaler().fit(x_test)
            x_test = preprocessor.transform(x_test)
    return x_train, x_test
