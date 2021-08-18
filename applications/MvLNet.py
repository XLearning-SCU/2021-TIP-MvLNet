import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from keras.layers import Input

from . import Task
from core import networks


def run_net(datalist, config):

    # network input shapes stored as view-list
    MvLNet_input_shape = []
    SiameseNet_input_shape = []

    # training, validation and testing data stored as view-list
    x_train_list = []
    x_val_list = []
    x_test_list = []

    # get the labels
    _, y_train, _, y_val, _, y_test = datalist[0]["spectral"]

    for i in range(config["view_size"]):
        # UNPACK DATA
        data_i = datalist[i]
        x_train_i, y_train_i, x_val_i, y_val_i, x_test_i, y_test_i = data_i["spectral"]
        if config["val_set_fraction"] == 0:
            x = x_train_i
        else:
            x = np.concatenate((x_train_i, x_val_i), axis=0)

        x_test_list.append(x_test_i)
        x_train_list.append(x_train_i)
        x_val_list.append(x_val_i)

        # SiameseNet
        if config["affinity"] == "siamese":
            pairs_train, dist_train, pairs_val, dist_val = data_i["siamese"]

        # SET UP INPUTS
        batch_sizes = {
            "Embedding": config["batch_size"],
            "Orthogonal": config["batch_size_orthogonal"],
        }
        input_shape = x.shape[1:]

        # spectral input shape
        inputs = {
            "Embedding": Input(
                shape=input_shape, name="EmbeddingInput" + "view" + str(i)
            ),
            "Orthogonal": Input(
                shape=input_shape, name="OrthogonalInput" + "view" + str(i)
            ),
        }
        MvLNet_input_shape.append(inputs)

        # DEFINE AND TRAIN SIAMESE NET
        if config["affinity"] == "siamese":
            print("******** SiameseNet " + "view" + str(i + 1) + " ********")
            siamese_net = networks.SiameseNet(
                name=config["dset"] + "_" + "view" + str(i + 1),
                inputs=inputs,
                arch=config["arch"],
                siam_reg=config["siam_reg"],
            )
            history = siamese_net.train(
                pairs_train=pairs_train,
                dist_train=dist_train,
                pairs_val=pairs_val,
                dist_val=dist_val,
                lr=config["siam_lr"],
                drop=config["siam_drop"],
                patience=config["siam_patience"],
                num_epochs=config["siam_epoch"],
                batch_size=config["siam_batch_size"],
                pre_train=config["siam_pre_train"],
            )
        else:
            siamese_net = None
        SiameseNet_input_shape.append(siamese_net)

    if config["val_set_fraction"] == 0:
        x_val_list = None
    else:
        x_val_list = np.array(x_val_list).astype("float")

    # MvLNet
    mvlnet = networks.MvLNet(
        input_list=MvLNet_input_shape,
        arch=config["arch"],
        spec_reg=config["spectral_reg"],
        n_clusters=config["n_clusters"],
        affinity=config["affinity"],
        scale_nbr=config["scale_nbr"],
        n_nbrs=config["n_nbrs"],
        batch_sizes=batch_sizes,
        view_size=config["view_size"],
        siamese_list=SiameseNet_input_shape,
        x_train_siam=x_train_list,
        lamb=config["lamb"],
    )

    # training
    print("********", "Training", "********")
    mvlnet.train(
        x_train=x_train_list,
        x_val=x_val_list,
        lr=config["spectral_lr"],
        drop=config["spectral_drop"],
        patience=config["spectral_patience"],
        num_epochs=config["spectral_epoch"],
    )

    print("Finished training")
    print("********", "Prediction", "********")

    # get final training embeddings
    x_train_final_list = mvlnet.predict(x_train_list)

    print("Testing data")
    # get final testing embeddings
    x_test_final_list = mvlnet.predict(x_test_list)

    print("Learned representations for testing", x_test_final_list.shape)
    scores = {}

    # clustering
    y_preds, scores["clustering"] = Task.Clustering(
        x_test_final_list, y_test, view_specific=False
    )

    if config["dset"] == "Caltech101-20":
        # classification
        x_train_final_concat = np.concatenate(x_train_final_list[:], axis=1)
        x_test_final_concat = np.concatenate(x_test_final_list[:], axis=1)

        scores["classification"] = Task.Classification(
            x_train_final_concat, y_train, x_test_final_concat, y_test
        )

    if config["dset"] == "wiki":
        # retreival
        scores["retrieval"] = Task.Retrieval(x_test_final_list, y_test, [0, 1], [1, 0])

    return x_test_final_list, scores
