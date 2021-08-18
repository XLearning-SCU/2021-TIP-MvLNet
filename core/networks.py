import itertools

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda

from . import train
from . import costs
from .layer import stack_layers
from .util import LearningHandler, make_layer_list, train_gen, get_scale
from applications import Task


class SiameseNet:
    def __init__(self, name, inputs, arch, siam_reg):
        self.orig_inputs = inputs
        self.name = name
        # set up input shapes
        self.inputs = {
            "A": inputs["Embedding"],
            "B": Input(shape=inputs["Embedding"].get_shape().as_list()[1:]),
        }

        # generate layers
        self.layers = []
        self.layers += make_layer_list(arch, "siamese", siam_reg)

        # create the SiameseNet
        self.outputs = stack_layers(self.inputs, self.layers)

        # add the distance layer
        self.distance = Lambda(
            costs.euclidean_distance, output_shape=costs.eucl_dist_output_shape
        )([self.outputs["A"], self.outputs["B"]])

        # create the distance model for training
        self.net = Model([self.inputs["A"], self.inputs["B"]], self.distance)

        # compile the siamese network
        self.net.compile(
            loss=costs.get_contrastive_loss(m_neg=1, m_pos=0.05), optimizer="rmsprop"
        )

    def train(
        self,
        pairs_train,
        dist_train,
        pairs_val,
        dist_val,
        lr,
        drop,
        patience,
        num_epochs,
        batch_size,
        pre_train=False,
    ):
        # create handler for early stopping and learning rate scheduling
        self.lh = LearningHandler(
            lr=lr, drop=drop, lr_tensor=self.net.optimizer.lr, patience=patience
        )

        # initialize the training generator
        train_gen_ = train_gen(pairs_train, dist_train, batch_size)

        if pairs_val is None:
            validation_data = None
        else:
            # format the validation data for keras
            validation_data = ([pairs_val[:, 0], pairs_val[:, 1]], dist_val)

        # compute the steps per epoch
        steps_per_epoch = int(len(pairs_train) / batch_size)

        if pre_train:
            print("Use pretrained weights")
            self.net.load_weights(
                "./pretrain/siamese/" + self.name + "_siamese_weight" + ".h5"
            )
            return 0
        else:
            # train the network
            hist = self.net.fit_generator(
                train_gen_,
                epochs=num_epochs,
                validation_data=validation_data,
                steps_per_epoch=steps_per_epoch,
                callbacks=[self.lh],
            )
            self.net.save_weights(
                "./pretrain/siamese/" + self.name + "_siamese_weight" + ".h5"
            )
            return hist

    def predict(self, x, batch_sizes):
        # compute the siamese embeddings of the input data
        return train.predict_siamese(
            self.outputs["A"], x=x, inputs=self.orig_inputs, batch_sizes=batch_sizes
        )


class MvLNet:
    def __init__(
        self,
        input_list,
        arch,
        spec_reg,
        n_clusters,
        affinity,
        scale_nbr,
        n_nbrs,
        batch_sizes,
        view_size,
        siamese_list,
        x_train_siam,
        lamb,
    ):
        self.n_clusters = n_clusters
        self.input_list = input_list
        self.batch_sizes = batch_sizes
        # generate layers
        self.view_size = view_size

        # Embedding inputs shape
        self.input_shape = {"Embedding": [], "Orthogonal": []}
        for i in range(self.view_size):
            self.input_shape["Embedding"].append(self.input_list[i]["Embedding"])
            self.input_shape["Orthogonal"].append(self.input_list[i]["Orthogonal"])

        self.output_shape = []
        for i in range(self.view_size):
            input_item = self.input_list[i]
            self.layers = make_layer_list(
                arch[:-1], "Embedding_view" + str(i), spec_reg
            )
            self.layers += [
                {
                    "type": "tanh",
                    "size": n_clusters,
                    "l2_reg": spec_reg,
                    "name": "Embedding_"
                    + "view"
                    + str(i)
                    + "_{}".format(len(arch) - 1),
                },
                {"type": "Orthogonal", "name": "Orthogonal" + "_view" + str(i)},
            ]

            output = stack_layers(input_item, self.layers)
            self.output_shape.append(output["Embedding"])

        # [input1, input2 ..]  [output1, output2, ..]

        self.net = Model(
            inputs=self.input_shape["Embedding"], outputs=self.output_shape
        )

        # DEFINE LOSS

        loss_1 = 0

        # W
        for (x_train, siamese_net, input_shape, output) in zip(
            x_train_siam, siamese_list, self.input_list, self.output_shape
        ):
            # generate affinity matrix W according to params
            if affinity == "siamese":
                input_affinity = siamese_net.outputs["A"]
                x_affinity = siamese_net.predict(x_train, batch_sizes)
            elif affinity in ["knn", "full"]:
                input_affinity = input_shape["Embedding"]
                x_affinity = x_train

            # calculate scale for affinity matrix
            scale = get_scale(x_affinity, self.batch_sizes["Embedding"], scale_nbr)

            # create affinity matrix
            if affinity == "full":
                W = costs.full_affinity(input_affinity, scale=scale)
            elif affinity in ["knn", "siamese"]:
                W = costs.knn_affinity(
                    input_affinity, n_nbrs, scale=scale, scale_nbr=scale_nbr
                )
            # K.eval(W)
            # LOSS 1
            Dy = costs.squared_distance(output)
            # define loss
            loss_1 = loss_1 + K.sum(W * Dy) / (batch_sizes["Embedding"] ** 2)

        loss_1 = loss_1 / self.view_size

        # LOSS 2
        loss_2 = 0
        # generate permutation of different views
        ls = itertools.permutations(self.output_shape, 2)

        for (view_i, view_j) in ls:
            loss_2 += K.sum(costs.pairwise_distance(view_i, view_j)) / (
                batch_sizes["Embedding"]
            )

        self.loss = (1 - lamb) * loss_1 + lamb * loss_2 / (view_size ** 2)

        # create the train step update
        self.learning_rate = tf.Variable(0.0, name="spectral_net_learning_rate")
        self.train_step = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate
        ).minimize(self.loss, var_list=self.net.trainable_weights)
        # self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=self.net.trainable_weights)
        # initialize spectralnet variables
        K.get_session().run(tf.variables_initializer(self.net.trainable_weights))

    def train(
        self, x_train, x_val, lr, drop, patience, num_epochs, x_test=None, y_test=None
    ):
        # create handler for early stopping and learning rate scheduling
        self.lh = LearningHandler(
            lr=lr, drop=drop, lr_tensor=self.learning_rate, patience=patience
        )

        losses = np.empty((num_epochs,))
        val_losses = np.empty((num_epochs,))

        # begin spectralnet training loop
        self.lh.on_train_begin()
        for i in range(num_epochs):
            # train spectralnet
            losses[i] = train.train_step(
                return_var=[self.loss],
                updates=self.net.updates + [self.train_step],
                x_unlabeled=x_train,
                inputs=self.input_shape,
                batch_sizes=self.batch_sizes,
                batches_per_epoch=100,
            )[0]
            if x_val is not None:
                # get validation loss
                val_losses[i] = train.predict_sum(
                    self.loss,
                    x_unlabeled=x_val,
                    inputs=self.input_shape,
                    batch_sizes=self.batch_sizes,
                )

                # do early stopping if necessary
                if self.lh.on_epoch_end(i, val_losses[i]):
                    print("STOPPING EARLY")
                    return losses[:i]
                # print training status
                print(
                    "Epoch: {}, loss={:2f}, val_loss={:2f}".format(
                        i, losses[i], val_losses[i]
                    )
                )
            else:
                # do early stopping if necessary
                if self.lh.on_epoch_end(i, losses[i]):
                    print("STOPPING EARLY")
                    return losses[:i]
                # print training status
                print("Epoch: {}, loss={:2f}".format(i, losses[i]))

            if x_test is not None:
                if i % 30 == 0:
                    x_view_list = self.predict(x_test)
                    _, scores = Task.Clustering(
                        x_view_list, y_test, view_specific=False
                    )
                    scores = scores["kmeans"]
                    loss_ = np.round(losses[i], 4)
                    save_name = (
                        "epoch"
                        + str(i)
                        + "_loss_"
                        + str(loss_)
                        + "_ACC_"
                        + str(scores["accuracy"])
                        + "_f-mea_"
                        + str(scores["f_measure"])
                        + "_NMI_"
                        + str(scores["NMI"])
                    )
                    np.save(save_name + "_rep", x_view_list)

        return losses[:i]

    def predict(self, x):
        # test inputs do not require the 'Labeled' input

        return train.predict(
            self.output_shape,
            x_unlabeled=x,
            inputs=self.input_shape,
            batch_sizes=self.batch_sizes,
            view_size=self.view_size,
        )
