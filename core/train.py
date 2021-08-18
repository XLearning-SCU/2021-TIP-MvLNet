import numpy as np
from keras import backend as K

from .util import make_batches


def check_inputs(x_unlabeled, x_labeled, y_labeled, y_true):
    """
    Checks the data inputs to both train_step and predict and creates
    empty arrays if necessary
    """
    if x_unlabeled is None:
        if x_labeled is None:
            raise Exception("No data, labeled or unlabeled, passed to check_inputs!")
        x_unlabeled = x_labeled[0:0]
    if x_labeled is not None and y_labeled is not None:
        pass
    elif x_labeled is None and y_labeled is None:
        x_labeled = x_unlabeled[0:0]
        y_shape = y_true.get_shape()[1 : K.ndim(y_true)].as_list()
        y_labeled = np.empty([0] + y_shape)
    else:
        raise Exception("x_labeled and y_labeled must both be None or have a value")
    return x_unlabeled, x_labeled, y_labeled


def train_step(
    return_var, updates, x_unlabeled, inputs, batch_sizes, batches_per_epoch=100
):
    """
    Performs one training step. Evaluates the tensors in return_var and
    updates, then returns the values of the tensors in return_var.

    return_var:         list of tensors to evaluate and return
    updates:            list of tensors to evaluate only
    x_unlabeled:        unlabeled input data
    inputs:             dictionary containing input_types and
                        input_placeholders as key, value pairs, respectively
    y_true:             true labels placeholder
    batch_sizes:        dictionary containing input_types and batch_sizes as
                        key, value pairs, respectively
    x_labeled:          labeled input data
    y_labeled:          labeled input labels
    batches_per_epoch:  parameter updates per epoch*

    returns:    the evaluated result of all tensors in return_var, summed
                across all epochs

    *note: the term epoch is used loosely here, it does not necessarily
           refer to one iteration over the entire dataset. instead, it
           is just batches_per_epoch parameter updates.
    """
    return_vars_ = np.zeros(shape=(len(return_var)))
    # train batches_per_epoch batches
    for batch_num in range(0, batches_per_epoch):
        feed_dict = {K.learning_phase(): 1}
        # feed corresponding input for each input_type
        for input_type, input_placeholder in inputs.items():
            batch_ids = np.random.choice(
                len(x_unlabeled[0]), size=batch_sizes[input_type], replace=False
            )
            for i in range(len(input_placeholder)):
                feed_dict[input_placeholder[i]] = x_unlabeled[i][batch_ids]

        all_vars = return_var + updates
        return_vars_ += np.asarray(
            K.get_session().run(all_vars, feed_dict=feed_dict)[: len(return_var)]
        )

    return return_vars_


def evaluate(predict_var, x_unlabeled, inputs, batch_sizes):
    """
    Evaluates predict_var, batchwise, over all points in x_unlabeled
    and x_labeled.

    predict_var:        list of tensors to evaluate and return
    x_unlabeled:        unlabeled input data
    inputs:             dictionary containing input_types and
                        input_placeholders as key, value pairs, respectively
    y_true:             true labels tensorflow placeholder
    batch_sizes:        dictionary containing input_types and batch_sizes as
                        key, value pairs, respectively
    x_labeled:          labeled input data
    y_labeled:          labeled input labels

    returns:    a list of length n containing the result of all tensors
                in return_var, where n = len(x_unlabeled) + len(x_labeled)
    """
    x = x_unlabeled

    # calculate batches for predict loop
    unlabeled_batch_size = batch_sizes.get("Embedding", 0)
    batch_size = min(len(x[0]), unlabeled_batch_size)
    batches = make_batches(len(x[0]), batch_size)

    y_preds = []
    # predict over all points
    for j, (batch_start, batch_end) in enumerate(batches):
        feed_dict = {K.learning_phase(): 0}
        # feed corresponding input for each input_type
        for input_type, input_placeholder in inputs.items():
            if input_type == "Embedding":
                for i in range(len(input_placeholder)):
                    feed_dict[input_placeholder[i]] = x[i][batch_start:batch_end]
            elif input_type == "Orthogonal":
                batch_ids = np.random.choice(
                    len(x), size=min(len(x), batch_sizes[input_type]), replace=False
                )
                for i in range(len(input_placeholder)):
                    feed_dict[input_placeholder[i]] = x[i][batch_ids]
            else:
                raise Exception("Unrecognized feed name ['{}']".format(input_type))
        # evaluate the batch
        y_pred_batch = np.asarray(K.get_session().run(predict_var, feed_dict=feed_dict))
        y_preds.append(y_pred_batch)

    if len(y_preds[0].shape):
        return np.concatenate(y_preds)
    else:
        return np.sum(y_preds)


def predict(predict_var, x_unlabeled, inputs, batch_sizes, view_size):
    """
    Evaluates predict_var, batchwise, over all points in x_unlabeled
    and x_labeled.

    predict_var:        list of tensors to evaluate and return
    x_unlabeled:        unlabeled input data
    inputs:             dictionary containing input_types and
                        input_placeholders as key, value pairs, respectively
    y_true:             true labels tensorflow placeholder
    batch_sizes:        dictionary containing input_types and batch_sizes as
                        key, value pairs, respectively
    x_labeled:          labeled input data
    y_labeled:          labeled input labels

    returns:    a list of length n containing the result of all tensors
                in return_var, where n = len(x_unlabeled) + len(x_labeled)
    """
    x = x_unlabeled

    # calculate batches for predict loop
    unlabeled_batch_size = batch_sizes.get("Embedding", 0)
    batch_size = min(len(x[0]), unlabeled_batch_size)
    batches = make_batches(len(x[0]), batch_size)

    y_preds = []
    # predict over all points
    for j, (batch_start, batch_end) in enumerate(batches):
        feed_dict = {K.learning_phase(): 0}
        # feed corresponding input for each input_type
        for input_type, input_placeholder in inputs.items():
            if input_type == "Embedding":
                for i in range(view_size):
                    feed_dict[input_placeholder[i]] = x[i][batch_start:batch_end]
            elif input_type == "Orthogonal":
                batch_ids = np.random.choice(
                    len(x), size=min(len(x), batch_sizes[input_type]), replace=False
                )
                for i in range(view_size):
                    feed_dict[input_placeholder[i]] = x[i][batch_ids]
            else:
                raise Exception("Unrecognized feed name ['{}']".format(input_type))
        # evaluate the batch
        y_pred_batch = np.asarray(K.get_session().run(predict_var, feed_dict=feed_dict))
        y_preds.append(y_pred_batch)
    y_list = np.concatenate(y_preds, axis=1)

    return y_list


def predict_sum(predict_var, x_unlabeled, inputs, batch_sizes):
    """
    Convenience function: sums over all the points to return a single value
    per tensor in predict_var
    """
    y = evaluate(predict_var, x_unlabeled, inputs, batch_sizes)
    return np.sum(y)


def predict_siamese(predict_var, x, inputs, batch_sizes):
    """
    Evaluates predict_var, batchwise, over all points in x_unlabeled
    and x_labeled.

    predict_var:        list of tensors to evaluate and return
    x:                  unlabeled input data
    inputs:             dictionary containing input_types and
                        input_placeholders as key, value pairs, respectively
    batch_sizes:        dictionary containing input_types and batch_sizes as
                        key, value pairs, respectively

    returns:    a list of length n containing the result of all tensors
                in return_var, where n = len(x_unlabeled) + len(x_labeled)
    """

    # calculate batches for predict loop
    unlabeled_batch_size = batch_sizes.get("Embedding", 0)
    batch_size = min(len(x), unlabeled_batch_size)
    batches = make_batches(len(x), batch_size)

    y_preds = []
    # predict over all points
    for i, (batch_start, batch_end) in enumerate(batches):
        feed_dict = {K.learning_phase(): 0}
        # feed corresponding input for each input_type
        for input_type, input_placeholder in inputs.items():
            if input_type == "Embedding":
                feed_dict[input_placeholder] = x[batch_start:batch_end]
            elif input_type == "Orthogonal":
                batch_ids = np.random.choice(
                    len(x), size=min(len(x), batch_sizes[input_type]), replace=False
                )
                feed_dict[input_placeholder] = x[batch_ids]
            else:
                raise Exception("Unrecognized feed name ['{}']".format(input_type))

        # evaluate the batch
        y_pred_batch = np.asarray(K.get_session().run(predict_var, feed_dict=feed_dict))
        y_preds.append(y_pred_batch)

    if len(y_preds[0].shape):
        return np.concatenate(y_preds)
    else:
        return np.sum(y_preds)
