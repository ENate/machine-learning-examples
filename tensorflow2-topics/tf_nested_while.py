import numpy as np
import tensorflow as tf


def tf_nested_while_loop(self, w2_ones):
    """A nested while loop in tensorflow 2

    Args:
        w2_ones (Tensor): A tensor array or matrix which is used to
        demonstrate an implementation of the nested while loop statement.
    Returns:
        [type]: [description]
    """
    i = tf.constant(0)
    while_condition = lambda i: tf.less(i, np.shape(w2_ones.numpy())[0])

    def body(i):
        # do something here which you want to do in your outer loop
        # print(w2[:, i] * 2)
        j = tf.constant(0)
        print("---------------------------------------")
        print(i)
        print("---------------------------------------")
        while_condition2 = lambda j, i: tf.less(j, np.shape(w2_ones.numpy())[1])

        def body2(j, i):
            print(w2_ones[i, j])
            # do anything you want in your inner loop!
            """
                with tf.GradientTape(persistent=True) as tape:
                    y_loss_fun, w2 = ArtificialDataset().example_nn()
                    y_loss = tf.reduce_sum(y_loss_fun ** 2)
                    w2_ones_id = tf.Variable(tf.identity(w2_ones))
                    w2_ones_id[i, j].assign(0.0)
                    # if i == j:
                    #    w2_ones[i, j].assign(0.0)
                dl_dw2 = tape.gradient(y_loss, w2_ones_id[i, j])
                del tape
                print(dl_dw2)
                """
            return j + 1, i

        # increment i
        j, i = tf.while_loop(while_condition2, body2, [j, i])
        return [tf.add(i, 1)]

    # do the loop
    r = tf.while_loop(while_condition, body, [i])
    return r
