import sys
import tensorflow as tf
import logging
from typing import *
from tensorflow import Tensor
from tensorflow_addons.layers.embedding_bag import EmbeddingBag
from keras import Sequential, Model, layers, losses


logger = logging.getLogger("general_logger")

class DLRM(Model):
    """
    Customized DLRM model, using embedding bags to replace latent_factor.
    """

    def __init__(
            self,
            embed_dim,
            num_embed,
            embed_vocab_size,
            ln_bot,
            ln_top,
            # arch_interaction_op='dot',
            # arch_interaction_itself=False,
            sigmoid_bot=False,
            sigmoid_top=True,
            loss_func='bce',
            loss_threshold=0.0):
        '''
        embed_dim: the dimensionality of sparse feature embeddings
        num_embed: the size of sparse feature embeddings (num_instances)
        embed_vocab_size: the input vocab size of embedding bags
        ln_bot: the size of the bottom MLP
        ln_top: the size of the top MLP
        '''

        super(DLRM, self).__init__()

        self._num_embed = num_embed
        self._embed_dim = embed_dim
        self._embed_vocab_size = embed_vocab_size
        self._ln_bot = ln_bot
        self._ln_top = ln_top

        self._loss_threshold = loss_threshold
        self._loss_func = loss_func
        self._embedding_bag: List[EmbeddingBag] = [EmbeddingBag(input_dim=num,
                                            output_dim=self._embed_dim) for num in embed_vocab_size]
            
        # for eb in self._embedding_bag:
        #     print(eb.get_config())

        # self._mlp_bot = MLP(units_list=ln_bot,
        #                     out_activation='sigmoid' if sigmoid_bot else 'relu')
        # self._mlp_top = MLP(units_list=ln_top,
        #                     out_activation='sigmoid' if sigmoid_top else 'relu')

        self._mlp_bot: Sequential = Sequential(name="bottom_mlp")
        self._mlp_bot.add(layers.Dense(ln_bot, activation=None))

        self._mlp_top: Sequential = Sequential(name="top_mlp")
        self._mlp_top.add(layers.Dense(ln_top, activation=None))

        self._dot_interaction = None
        # if arch_interaction_op == 'dot':
        #     self._dot_interaction = SecondOrderFeatureInteraction(
        #         self_interaction=arch_interaction_itself
        #     )

        # elif arch_interaction_op != 'cat':
        #     sys.exit(
        #         "ERROR: arch_interaction_op="
        #         + arch_interaction_op
        #         + " is not supported"
        #     )

        if loss_func == 'mse':
            self._loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        elif loss_func == 'bce':
            self._loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        else:
            sys.exit(
                "ERROR: loss_func="
                + loss_func
                + " is not supported"
            )

    def get_weights(self):
        weights = {}
        weights["ln_bot"] = self._mlp_bot.weights
        weights["ln_top"] = self._mlp_top.weights
        weights["embedding_bags"] = {}
        for i, eb in enumerate(self._embedding_bag):
            weights["embedding_bags"][i] = eb.weights
        return weights

    def get_params(self):
        params = {}
        params["num_embed"] = self._num_embed
        params["embed_dim"] = self._embed_dim
        params["ln_bot"] = self._ln_bot
        params["ln_top"] = self._ln_top
        params["embed_vocab_size"] = self._embed_vocab_size
        return params

    def get_myloss(self, dense_features, sparse_features, label):
        '''
        dense_features shape: [batch_size, num of dense features]
        sparse_features shape: [batch_size, num_of_sparse_features]
        label shape: [batch_size]
        '''

        prediction = self.inference(dense_features, sparse_features)
        
        prediction = tf.reshape(prediction, [-1, 1])
        label = tf.reshape(label, [-1, 1])

        loss = self._loss(y_true=label,
                          y_pred=prediction)
        
        return loss

    def get_myloss_dist(self, dense_features: Tensor, sparse_features: Tensor, label: Tensor, global_batch_size: int):
        prediction = self.inference(dense_features, sparse_features)
        # print(tf.shape(prediction), tf.shape(label))
        prediction = tf.reshape(prediction, [-1, 1])
        label = tf.reshape(label, [-1, 1])

        per_example_loss = self._loss(label, prediction)
        # logger.info("per example loss:{l}".format(l=per_example_loss))
        # return per_example_loss
        
        loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
        # print(tf.shape(loss))
        return loss

    def call(self, inputs, training=None, mask=None):
        dense_features, sparse_features = inputs
        return self.inference(dense_features, sparse_features)

    def predict(self, inputs: tf.data.Dataset):
        inf_result: List[Tensor] = []
        for data in inputs:
            inf_result.append(self.inference(**data))
        return tf.stack(inf_result)

    def sparse_func(self, pair):
        return pair[1](pair[0])

    def inference(self, dense_features, sparse_features):
        '''
        dense_features shape: [batch_size, num of dense features]
        sparse_features shape: [batch_size, num_of_sparse_features]
        '''
        # print(dense_features, '\n', sparse_features)

        self._set_inputs([dense_features, sparse_features])
        # for i in zip(tf.unstack(sparse_features, axis=1), self._embedding_bag):
        #     print(i)
        sparse_emb_vecs = list(map(self.sparse_func,
                                   zip(tf.unstack(sparse_features, axis=1),
                                       self._embedding_bag)))

        dense_emb_vec = self._mlp_bot(dense_features)

        if self._dot_interaction is not None:
            prediction = self._mlp_top(tf.concat([dense_emb_vec,
                                                  self._dot_interaction(sparse_emb_vecs + [dense_emb_vec])],
                                                 axis=1))
        else:
            prediction = self._mlp_top(tf.concat(sparse_emb_vecs + [dense_emb_vec],
                                                 axis=1))

        out = tf.sigmoid(tf.reduce_mean(prediction, axis=1))

        # if 0.0 < self._loss_threshold and self._loss_threshold < 1.0:
        #     prediction = tf.clip_by_value(
        #         prediction, self._loss_threshold, 1.0 - self._loss_threshold)
        # print(out)

        return tf.reshape(out, [-1])
