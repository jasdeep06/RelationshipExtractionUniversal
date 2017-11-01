import tensorflow as tf
from tensorflow.contrib import rnn
from attention_model import attention
import numpy as np


class Model():
    def __init__(self,num_units,embedding_size,learning_rate,batch_size):

        self.num_units=num_units
        self.embedding_size=embedding_size
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        max_length=112
        num_classes=19
        vocab_size=21594


        with tf.name_scope("Placeholders"):
            self.sentence_vectors = tf.placeholder(tf.int64, [None, max_length], name="sentence_placeholder")
            self.label_vector = tf.placeholder(tf.int64, [None, num_classes], name="label_placeholder")
            self.seq_lengths = tf.placeholder(tf.int64, [None], name="seq_length_placeholder")



        with tf.name_scope("FC_Layer_weight_and_bias"):
            self.out_weight = tf.get_variable(name="out_weight", shape=[self.num_units, num_classes],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.out_bias = tf.get_variable(name="out_bias", shape=[num_classes], initializer=tf.zeros_initializer)

        embedding = tf.get_variable(name="embedding", shape=[vocab_size, self.embedding_size],
                                initializer=tf.contrib.layers.xavier_initializer())
        input = tf.nn.embedding_lookup(embedding, self.sentence_vectors)

        tf.summary.histogram("embedding_summary", embedding)

        tf.summary.histogram("out_weight_summary", self.out_weight)
        tf.summary.histogram("out_bias_summary", self.out_bias)

        cell_fw = rnn.LSTMCell(num_units=self.num_units, use_peepholes=True)
        cell_bw = rnn.LSTMCell(num_units=self.num_units, use_peepholes=True)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=input,
                                                      sequence_length=self.seq_lengths, dtype="float32")

        output_fw, output_bw = outputs
    # (batch_size,seq_length,num_units)
        net_output = output_fw + output_bw

        r_net, alpha = attention(net_output, 50, True)
        logits = tf.add(tf.matmul(r_net, self.out_weight), self.out_bias)

        # loss_function
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.label_vector))
        tf.summary.scalar("loss", self.loss)

        with tf.name_scope("train"):
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        # recall,recall_op=tf.metrics.recall(label_vector,logits)
        # f1=(2*(precision)*(recall))/(precision+recall)
        # model evaluation
        self.max_indices = tf.add(tf.argmax(logits, axis=1), 1)
        self.pred = tf.add(tf.argmax(self.label_vector, axis=1), 1)
