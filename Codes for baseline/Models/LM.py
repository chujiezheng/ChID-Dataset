import tensorflow as tf
from Models.BasicModel import BasicModel

class Model(BasicModel):
    def __init__(self,
                 learning_rate,
                 init_word_embed,
                 init_idiom_embed,
                 size_embed=200,
                 num_units=100, # make sure that num_units = size_embed / 2
                 max_gradient_norm=5.0):

        assert size_embed == 2 * num_units

        super(Model, self).__init__()
        super(Model, self)._create_embedding(init_word_embed, init_idiom_embed)

        doc_embedding = tf.cond(self.is_train,
                                lambda: tf.nn.dropout(tf.nn.embedding_lookup(self.word_embed_matrix, self.document), 0.5),
                                lambda: tf.nn.embedding_lookup(self.word_embed_matrix, self.document))
        # [batch, length, size_embed]
        can_embedding = tf.nn.embedding_lookup(self.idiom_embed_matrix, self.candidates)  # [batch, num_labels, 10, size_embed]

        with tf.variable_scope("doc"):
            cell_fw_doc = tf.nn.rnn_cell.LSTMCell(num_units, initializer=tf.orthogonal_initializer())
            cell_bw_doc = tf.nn.rnn_cell.LSTMCell(num_units, initializer=tf.orthogonal_initializer())
            h_doc, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw_doc, cell_bw_doc, doc_embedding, self.doc_length, dtype=tf.float32,
                                                       scope="bi_lstm")
            state_doc = tf.concat(h_doc, 2)  # [batch, length, 2 * num_units]

        blanks_states = tf.matmul(self.locations, state_doc)  # [batch, labels, 2 * num_units]
        match_matrix = tf.einsum("abcd,abd->abc", can_embedding, blanks_states)  # [batch, labels, 10]
        self.logits = tf.nn.softmax(match_matrix)

        super(Model, self)._create_loss()
        super(Model, self)._create_train_step(learning_rate, max_gradient_norm)