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
        can_embedding = tf.nn.embedding_lookup(self.idiom_embed_matrix, self.candidates)  # [batch, 10, size_embed]

        with tf.variable_scope("doc"):
            cell_fw_doc = tf.nn.rnn_cell.LSTMCell(num_units, initializer=tf.orthogonal_initializer())
            cell_bw_doc = tf.nn.rnn_cell.LSTMCell(num_units, initializer=tf.orthogonal_initializer())
            h_doc, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw_doc, cell_bw_doc, doc_embedding, self.doc_length,
                                                       dtype=tf.float32, scope="bi_lstm")
            state_doc = tf.concat(h_doc, 2) # [batch, length, 2 * num_units]

        temp_blanks_states = tf.matmul(self.locations, state_doc) # query, [batch, labels, 2 * num_units]
        tile_blanks_states = tf.tile(tf.expand_dims(temp_blanks_states, 2), [1, 1, tf.shape(state_doc)[1], 1])
        tile_state_doc = tf.tile(tf.expand_dims(state_doc, 1), [1, tf.shape(temp_blanks_states)[1], 1, 1])
        # [batch, labels, length, 2 * num_units]
        Wym = tf.get_variable("ym_weight", [2 * num_units, 2 * num_units], tf.float32)
        Wum = tf.get_variable("um_weight", [2 * num_units, 2 * num_units], tf.float32)
        wms = tf.get_variable("ms_weight", [2 * num_units], tf.float32)

        mt = tf.nn.tanh(tf.einsum("abcd,de->abce", tile_state_doc, Wym) +
                        tf.einsum("abcd,de->abce", tile_blanks_states, Wum))
        attention_matrix = tf.einsum("abcd,d->abc", mt, wms)

        tmp = tf.exp(attention_matrix) * tf.tile(tf.expand_dims(self.mask, axis=1), [1, tf.shape(temp_blanks_states)[1], 1])
        attention = tf.div(tmp, tf.reduce_sum(tmp, axis=-1, keep_dims=True))
        state_attention = tf.matmul(attention, state_doc) # [batch, labels, 2 * num_units]

        Wrg = tf.get_variable("rg_weight", [2 * num_units, 2 * num_units], tf.float32)
        Wug = tf.get_variable("ug_weight", [2 * num_units, 2 * num_units], tf.float32)
        complete = tf.nn.tanh(tf.einsum("abd,de->abe", state_attention, Wrg) +
                              tf.einsum("abd,de->abe", temp_blanks_states, Wug))

        match_matrix = tf.matmul(complete, tf.transpose(can_embedding, [0, 2, 1])) # [batch, labels, 10]
        self.logits = tf.nn.softmax(match_matrix)

        super(Model, self)._create_loss()
        super(Model, self)._create_train_step(learning_rate, max_gradient_norm)