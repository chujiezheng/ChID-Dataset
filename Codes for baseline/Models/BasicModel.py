import tensorflow as tf

class BasicModel(object):

    def __init__(self):
        self._create_placeholder()


    def _create_placeholder(self):
        self.document = tf.placeholder(tf.int32, [None, None])  # [batch, length]
        self.doc_length = tf.placeholder(tf.int32, [None])
        self.candidates = tf.placeholder(tf.int32, [None, None, None])  # [batch, num_labels, 10]
        self.labels = tf.placeholder(tf.float32, [None, None, None])  # [batch, labels, choices]
        # 1 for right, -1 for false, 0 for padding
        self.locations = tf.placeholder(tf.float32, [None, None, None])  # [batch, num_labels, seq_length], used to mask
        self.mask = tf.placeholder(tf.float32, [None, None])
        self.is_train = tf.placeholder(tf.bool)


    def _create_embedding(self, init_word_embed, init_idiom_embed):
        self.word_embed_matrix = tf.get_variable("word_embed_matrix", dtype=tf.float32, initializer=init_word_embed)
        self.idiom_embed_matrix = tf.get_variable("idiom_embed_matrix", dtype=tf.float32, initializer=init_idiom_embed)


    def _create_loss(self):
        cross_entropy = - tf.log(self.logits + 1e-12)
        valid_loss = tf.gather_nd(cross_entropy, tf.where(self.labels > 0))
        self.loss = tf.reduce_sum(valid_loss) / tf.cast(tf.shape(valid_loss), tf.float32)


        self.pred = tf.argmax(self.logits, -1)


    def _create_train_step(self, learning_rate, max_gradient_norm):
        self.lr = tf.Variable(learning_rate, trainable=False, dtype=tf.float32, name="learning_rate")
        self.lr_decay = self.lr.assign(self.lr * 0.95)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        opt = tf.train.AdamOptimizer(self.lr)
        self.params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, self.params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2, max_to_keep=3,
                                    pad_step_number=True, keep_checkpoint_every_n_hours=1.0)


    def print_parameters(self):
        for item in self.params:
            print("%s: %s" % (item.name, item.get_shape()))


    def train_step(self, sess, document, doc_length, candidates, labels, locations, is_train, mask):
        feed_in = {self.document: document, self.doc_length: doc_length, self.candidates: candidates, self.labels: labels,
                   self.locations: locations, self.mask: mask}
        feed_out = [self.loss, self.pred, self.logits]
        if is_train:
            feed_out.append(self.update)
            feed_in.update({self.is_train: True})
        else:
            feed_in.update({self.is_train: False})

        return sess.run(feed_out, feed_in)