#! SETUP 1 - DO NOT CHANGE, MOVE NOR COPY
import sys

import numpy

_snlp_book_dir = "./"
sys.path.append(_snlp_book_dir)
# docker image contains tensorflow 0.10.0rc0. We will support execution of only that version!
import statnlpbook.nn as nn
import tensorflow as tf
import nlp_util as util

# ! SETUP 2 - DO NOT CHANGE, MOVE NOR COPY
data_path = _snlp_book_dir + "data/nn/"
data_train = nn.load_corpus(data_path + "train.tsv")
data_dev = nn.load_corpus(data_path + "dev.tsv")
assert (len(data_train) == 45502)

glove_embeddings = util.load_glove_model("./data/glove_filtered_train_300d.txt")

train_stories, vocab, embed = util.pipeline(data_train,
                                            ext_embeddings=glove_embeddings)  # convert train set to integer IDs
dev_stories, _, _ = util.pipeline(data_dev, vocab)  # convert dev set to integer IDs

### MODEL PARAMETERS ###
vocab_size = len(vocab)
target_size = 5
embedding_size = 300
hidden_size = 1024


def scoring_fn(last_state):
    """

    :param last_state: [batch_size x hidden_size]
    :return: The score for all sentences at this timestep given the hidden state [batch_size x 5]
    """
    weight = score_weight
    bias = score_bias

    sents = tf.transpose(sentences_encoded_packed, perm=[0, 1, 2])  # batch_size x 5 x hidden_size
    score = tf.add(tf.transpose(tf.matmul(weight, last_state, transpose_b=True)), bias)
    score = tf.reshape(score, [-1, hidden_size, 1])  # tested, and works, makes [a, b, c] into [[a], [b], [c]]

    score = tf.batch_matmul(sents, score)  # batch_size x 5 x 1
    score = tf.reshape(score, [-1, 5])
    return score


def dec_loop_fun(last_state, idx):  # batch_size x hidden_size

    score = tf.nn.softmax(scoring_fn(last_state))
    best_indices = tf.cast(tf.arg_max(score, 1), tf.int32)
    cat_idx = tf.concat(0, [tf.range(0, tf.shape(sentences_encoded_packed, out_type=tf.int32)[0]), best_indices])
    gather_idx = tf.transpose(tf.reshape(cat_idx, [2, -1]))
    out = tf.gather_nd(sentences_encoded_packed, gather_idx)

    # TODO: don't re-use already used sentences

    return out


### MODEL ###
with tf.Graph().as_default():
    ## PLACEHOLDERS
    story = tf.placeholder(tf.int64, [None, None, None], "story")  # [batch_size x 5 x max_length]
    story_sent_lens = tf.placeholder(tf.int64, [None, None], "story_sent_lens")  # [batch_size x 5]
    order = tf.placeholder(tf.int32, [None, None], "order")  # [batch_size x 5]

    score_weight = tf.get_variable("W_score", shape=[hidden_size, hidden_size], dtype=tf.float64,
                                   initializer=tf.random_uniform_initializer(-0.1, 0.1))
    score_bias = tf.get_variable("b_score", shape=[hidden_size], dtype=tf.float64,
                                 initializer=tf.random_uniform_initializer(-0.1, 0.1))

    batch_size = tf.shape(story)[0]

    sentences = [tf.reshape(x, [batch_size, -1]) for x in
                 tf.split(1, target_size, story)]  # 5 times [batch_size x max_length]
    sentence_lens = tf.unstack(story_sent_lens, num=target_size, axis=1)

    # Word embeddings
    # initializer = tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float64)
    initializer = tf.constant_initializer(embed.as_np_array(embedding_size), dtype=tf.float64)
    embeddings = tf.get_variable("W", [vocab_size, embedding_size], initializer=initializer, dtype=tf.float64)

    sentences_embedded = [tf.nn.embedding_lookup(embeddings, sentence) for sentence in sentences]
    sentences_encoded = []  # batch_size x 5 x hidden_size

    with tf.variable_scope("sentence_encoder") as sent_enc_varscope:
        cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)

        for sent_emb, sent_lens in zip(sentences_embedded, sentence_lens):
            output, state = tf.nn.dynamic_rnn(cell, sent_emb, sequence_length=sent_lens, dtype=tf.float64)
            sentences_encoded.append(state.h)
            # reuse weights for all sentences
            sent_enc_varscope.reuse_variables()

    sentences_encoded_packed = tf.pack(sentences_encoded, axis=1)

    # reshape the orders into something that allows us to really reorder the sentences
    reorder_idx = tf.transpose(
        tf.reshape(tf.tile(tf.range(0, tf.shape(sentences_encoded_packed, out_type=tf.int32)[0]), [5]), [5, -1]))
    reorder_idx = tf.concat(1, [reorder_idx, order])
    reorder_idx = tf.reshape(reorder_idx, [-1, 2, 5])
    reorder_idx = tf.reshape(tf.pack(tf.split(split_dim=2, num_split=5, value=reorder_idx), axis=1), [-1, 5, 2])

    sentences_ordered = tf.gather_nd(sentences_encoded_packed, reorder_idx)

    sentences_encoded_ordered = tf.unstack(tf.gather(sentences_encoded, order), num=target_size, axis=1)
    sentences_encoded_ordered = [x[0] for x in sentences_encoded_ordered]  # 5 times batch_size x hidden_size
    sentences_encoded_ordered.insert(0, tf.zeros(tf.shape(sentences_encoded_ordered[0]), dtype=tf.float64))

    with tf.variable_scope("seq2seq") as seq2seq_varscope:
        cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)

        dec_outputs_train, _ = tf.nn.seq2seq.tied_rnn_seq2seq(sentences_encoded,
                                                              sentences_encoded_ordered, cell,
                                                              dtype=tf.float64)

        seq2seq_varscope.reuse_variables()
        sentences_encoded_test = [tf.identity(s) for s in sentences_encoded]
        sentences_encoded_test.insert(0, tf.zeros(tf.shape(sentences_encoded_test[0]), dtype=tf.float64))
        dec_outputs_test, _ = tf.nn.seq2seq.tied_rnn_seq2seq(sentences_encoded, sentences_encoded_test,
                                                             cell,
                                                             dtype=tf.float64,
                                                             loop_function=dec_loop_fun)

    # Scoring
    scores_train = []
    scores_test = []
    for dec_out_train, dec_out_test in zip(dec_outputs_train[:-1], dec_outputs_test[:-1]):
        # dec_outputs: batch_size x
        score_train = scoring_fn(dec_out_train)
        score_test = scoring_fn(dec_out_test)
        scores_train.append(score_train)
        scores_test.append(score_test)

    scores_train = tf.transpose(scores_train, [1, 0, 2])  # batch_size x 5 x 5
    scores_test = tf.transpose(scores_test, [1, 0, 2])  # batch_size x 5 x 5

    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(scores_train, order))
    opt_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    # prediction function
    unpacked_logits = [tensor for tensor in tf.unpack(scores_test, axis=1)]
    softmaxes = [tf.nn.softmax(tensor) for tensor in unpacked_logits]
    softmaxed_logits = tf.pack(softmaxes, axis=1)
    predict = tf.arg_max(softmaxed_logits, 2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        BATCH_SIZE = 25
        n = train_stories.num_stories()

        for epoch in range(100):
            print('----- Epoch', epoch + 1, '-----')
            train_stories.shuffle()
            total_loss = 0
            for i in range(n // BATCH_SIZE):
                cur_batch = train_stories.get_batch(i * BATCH_SIZE, (i + 1) * BATCH_SIZE)

                feed_dict = {story: cur_batch.as_np_sents(), order: cur_batch.as_np_orders(),
                             story_sent_lens: cur_batch.as_np_sent_lens()}
                _, current_loss = sess.run([opt_op, loss], feed_dict=feed_dict)
                total_loss += current_loss

            print(' Train loss:', total_loss / n)

            t_ords = train_stories.as_np_orders()
            train_feed_dict = {story: train_stories.as_np_sents(), order: t_ords,
                               story_sent_lens: train_stories.as_np_sent_lens()}
            train_predicted = sess.run(predict, feed_dict=train_feed_dict)
            train_accuracy = nn.calculate_accuracy(t_ords, train_predicted)
            print(' Train accuracy:', train_accuracy)

            d_ords = dev_stories.as_np_orders()
            dev_feed_dict = {story: dev_stories.as_np_sents(), order: d_ords,
                             story_sent_lens: dev_stories.as_np_sent_lens()}
            dev_predicted = sess.run(predict, feed_dict=dev_feed_dict)
            dev_accuracy = nn.calculate_accuracy(d_ords, dev_predicted)
            print(' Dev accuracy:', dev_accuracy)