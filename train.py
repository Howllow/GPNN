from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import GCN, MLP
from dataProcessing import *
import pickle
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'pubmed', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 280, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('sub_num', 4, 'Number of subgraphs')
flags.DEFINE_integer("inner_loop", 4, 'Number of loops inside the subgraph')
flags.DEFINE_integer("inter_loop", 1, 'Number of loops between different subgraphs')
flags.DEFINE_integer('train_method', 1, '0 stands for sub, 1 stands for classic')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data(FLAGS.dataset)
# Some preprocessing
features = preprocess_features(features)

if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

#genMETIS(FLAGS.dataset, support)

sub_mask, sub_graphs, test_mask, train_mask, sub_reorder = loadGraph(FLAGS.dataset, FLAGS.sub_num, support)
relabels = np.array([labels[i] for i in sub_reorder])
#reorder_sup_feat(FLAGS.dataset, FLAGS.sub_num, support, features, sub_reorder)
m = open(str(FLAGS.sub_num) + '.' + FLAGS.dataset +'.reorder_res.pkl', 'rb')
reorder_list = pickle.load(m)
refeatures = reorder_list[1]
resupport = reorder_list[0]
y_test = np.zeros(labels.shape)
y_test[test_mask, :] = labels[test_mask, :]


#saveInnerSupport(FLAGS.dataset, support, FLAGS.sub_num, sub_graphs)


x = open(str(FLAGS.sub_num) + '.' + FLAGS.dataset + '.inner_support.pkl', 'rb')
inner_support = pickle.load(x)

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
}
# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)
# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter("logs/", sess.graph)
cost_val = []

'''
# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration) )
'''

label_train = np.zeros(labels.shape)
label_train[train_mask, :] = labels[train_mask, :]

if FLAGS.train_method == 0:
    # Train model using subgraphs
    iters = FLAGS.epochs
    while iters > 0:
        t = time.time()

        # Inner loop
        for loop in range(FLAGS.inner_loop):
            iters = iters - 1
            feed_dict = construct_feed_dict(features, inner_support, label_train, train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Training step
            outs1 = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        # Inter loop
        for _ in range(FLAGS.inter_loop):
            iters = iters - 1
            feed_dict = construct_feed_dict(features, support, label_train, train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Training step
            outs2 = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        print("Sub, ", "Iterations:", '%04d' % (FLAGS.epochs - iters), "train_loss=", "{:.5f}".format(outs2[1]),
                  "train_acc=", "{:.5f}".format(outs2[2]), "time=", "{:.5f}".format(time.time() - t))

    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

else:
    for epoch in range(FLAGS.epochs):
        t = time.time()
        feed_dict = construct_feed_dict(features, support, label_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        if (epoch + 1) % 5 == 0:
            print("Classic, ", "Iterations:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "time=", "{:.5f}".format(time.time() - t))

    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
