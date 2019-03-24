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
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 10, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('sub_num', 15, 'Number of subgraphs')
flags.DEFINE_integer("inner_loop", 5, 'Number of loops inside the subgraph')
flags.DEFINE_integer("inter_loop", 2, 'Number of loops between different subgraphs')
flags.DEFINE_integer('node_num', 2708, 'Number of nodes')
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

'''
genMETIS(support)
'''

sub_mask, sub_graphs, test_mask, train_mask = loadGraph(FLAGS.sub_num, support)
y_test = np.zeros(labels.shape)
y_test[test_mask, :] = labels[test_mask, :]

#saveSubSupport(support, FLAGS.sub_num, sub_graphs)


x = open(str(FLAGS.sub_num) + '.sub_support.pkl', 'rb')
sub_support = pickle.load(x)
'''
for i in range(FLAGS.sub_num):
    for j in range(FLAGS.sub_num):
        if len(sub_support[i][j][0]) == 0:
            sub_support[i][j] = (np.array([]), np.array([]), (2708, 2708))
'''
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

# Train model using subgraphs
for epoch in range(FLAGS.epochs):
    t = time.time()
    for i in range(FLAGS.sub_num):
        sync_train = np.zeros(labels.shape)
        sync_train[train_mask, :] = labels[train_mask, :]
        sub_train = np.zeros(labels.shape)
        sub_train[sub_mask[i], :] = labels[sub_mask[i], :]
        inter_support = []
        for x in range(FLAGS.sub_num):
            if x != i:
                inter_support.append(sub_support[x][i])
        con_edge = []
        con_weight = []
        for tmp_sup in inter_support:
            edge_nodes = tmp_sup[0]
            edge_weight = tmp_sup[1]
            for edge in edge_nodes:
                con_edge.append(edge)
            for w in edge_weight:
                con_weight.append(w)
        con_support = [(np.array(con_edge), np.array(con_weight), (FLAGS.node_num, FLAGS.node_num))]
        self_support = [sub_support[i][i]]
        # Inner loop

        for loop in range(FLAGS.inner_loop):
            feed_dict = construct_feed_dict(features, self_support, sub_train, sub_mask[i], placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Training step
            outs1 = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Inter loop
    for _ in range(FLAGS.inter_loop):
        feed_dict = construct_feed_dict(features, support, sync_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Training step
        outs2 = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    print("Inter, ", "Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs2[1]),
              "train_acc=", "{:.5f}".format(outs2[2]), "time=", "{:.5f}".format(time.time() - t))
    print("Inner, ", "Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs1[1]),
                "train_acc=", "{:.5f}".format(outs1[2]), "time=", "{:.5f}".format(time.time() - t))

test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))