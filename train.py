from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP
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
flags.DEFINE_integer('epochs', 2, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('sub_num', 50, 'Number of subgraphs')
flags.DEFINE_integer("inner_loop", 5, 'Number of loops inside the subgraph')
flags.DEFINE_integer("inter_loop", 2, 'Number of loops between different subgraphs')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
print(train_mask)
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

# Generate the file used in METIS

'''
vertices = len(support[0][0])
npadj = np.zeros((vertices, vertices))
edges = support[0][0]
edge_num = 0
newedge = []
for edge in edges:
    a = edge[0]
    b = edge[1]
    if (a != b):
        npadj[a][b] = 1
        npadj[b][a] = 1

for i in range(len(npadj[0])):
    for j in range(0, i + 1):
        if npadj[i][j] == 1:
            edge_num = edge_num + 1;

print(edge_num)
f = open("./graph", "w")
f.write(str(vertices) + " " + str(edge_num) + "\n")
for i in range(len(npadj[0])):
    for j in range(len(npadj[0])):
        if npadj[i][j] == 1:
            f.write(str(j + 1) + " ")
    f.write("\n")
f.close()
'''

f = open("./graph.part.50", "r")
ver_info = f.read().splitlines()
f.close()
vertices = len(ver_info)
subgraphs = []
for i in range(FLAGS.sub_num):
    subgraphs.append([])
for i in range(vertices):
    subid = int(ver_info[i])
    subgraphs[subid].append(i)

'''
maxvert = -1
minvert = 100000
for subgraph in subgraphs:
    if (len(subgraph) > maxvert):
        maxvert = len(subgraph)
    elif (len(subgraph) < minvert):
        minvert = len(subgraph)
print(maxvert)
print(minvert)
'''
'''
edge_weight = support[0][1]
edge_node = support[0][0]
ver_num = support[0][2][0]
sub_support = []

for i in range(FLAGS.sub_num):
    tmp = []
    sub_edge_node = []
    sub_edge_weight = []
    for j in range(len(edge_node)):
        if (edge_node[j][0] in subgraphs[i]) and (edge_node[j][1] in subgraphs[i]):
            sub_edge_node.append(edge_node[j])
            sub_edge_weight.append(edge_weight[j])
    tmp.append(np.array(sub_edge_node))
    tmp.append(np.array(sub_edge_weight))
    tmp.append((ver_num, ver_num))
    sub_support.append(tmp)

output = open('sub_support.pkl', 'wb')
pickle.dump(sub_support, output)
output.close()
'''
input = open('sub_support.pkl', 'rb')
sub_support = pickle.load(input)
print(sub_support)


# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
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
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
