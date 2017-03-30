import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from operator import mul

file.seek(3)
depth = int.from_bytes(file.read(1), byteorder='big')
dims = []
n = 1
for i in range(depth):
    d = int.from_bytes(file.read(4), byteorder='big')
    n = n * d
    dims.append(d)
all_bytes = file.read(n)
uints = np.frombuffer(all_bytes, dtype=np.uint8)


def read_targets(file):
    indices = read_idx_ubyte(file)
    (n,) = np.shape(indices)
    maxi = np.max(indices)
    one_hot = np.zeros((n, maxi + 1)).astype(np.float32)
    one_hot[np.arange(n), indices] = 1
    return one_hot


with open("train-labels-idx1.ubyte", "rb") as f:
    train_labels = read_targets(f)

with open("test-labels-idx1.ubyte", "rb") as f:
    test_labels = read_targets(f)


def read_images(file):
    pixels = read_idx_ubyte(file)
    (n, v, h, w) = np.shape(pixels)
    pixels_flt = pixels.astype(np.float32)
    for i in range(n):
        for j in range(v):
            im = np.reshape(pixels_flt[i, j, :, :], (h * w))
            im = (im - np.mean(im)) / np.std(im)
            pixels_flt[i, j, :, :] = np.reshape(im, (1, 1, h, w))
    return np.transpose(pixels_flt, (0, 2, 3, 1))


with open("train-images-idx4.ubyte", "rb") as f:
    train_images = read_images(f)

with open("test-images-idx4.ubyte", "rb") as f:
    test_images = read_images(f)

plt.figure()
(n_tests, h, w, n_views) = np.shape(test_images)
fig, axes = plt.subplots(1, n_views)
for i in range(n_views):
    axes[i].imshow(np.reshape(test_images[1, :, :, i], (h, w)))
    axes[i].set_title('View number ' + str(i + 1))

plt.tight_layout()
plt.savefig('images/premier_test.png')
_ = 'images/premier_test.png'


def define_convlayer_weights(size, n_views, n_layers):  # (ref:def_define_convlayer_weights)
    weights = tf.Variable(0.1 * tf.random_normal \
        ([size, size, n_views, n_layers]))
    bias = tf.Variable(tf.zeros((n_layers)))
    return (weights, bias)


def define_convlayer(input_data, w):  # (ref:def_define_convlayer)
    (weights, bias) = w
    conv_filter = tf.nn.conv2d(input_data, weights, \
                               strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.bias_add(conv_filter, bias)


def define_relulayer(input_data):
    return tf.nn.relu(input_data)


def define_maxpoolinglayer(input_data):
    shape = [1, 2, 2, 1]  # (ref:maxpooling-shape)
    return tf.nn.max_pool(input_data, ksize=shape, strides=shape, \
                          padding='SAME')


def define_fclayer_weights(input_shape, out_dim):
    n = int(reduce(mul, input_shape))
    weights = tf.Variable(tf.random_normal([n, out_dim]))
    bias = tf.Variable(tf.zeros((1, out_dim)))
    return (weights, bias)


def define_fclayer(in_data, w):
    (weights, bias) = w
    n_input = in_data.get_shape().as_list()[0]
    in_dim = weights.get_shape().as_list()[0]
    vector_input = tf.reshape(in_data, [n_input, in_dim])  # (ref:reshape-to-vector)
    bias_mat = tf.tile(bias, [n_input, 1])
    return tf.matmul(vector_input, weights) + bias_mat


I


def define_convrelu_weights(size, n_views, n_layers):
    return define_convlayer_weights(size, n_views, n_layers)


def define_convrelu(input_data, w):
    conv_filter = define_convlayer(input_data, w)
    return define_relulayer(conv_filter)


def define_triconvrelu_weights(size, n_views, n_layers):
    w1 = define_convrelu_weights(size, n_views, n_layers)
    w2 = define_convrelu_weights(size, n_views, n_layers)
    w3 = define_convrelu_weights(size, n_views, n_layers)
    return (w1, w2, w3)


def define_triconvrelu(input_data, w):
    (w1, w2, w3) = w
    exit_1 = define_convrelu(input_data, w1)
    exit_2 = define_convrelu(exit_1, w2)
    exit_3 = define_convrelu(exit_2, w3)
    return define_maxpoolinglayer(exit_3)


def define_graph_weights(data_shape, out_dim):
    n_items = data_shape[0]
    n = data_shape[1]
    p = data_shape[2]
    n_views = data_shape[3]
    tcr1 = define_triconvrelu_weights(5, n_views, n_views)
    tcr2 = define_triconvrelu_weights(5, n_views, n_views)
    tcr3 = define_triconvrelu_weights(5, n_views, n_views)
    fc = define_fclayer_weights((n / 8, p / 8, n_views), \
                                out_dim)
    return (tcr1, tcr2, tcr3, fc)


def define_graph(X, y, weights, reg):
    (tcr1, tcr2, tcr3, fc) = weights
    conv1 = define_triconvrelu(tf.constant(X), tcr1)
    conv2 = define_triconvrelu(conv1, tcr2)
    conv3 = define_triconvrelu(conv2, tcr3)
    prediction = define_fclayer(conv3, fc)
    error = tf.subtract(prediction, tf.constant(y))
    loss_functions = tf.nn.softmax(error)
    (w1, w2, w3, w4) = weights
    (w11, w12, w13) = w1
    (w21, w22, w23) = w2
    (w31, w32, w33) = w3
    (w4w, w4b) = w4
    (w11w, w11b) = w11
    (w12w, w12b) = w12
    (w13w, w13b) = w13
    (w21w, w21b) = w21
    (w22w, w22b) = w22
    (w23w, w23b) = w23
    (w31w, w31b) = w31
    (w32w, w32b) = w32
    (w33w, w33b) = w33
    regularization = reg * \
                     (tf.nn.l2_loss(w4w) + tf.nn.l2_loss(w4b) \
                      + tf.nn.l2_loss(w11w) + tf.nn.l2_loss(w11b) \
                      + tf.nn.l2_loss(w12w) + tf.nn.l2_loss(w12b) \
                      + tf.nn.l2_loss(w13w) + tf.nn.l2_loss(w13b) \
                      + tf.nn.l2_loss(w21w) + tf.nn.l2_loss(w21b) \
                      + tf.nn.l2_loss(w22w) + tf.nn.l2_loss(w22b) \
                      + tf.nn.l2_loss(w23w) + tf.nn.l2_loss(w23b) \
                      + tf.nn.l2_loss(w31w) + tf.nn.l2_loss(w31b) \
                      + tf.nn.l2_loss(w32w) + tf.nn.l2_loss(w32b) \
                      + tf.nn.l2_loss(w33w) + tf.nn.l2_loss(w33b))
    loss = tf.reduce_mean(loss_functions) + regularization
    return ((conv1, conv2, conv3, prediction), loss, prediction, error)


n_steps = 3000
learning_rate = 0.5
loss_amount = np.zeros((n_steps))
learned_weights = []


def learn(n_steps, learning_rate):
    loss_amount = np.zeros((n_steps))
    with tf.Session() as session:
        graph_weights = define_graph_weights \
            (train_images.shape, train_labels.shape[1])
        tf.global_variables_initializer().run()
        ((d1, d2, d3, d4), loss, prediction, error) = \
            define_graph(train_images, train_labels, graph_weights, 1e-3)
        descend = tf.train.GradientDescentOptimizer(learning_rate) \
            .minimize(loss)
        for i in range(n_steps):
            descend.run()
            loss_amount[i] = loss.eval()
            # print ("Attendu :")
            # print (train_labels)
            # print ("Prédiction :")
            # print (prediction.eval ())
            # print ("Erreur :")
            # print (error.eval ())
        (w1, w2, w3, w4) = graph_weights
        (w11, w12, w13) = w1
        (w21, w22, w23) = w2
        (w31, w32, w33) = w3
        (w4w, w4b) = w4
        (w11w, w11b) = w11
        (w12w, w12b) = w12
        (w13w, w13b) = w13
        (w21w, w21b) = w21
        (w22w, w22b) = w22
        (w23w, w23b) = w23
        (w31w, w31b) = w31
        (w32w, w32b) = w32
        (w33w, w33b) = w33
        e11 = (w11w.eval(), w11b.eval())
        e12 = (w12w.eval(), w12b.eval())
        e13 = (w13w.eval(), w13b.eval())
        e21 = (w21w.eval(), w21b.eval())
        e22 = (w22w.eval(), w22b.eval())
        e23 = (w23w.eval(), w23b.eval())
        e31 = (w31w.eval(), w31b.eval())
        e32 = (w32w.eval(), w32b.eval())
        e33 = (w33w.eval(), w33b.eval())
        e1 = (e11, e12, e13)
        e2 = (e21, e22, e23)
        e3 = (e31, e32, e33)
        e4 = (w4w.eval(), w4b.eval())
        return (loss_amount, (e1, e2, e3, e4))


(loss_amount, learned_weights) = learn(n_steps, learning_rate)

fig, ax = plt.subplots()
plt.plot(range(0, n_steps), loss_amount, c="g")
ax.set_ylabel("Error")
ax.set_xlabel("Number of steps")
fig.tight_layout()
plt.savefig('images/loss.png')
_ = 'images/loss.png'


def get_descriptors(images, labels, learned_weights):
    with tf.Session() as session:
        (l1, l2, l3, l4) = learned_weights

        def make_constant_tuple(t):
            return tuple(map(tf.constant, t))

        w1 = tuple(map(make_constant_tuple, l1))
        w2 = tuple(map(make_constant_tuple, l2))
        w3 = tuple(map(make_constant_tuple, l3))
        w4 = make_constant_tuple(l4)
        w = (w1, w2, w3, w4)
        ((d1, d2, d3, d4), loss, prediction, error) = \
            define_graph(images, labels, w, 0.001)
        tf.global_variables_initializer().run()
        data = (d1.eval(), d2.eval(), d3.eval(), d4.eval())
    return data


def distance(weights, descr_test, descr_train):
    n = weights.shape[0]
    s = 0
    for i in range(n):
        diff = np.subtract(descr_test[i], descr_train[i])
        s += weights[i] * np.linalg.norm(diff)
    return s


def rank(database, weights, test):
    db_1, db_2, db_3, db_4 = database
    n = db_1.shape[0]
    dist_to_test = np.zeros((n))
    for i in range(n):
        db = (db_1[i, :, :, :], db_2[i, :, :, :], \
              db_3[i, :, :, :], db_4[i, :])
        dist_to_test[i] = distance(weights, test, db)
    return np.argsort(dist_to_test)


def knn(database, k, weights, test):
    return rank(database, weights, test)[:k]


database = get_descriptors(train_images, train_labels, learned_weights)
X = np.array([test_images[1]])
y = np.array([test_labels[1]])
request = get_descriptors(X, y, learned_weights)
knn_weights = np.array([0.125, 0.125, 0.25, 0.5])
response = knn(database, 3, knn_weights, request)

fig = plt.figure()
(n_tests, h, w, n_views) = np.shape(test_images)
for i in range(3):
    for j in range(n_views):
        sp = fig.add_subplot(3, n_views, i * 3 + j + 1)
        img = train_images[response[i], :, :, j]
        img = np.reshape(img, (h, w))
        imgplot = plt.imshow(img)
        title = 'Choice ' + str(i + 1) + '/3: ' \
                + str(response[i]) + ', view ' + str(j + 1)
        sp.set_title(title)

fig.tight_layout()
fig.savefig('images/requete.png')
_ = 'images/requete.png'


def compute_precision(tp, fp, fn, tn):
    if tp + fp == 0:
        return (0)
    return (tp / (tp + fp))


def compute_recall(tp, fp, fn, tn):
    if tp + fn == 0:
        return (0)
    return (tp / (tp + fn))


def score(ranks):
    map = 0
    for i in range(ranks.shape[0]):
        reality = test_labels[i, :]
        last_recall = 0
        avep = 0
        for cutoff in range(ranks.shape[1]):
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for j in ranks[i, :cutoff]:
                other = train_labels[j, :]
                if np.dot(other, reality) > 0.99:
                    tp = tp + 1
                else:
                    fp = fp + 1
            for j in ranks[i, cutoff:]:
                other = train_labels[j, :]
                if np.dot(other, reality) > 0.99:
                    fn = fn + 1
                else:
                    tn = tn + 1

            precision = compute_precision(tp, fp, fn, tn)
            recall = compute_recall(tp, fp, fn, tn)
            # f = (2 * precision * recall) / (precision + recall)
            avep = avep + precision * (recall - last_recall)
        avep = avep / ranks.shape[1]
        map = map + avep
    map = map / ranks.shape[0]
    return map


n_test = test_labels.shape[0]
n_train = train_labels.shape[0]
ranks = np.zeros((n_test, n_train)).astype(np.uint8)
for i in range(n_test):
    X = np.array([test_images[i, :, :, :]])
    y = np.array([test_labels[i]])
    req = get_descriptors(X, y, learned_weights)
    ranks[i, :] = rank(database, knn_weights, req)

the_map = score(ranks)


