import tensorflow as tf

class Node(object):
    def __init__(self, sess, n_features, n_classes):
        self.sess = sess

        self.X_ = tf.placeholder(tf.float32, shape=[None, n_features])
        self.y_ = tf.placeholder(tf.float32, shape=[None, n_classes])
        self.w1 = tf.Variable(tf.random_uniform([n_features, n_classes]), name="w1")

        z1 = tf.matmul(self.X_, self.w1)
        probs = tf.nn.softmax(z1)

        self.loss = tf.losses.log_loss(labels=self.y_, predictions=probs)
        self.op = tf.train.AdamOptimizer(0.01).minimize(self.loss)
        
        self.grad = tf.gradients(self.loss, self.w1)
        
        predictions = tf.argmax(probs, 1)
        correct_prediction = tf.equal(predictions, tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
    def fetch_grad(self, w, X, y):
        """
        given weight w_i and data {X_i, y_i}, compute gradient.
        """
        self.w1.load(w, self.sess)
        return self.sess.run(self.grad, feed_dict={self.X_:X, self.y_:y})[0]
        
    def fit(self, X, y, epochs):
        for epoch in range(epochs):
            _ = self.sess.run(self.op, feed_dict={self.X_:X, self.y_:y})
            
    def score(self, X, y):
        return self.sess.run(self.accuracy, feed_dict={self.X_:X, self.y_:y})

    def fetch_weights(self):
        return self.sess.run(self.w1)
    
    def load(self, w):
        self.w1.load(w, self.sess)