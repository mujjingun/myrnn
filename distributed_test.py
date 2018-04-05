import tensorflow as tf
c = tf.constant("Hello, distributed Tensorflow!")

cluster = tf.train.ClusterSpec({
    "worker": [
        "172.30.10.2:2222", # cspro2
        "172.30.10.4:2222", # cspro4
        "172.30.10.5:2222", # cspro5
        "172.30.10.8:2222", # cspro8
    ],
    "ps": [
        "172.30.10.11:2222", # cspro
    ]})


server = tf.train.Server(cluster, job_name="worker", task_index=2)
with tf.Session(server.target) as sess:
    print(sess.run(c))
