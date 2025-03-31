import numpy as np
import tensorflow as tf

from utils import extract_time, random_generator, batch_generator
from components import (
    build_embedder, build_recovery,
    build_generator, build_supervisor,
    build_discriminator
)

def timegan(ori_data, parameters):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    no, seq_len, dim = np.asarray(ori_data).shape
    ori_time, max_seq_len = extract_time(ori_data)

    
    def min_max_scaler(data):
        min_val = np.min(np.min(data, axis=0), axis=0)
        max_val = np.max(np.max(data, axis=0), axis=0)
        scaled = (data - min_val) / (max_val + 1e-7)
        return scaled, min_val, max_val

    ori_data, min_val, max_val = min_max_scaler(ori_data)

    
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    z_dim = dim
    gamma = 1


    X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name='input_x')
    Z = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, z_dim], name='input_z')
    T = tf.compat.v1.placeholder(tf.int32, [None], name='input_t')

    
    H = build_embedder(X, T, module_name, hidden_dim, num_layers)
    X_tilde = build_recovery(H, T, module_name, hidden_dim, num_layers, dim)

    E_hat = build_generator(Z, T, module_name, hidden_dim, num_layers)
    H_hat = build_supervisor(E_hat, T, module_name, hidden_dim, num_layers)
    H_hat_supervise = build_supervisor(H, T, module_name, hidden_dim, num_layers)
    X_hat = build_recovery(H_hat, T, module_name, hidden_dim, num_layers, dim)

    Y_fake = build_discriminator(H_hat, T, module_name, hidden_dim, num_layers)
    Y_real = build_discriminator(H, T, module_name, hidden_dim, num_layers)
    Y_fake_e = build_discriminator(E_hat, T, module_name, hidden_dim, num_layers)

    
    e_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('embedder')]
    r_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('recovery')]
    g_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('generator')]
    s_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('supervisor')]
    d_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('discriminator')]

   
    reconstruction_loss = tf.losses.mean_squared_error(X, X_tilde)
    supervised_loss = tf.losses.mean_squared_error(H[:, 1:, :], H_hat_supervise[:, :-1, :])

    generator_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_fake, labels=tf.ones_like(Y_fake))
    )
    discriminator_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_real, labels=tf.ones_like(Y_real))
    ) + tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_fake, labels=tf.zeros_like(Y_fake))
    ) + gamma * tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_fake_e, labels=tf.zeros_like(Y_fake_e))
    )

    optimizer = tf.compat.v1.train.AdamOptimizer()
    embedder_step = optimizer.minimize(reconstruction_loss, var_list=e_vars + r_vars)
    generator_step = optimizer.minimize(generator_loss + supervised_loss, var_list=g_vars + s_vars)
    discriminator_step = optimizer.minimize(discriminator_loss, var_list=d_vars)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for _ in range(iterations):
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)

            sess.run(embedder_step, feed_dict={X: X_mb, T: T_mb})
            sess.run([generator_step, discriminator_step], feed_dict={X: X_mb, Z: Z_mb, T: T_mb})

        Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
        gen_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time})
        generated_data = [gen_curr[i, :ori_time[i], :] for i in range(no)]

    return np.array(generated_data) * max_val + min_val
