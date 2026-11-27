import tensorflow as tf

################################# loss functions ##########################################################

def denoise_loss_mse(denoise, clean):      
  # Force both tensors to be float32 before subtracting
  d = tf.cast(denoise, tf.float32)
  c = tf.cast(clean, tf.float32)
  return tf.reduce_mean(tf.square(d - c))

def denoise_loss_rmse(denoise, clean):      #tmse
  d = tf.cast(denoise, tf.float32)
  c = tf.cast(clean, tf.float32)
  loss = tf.reduce_mean(tf.square(d - c))
  return tf.math.sqrt(loss)

def denoise_loss_rrmset(denoise, clean):      #tmse
  rmse1 = denoise_loss_rmse(denoise, clean)
  c = tf.cast(clean, tf.float32)
  rmse2 = tf.math.sqrt(tf.reduce_mean(tf.square(c)))
  return rmse1 / rmse2
 