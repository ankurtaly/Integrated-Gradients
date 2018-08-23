import numpy as np
import tensorflow as tf


def load_model(model_path):
  '''Loads the Inception (v1) model and creates a TensorFlow session for it.'''
  graph = tf.Graph()
  cfg = tf.ConfigProto(gpu_options={'allow_growth':True})
  sess = tf.InteractiveSession(graph=graph, config=cfg)
  graph_def = tf.GraphDef.FromString(open(model_path).read())
  tf.import_graph_def(graph_def)
  return sess, graph


def load_labels_vocabulary(labels_path):
  # Load the labels vocabulary.
  return np.array(open(labels_path).read().split('\n'))


def T(graph, layer):
  '''Helper for getting layer output tensor'''
  return graph.get_tensor_by_name("import/%s:0" % layer)


def supplement_graph(graph):
  """Supplement the Inception graph with a gradients operator to compute the
  gradients for the prediction at a particular label (specified by a placeholder)
  with respect to the input.
  """
  with graph.as_default():
    label_index = tf.placeholder(tf.int32, [])
    inp = T(graph, 'input')
    label_prediction = T(graph, 'softmax2')[:, label_index]
    return inp, label_index, T(graph, 'softmax2'), tf.gradients(label_prediction, inp)[0]


def make_predictions_and_gradients(sess, graph):
  """Returns a function that can be used to obtain the predictions and gradients
  from the Inception network for a set of inputs. 
  
  The function is meant to be provided as an argument to the integrated_gradients
  method.
  """
  inp, label_index, predictions, grads = supplement_graph(graph)
  run_graph = sess.make_callable([predictions, grads], feed_list=[inp, label_index])
  def f(images, target_label_index):
    inception_mean = 117.0
    return run_graph([img - inception_mean for img in images], target_label_index)
  return f


def top_label_id_and_score(img, preds_and_grads_fn):
  '''Returns the label id and score of the object class that receives the highest SOFTMAX score.

     The provided image must of shape (224, 224, 3).
  '''
  # Evaluate the SOFTMAX output layer for the image and
  # determine the label for the highest-scoring class
  preds, _ = preds_and_grads_fn([img], 0)
  id = np.argmax(preds[0])
  return id, preds[0][id]
