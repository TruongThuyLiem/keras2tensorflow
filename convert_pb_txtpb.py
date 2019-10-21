import tensorflow as tf

#from google.protobuf import text_format
from tensorflow.python.platform import gfile

def converter(filename): 
  with gfile.FastGFile(filename,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    tf.train.write_graph(graph_def, 'pbtxt/', 'protobuf.pbtxt', as_text=True)
    print(graph_def)
  return


#converter('ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb')  # here you can write the name of the file to be converted
# and then a new file will be made in pbtxt directory.

converter('./output_graph.pb')
