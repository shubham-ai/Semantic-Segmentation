


##  code for google colab

# import os

# from google.colab import drive
# drive.mount('/content/drive')

# %cd 'drive/My Drive/udacity files/CarND-Semantic-Segmentation'



import os.path
import tensorflow as tf

import helper

import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()
    
    input_layer = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return input_layer, keep_prob, layer3, layer4, layer7_out
tests.test_load_vgg(load_vgg, tf)

def conv_1x1(x,num_classes, k_size=(1,1), name='conv_1x1', strides=(1,1)):
    """
    1x1 convolution 
    """
    with tf.name_scope(name):
        initializer = tf.random_normal_initializer(stddev = 0.001 )

        conv_1x1_out = tf.layers.conv2d(x,num_classes,kernel_size=k_size, strides=strides, padding='same',kernel_initializer=initializer,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        tf.summary.histogram(name, conv_1x1_out)

        return conv_1x1_out
    
def upsampling(x,num_classes,kernel_size=5, strides=2, name='upsample'):
    with tf.name_scope(name):
        initializer =  tf.random_normal_initializer(stddev=0.01)
        upsampling_out = tf.layers.conv2d_transpose(x,num_classes,kernel_size=kernel_size, strides=strides,padding = 'SAME', kernel_initializer = initializer,kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
        tf.summary.histogram(name, upsampling_out)
        return upsampling_out 
def skip_layer(upsampling, convolution, name="skip_layer"):
    with tf.name_scope(name):
        skip = tf.add(upsampling, convolution)
        tf.summary.histogram(name,skip)
        return skip

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function 
    # regularizer is imp to reuduce overfitting and penalize the weights
    # its presiving the spatial information 
#     layer7a_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1,
#                 padding="same",kernal_regularizer = tf.contrib.layers.l2_relugarizer(1e-3))
#     # deconvlution  
#     #upsampling
#     output = tf.layer.conv2d_transpose(conv_1x1, num_classes, 4, 2,
#                 padding = 'same', kernal_regularizer = tf.contrib.layers.l2_relugarizer(1e-3))
    
    layer7_1x1 = conv_1x1(vgg_layer7_out,num_classes)
    layer7_upsampling = upsampling(layer7_1x1,num_classes,4,2)
    layer4_1x1 = conv_1x1(vgg_layer4_out,num_classes)
    layer4_skip = skip_layer(layer7_upsampling,layer4_1x1)
    layer4_upsampling = upsampling(layer4_skip,num_classes,4,2)
    layer3_1x1 = conv_1x1(vgg_layer3_out,num_classes)
    layer3_skip= skip_layer(layer4_upsampling,layer3_1x1)
    output = upsampling(layer3_skip, num_classes,16,8)
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # classification loss in scene understaing
    # ligits will be flattining the image 
    logits = tf.reshape(nn_last_layer,[-1, num_classes])
    labels = tf.reshape(correct_label,[-1,num_classes])
    corss_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(corss_entropy_loss)
#     softmax will do  Probablity 
#     cross_entorpy (labels should be match with the size)
#     adms optimizer
    return logits,train_op,corss_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    print("Training...")
    print()
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], 
                               feed_dict={input_image: image, correct_label: label,                                keep_prob: 0.5, learning_rate: 0.0009})
            print("Loss: = {:.3f}".format(loss))
        print()
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = "./data"
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll nee  d a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        correct_label = tf.placeholder(tf.int32,[None,None,None, num_classes],name="exact_label")
        learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        #layer from Vgg
        
        input_image, keep_prob, layer3_out, layer4_out, layer7_out= load_vgg(sess, vgg_path)
    
        
        #Creaste new layer
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        print("mighe be error")
        # create loss and optimizer operations.
        logits, train_op, corss_entropy_loss = optimize(layer_output,correct_label, learning_rate, num_classes)
                        
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Train NN using the train_nn function
        epochs=51

                              
        batch_size = 5
                              
        saver = tf.train.Saver()
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, corss_entropy_loss, input_image, correct_label, keep_prob, learning_rate)
        
        
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)



if __name__ == '__main__':
    run()






