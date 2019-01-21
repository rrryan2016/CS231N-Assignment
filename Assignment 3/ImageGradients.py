# As usual, a bit of setup

import time, os, json
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

from cs231n.classifiers.pretrained_cnn import PretrainedCNN
from cs231n.data_utils import load_tiny_imagenet
from cs231n.image_utils import blur_image, deprocess_image

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Introducing TinyImageNet 
data = load_tiny_imagenet('cs231n/datasets/tiny-imagenet-100-A', subtract_mean=True)
# print (data.keys())

# introduce a TinyImageNet dataset and a deep CNN that has been pretrained on this dataset
# Use this pretrained model to compute gradients with respect to images, and use these image gradients to produce class saliency maps and fooling images

# TinyImageNet-100-A Classes 
# TinyImageNet dataset is a subset of the ILSVRC-2012 classification dataset.
# Consists of 200 object classes, and for each object class it provides 500 training images, 50 validation images, and 50 test images.
# All images have been downsampled to 64 x 64 pixels
# have provided labels for all training and validation images, but have withheld the labels for the test images
# We have further split the full TinyImageNet-100-A and TinyImageNet-100-B
# The full TinyImageNet-100-A dataset will take up about 250MB of disk space, and loading the full TinyImageNet-100-A dataset into memory will use about 2.8 GB of memory


# TinyImageNet-100-A classes
# based on the WordNet ontology.

#### Start: Show TinyImageNet ####

# for i, names in enumerate(data['class_names']):
#   print (i, ' '.join('"%s"' % name for name in names))

#### End: Show TinyImageNet ####


#### Start: Visualize Examples ####

# # Visualize some examples of the training data
# classes_to_show = 7
# examples_per_class = 5

# class_idxs = np.random.choice(len(data['class_names']), size=classes_to_show, replace=False)
# for i, class_idx in enumerate(class_idxs):
#   train_idxs, = np.nonzero(data['y_train'] == class_idx)
#   train_idxs = np.random.choice(train_idxs, size=examples_per_class, replace=False)
#   for j, train_idx in enumerate(train_idxs):
#     img = deprocess_image(data['X_train'][train_idx], data['mean_image'])
#     plt.subplot(examples_per_class, classes_to_show, 1 + i + classes_to_show * j)
#     if j == 0:
#       plt.title(data['class_names'][class_idx][0])
#     plt.imshow(img)
#     plt.gca().axis('off')

# plt.show()

#### End: Visualize Examples ####




#### Start: Pretrained model performance ####

# # This part will be used for several times
# # Pretrained Model
# # this deep CNN trained on TinyImageNet-100-A dataset
# # this model has 9 convolutional layers (with spatial batch normalization) and 1 fully-connected hidden layer(with batch normalization )
# model = PretrainedCNN(h5_file='cs231n/datasets/pretrained_model.h5')
# print (model)

# batch_size = 100

# # Test the model on training data
# mask = np.random.randint(data['X_train'].shape[0], size=batch_size)
# X, y = data['X_train'][mask], data['y_train'][mask]
# y_pred = model.loss(X).argmax(axis=1)
# print ('Training accuracy: ', (y_pred == y).mean())

# # Test the model on validation data
# mask = np.random.randint(data['X_val'].shape[0], size=batch_size)
# X, y = data['X_val'][mask], data['y_val'][mask]
# y_pred = model.loss(X).argmax(axis=1)
# print ('Validation accuracy: ', (y_pred == y).mean())

#### End: Pretrained model performance ####



from cs231n.layers import softmax_loss

#### Start: Saliency Maps ####

# # Uncomment the previous part above

# # using pretrained model, compute class saliency maps
# # compute the gradient of the image with respect to the unnormalized class score, not with respect to the normalized class probability
# def compute_saliency_maps(X, y, model):
#     """
#     Compute a class saliency map using the model for images X and labels y.

#     Input:
#     - X: Input images, of shape (N, 3, H, W)
#     - y: Labels for X, of shape (N,)
#     - model: A PretrainedCNN that will be used to compute the saliency map.

#     Returns:
#     - saliency: An array of shape (N, H, W) giving the saliency maps for the input
#     images.
#     """
#     saliency = None
#     ##############################################################################
#     # TODO: Implement this function. You should use the forward and backward     #
#     # methods of the PretrainedCNN class, and compute gradients with respect to  #
#     # the unnormalized class score of the ground-truth classes in y.             #
#     ##############################################################################
#     pass
#     scores, cache = model.forward(X)
#     loss, dscores = softmax_loss(scores, y)
#     dX, grads = model.backward(dscores, cache)
#     saliency = dX.max(axis=1)
#     print (X.shape, dX.shape)
#     ##############################################################################
#     #                             END OF YOUR CODE                               #
#     ##############################################################################
#     return saliency

# # Visualize some class saliency maps on the validation set of TinyImageNet-100-A

# def show_saliency_maps(mask):
#   mask = np.asarray(mask)
#   X = data['X_val'][mask]
#   y = data['y_val'][mask]

#   saliency = compute_saliency_maps(X, y, model)

#   for i in range(mask.size):
#     plt.subplot(2, mask.size, i + 1)
#     plt.imshow(deprocess_image(X[i], data['mean_image']))
#     plt.axis('off')
#     plt.title(data['class_names'][y[i]][0])
#     plt.subplot(2, mask.size, mask.size + i + 1)
#     plt.title(mask[i])
#     plt.imshow(saliency[i])
#     plt.axis('off')
#   plt.gcf().set_size_inches(10, 4)
#   plt.show()

# # Show some random images
# mask = np.random.randint(data['X_val'].shape[0], size=5)
# show_saliency_maps(mask)

# # These are some cherry-picked images that should give good results
# show_saliency_maps([128, 3225, 2417, 1640, 4619])

#### End: Saliency Maps ####



#### Start: Fooling Images ####

# Use image gradients to generate "fooling images"
# Given an iamge and a target class, we can perform gradient ascent over the image to maximize the target class
# stopping when the network
import itertools
def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image, of shape (1, 3, 64, 64)
    - target_y: An integer in the range [0, 100)
    - model: A PretrainedCNN

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    X_fooling = X.copy()
    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. Use gradient ascent on the target class score, using   #
    # the model.forward method to compute scores and the model.backward method   #
    # to compute image gradients.                                                #
    #                                                                            #
    # HINT: For most examples, you should be able to generate a fooling image    #
    # in fewer than 100 iterations of gradient ascent.                           #
    ##############################################################################
    for i in itertools.count():
        print (i)
        scores, cache = model.forward(X_fooling, mode='test')
        if scores[0].argmax() == target_y:
            break
        loss, dscores = softmax_loss(scores, target_y)
        dX, grads = model.backward(dscores, cache)
        X_fooling -= dX * 1000
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling


# Find a correctly classified validation image
# choose a random validation set image that is correctly classified by the network, and then make a fooling image.
while True:
  i = np.random.randint(data['X_val'].shape[0])
  X = data['X_val'][i:i+1]
  y = data['y_val'][i:i+1]
  y_pred = model.loss(X)[0].argmax()
  if y_pred == y: break

target_y = 20 #
X_fooling = make_fooling_image(X, target_y, model)

# Make sure that X_fooling is classified as y_target
scores = model.loss(X_fooling)
assert scores[0].argmax() == target_y, 'The network is not fooled!'


# Show original image, fooling image, and difference
plt.subplot(1, 3, 1)
plt.imshow(deprocess_image(X, data['mean_image']))
plt.axis('off')
plt.title(data['class_names'][int(y)][0])# TypeError: only integer scalar arrays can be converted to a scalar index
plt.subplot(1, 3, 2)
plt.imshow(deprocess_image(X_fooling, data['mean_image'], renorm=True))
plt.title(data['class_names'][target_y][0])
plt.axis('off')
plt.subplot(1, 3, 3)
plt.title('Difference')
plt.imshow(deprocess_image(X - X_fooling, data['mean_image']))
plt.axis('off')
plt.show()

#### End: Fooling Images ####
