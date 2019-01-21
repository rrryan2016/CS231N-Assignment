import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

data = get_CIFAR10_data()

#### Start: Print loaded dataset ####

# for k, v in data.items():
#   print('%s: ' % k, v.shape)

#### End: Print loaded dataset ####


#### Start: Test the affine_forward function in layers.py ####

# # Receive inputs, weight, and other parameters and will return both an output and a 'cache' object storing data needed for the backward pass.
# num_inputs = 2
# input_shape = (4, 5, 6)
# # print("check * : ",*input_shape , "  type ",type(input_shape))
# output_dim = 3
#
# input_size = num_inputs * np.prod(input_shape)
# weight_size = output_dim * np.prod(input_shape)
#
# x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
# w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
# b = np.linspace(-0.3, 0.1, num=output_dim)
#
# out, _ = affine_forward(x, w, b)
# correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
#                         [ 3.25553199,  3.5141327,   3.77273342]])
#
# # Compare your output with mine. The error should be around 1e-9.
# print ('Testing affine_forward function:')
# print ('difference: ', rel_error(out, correct_out))

#### End: Test the affine_forward function in layers.py ####


# #### Start: Test the affine_backward function ####

# # Receive upstream derivatives and 'cache' object, and will return gradients with respect to the inputs and weights .
# x = np.random.randn(10, 2, 3)
# w = np.random.randn(6, 5)
# b = np.random.randn(5)
# dout = np.random.randn(10, 5)

# dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
# dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
# db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)

# _, cache = affine_forward(x, w, b)
# dx, dw, db = affine_backward(dout, cache)

# # The error should be around 1e-10
# print('Testing affine_backward function:')
# print('dx error: ', rel_error(dx_num, dx))
# print('dw error: ', rel_error(dw_num, dw))
# print('db error: ', rel_error(db_num, db))

# #### End: Test the affine_backward function ####


# #### Start: Test the relu_forward function ####

# # Edit relu_forward(x) in layers.py
# x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

# out, _ = relu_forward(x)
# correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
#                         [ 0.,          0.,          0.04545455,  0.13636364,],
#                         [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])

# print(x.shape, out.shape, correct_out.shape)# Compare your output with ours. The error should be around 1e-8
# print('Testing relu_forward function:')
# print('difference: ', rel_error(out, correct_out))

# #### End: Test the relu_forward function ####


#### Start: Test the relu_backward function ####

# x = np.random.randn(10, 10)
# dout = np.random.randn(*x.shape)

# dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)

# _, cache = relu_forward(x)
# dx = relu_backward(dout, cache)

# # The error should be around 1e-12
# print ('Testing relu_backward function:')
# print ('dx error: ', rel_error(dx_num, dx))

#### End: Test the relu_forward function ####


# Sandwich

# #### Start: Test several convenient layers in layer_utils.py ####

# # take a look at affine_relu_forward and affine_relu_backward
# from cs231n.layer_utils import affine_relu_forward, affine_relu_backward

# x = np.random.randn(2, 3, 4)
# w = np.random.randn(12, 10)
# b = np.random.randn(10)
# dout = np.random.randn(2, 10)

# out, cache = affine_relu_forward(x, w, b)
# dx, dw, db = affine_relu_backward(dout, cache)

# # The first term [0] is out
# # No idea about the meaning of dout
# dx_num = eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0], x, dout)
# dw_num = eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0], w, dout)
# db_num = eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0], b, dout)

# print('Testing affine_relu_forward:')
# print('dx error: ', rel_error(dx_num, dx))
# print('dw error: ', rel_error(dw_num, dw))

# #### End: Test several convenient layers in layer_utils.py ####

# # my own process result (take as reference)
# # dx error:  1.4512338590036844e-10
# # dw error:  8.631240030157025e-10
# # db error:  3.27565346718216e-12


# #### Start: Test softmax and SVM in layers.py ####
# num_classes, num_inputs = 10, 50
# x = 0.001 * np.random.randn(num_inputs, num_classes)
# y = np.random.randint(num_classes, size=num_inputs)

# dx_num = eval_numerical_gradient(lambda x: svm_loss(x, y)[0], x, verbose=False)
# loss, dx = svm_loss(x, y)

# # Test svm_loss function. Loss should be around 9 and dx error should be 1e-9
# print('Testing svm_loss:')
# print('loss: ', loss)
# print('dx error: ', rel_error(dx_num, dx))

# dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
# loss, dx = softmax_loss(x, y)

# # Test softmax_loss function. Loss should be 2.3 and dx error should be 1e-8
# print('\nTesting softmax_loss:')
# print('loss: ', loss)
# print('dx error: ', rel_error(dx_num, dx))

# #### End: Test softmax and SVM in layers.py ####

# # End of the test in layers.py
# #My own process result, take it as reference
# # Testing svm_loss:
# # loss:  9.001608952410084
# # dx error:  1.4021566006651672e-09
# #
# # Testing softmax_loss:
# # loss:  2.3027464187863287
# # dx error:  1.0054753708455757e-08


# # Test 2-layer network using modular implementations

#### Start: test of the core codes(class TwoLayerNet) are in classifiers/fc_net.py####

# N, D, H, C = 3, 5, 50, 7
# X = np.random.randn(N, D)
# y = np.random.randint(C, size=N)
# # C - total class
# # N - total input subject
# # D - total attributes for a single subject
# # H - hidden layers 中间隐藏层的维度 参数的数目
# std = 1e-2 # 是随机初始化权重时的标准偏差
# model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)

# print('Testing initialization ... ')
# W1_std = abs(model.params['W1'].std() - std)
# b1 = model.params['b1']
# W2_std = abs(model.params['W2'].std() - std)
# b2 = model.params['b2']
# assert W1_std < std / 10, 'First layer weights do not seem right'
# assert np.all(b1 == 0), 'First layer biases do not seem right'
# assert W2_std < std / 10, 'Second layer weights do not seem right'
# assert np.all(b2 == 0), 'Second layer biases do not seem right'

# print('Testing test-time forward pass ... ')
# model.params['W1'] = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
# model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
# model.params['W2'] = np.linspace(-0.3, 0.4, num=H*C).reshape(H, C)
# model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
# X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
# scores = model.loss(X)
# correct_scores = np.asarray(
#   [[11.53165108,  12.2917344,   13.05181771,  13.81190102,  14.57198434, 15.33206765,  16.09215096],
#    [12.05769098,  12.74614105,  13.43459113,  14.1230412,   14.81149128, 15.49994135,  16.18839143],
#    [12.58373087,  13.20054771,  13.81736455,  14.43418138,  15.05099822, 15.66781506,  16.2846319 ]])
# scores_diff = np.abs(scores - correct_scores).sum()
# assert scores_diff < 1e-6, 'Problem with test-time forward pass'

# print ('Testing training loss (no regularization)')
# y = np.asarray([0, 5, 1])
# loss, grads = model.loss(X, y)
# correct_loss = 3.4702243556
# assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'

# model.reg = 1.0
# loss, grads = model.loss(X, y)
# correct_loss = 26.5948426952
# assert abs(loss - correct_loss) < 1e-10, 'Problem with regularization loss'

# for reg in [0.0, 0.7]:
#   print ('Running numeric gradient check with reg = ', reg)
#   model.reg = reg
#   loss, grads = model.loss(X, y)

#   for name in sorted(grads):
#     f = lambda _: model.loss(X, y)[0]
#     grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
#     print ('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))

# # My own process result, take it as reference
# # Running numeric gradient check with reg =  0.0
# # W1 relative error: 1.83e-08
# # W2 relative error: 3.20e-10
# # b1 relative error: 9.83e-09
# # b2 relative error: 4.33e-10
# # Running numeric gradient check with reg =  0.7
# # W1 relative error: 2.53e-07
# # W2 relative error: 2.85e-08
# # b1 relative error: 1.56e-08
# # b2 relative error: 9.09e-10

#### End: test of the core codes(class TwoLayerNet) are in classifiers/fc_net.py####





#### Start: Test the accuracy on the validation set using Solver instance(in solver.py) to train a TwoLayerNet ####

# model = TwoLayerNet()
# solver = None

# ##############################################################################
# # TODO: Use a Solver instance to train a TwoLayerNet that achieves at least  #
# # 50% accuracy on the validation set.                                        #
# ##############################################################################
# pass
# solver = Solver(model, data,
#                     update_rule='sgd',
#                     optim_config={
#                       'learning_rate': 1e-3,
#                     },
#                     lr_decay=0.95,
#                     num_epochs=10, batch_size=100,
#                     print_every=100)
# solver.train()
# ##############################################################################
# #                             END OF YOUR CODE                               #
# ##############################################################################
# # End of the test on solver.py

# # Run this cell to visualize training loss and train / val accuracy
# # Please uncomment the previous test
# plt.subplot(2, 1, 1)
# plt.title('Training loss')
# plt.plot(solver.loss_history, 'o')
# plt.xlabel('Iteration')

# plt.subplot(2, 1, 2)
# plt.title('Accuracy')
# plt.plot(solver.train_acc_history, '-o', label='train')
# plt.plot(solver.val_acc_history, '-o', label='val')
# plt.plot([0.5] * len(solver.val_acc_history), 'k--')
# plt.xlabel('Epoch')
# plt.legend(loc='lower right')
# plt.gcf().set_size_inches(15, 12)
# plt.show()

#### End: Test the accuracy on the validation set using Solver instance(in solver.py) to train a TwoLayerNet ####


# Test Beginning
# Multiplayer Network
# implement a fully-connected network with an arbitrary number of hidden layers
# Read through the FullyConnectedNet class in the file classifier/fc_net.py

#### Initial loss and gradient check、
#### Start: Check the initial loss and to gradient check the network both with and without regularization. ####

# # For gradient checking, you should expect to see errors around 1e-6 or less
# N, D, H1, H2, C = 2, 15, 20, 30, 10
# X = np.random.randn(N, D)
# y = np.random.randint(C, size=(N,))

# for reg in [0, 3.14]:
#   print ('Running check with reg = ', reg)
#   model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,reg=reg, weight_scale=5e-2, dtype=np.float64)

#   loss, grads = model.loss(X, y)
#   print ('Initial loss: ', loss)
#   for name in sorted(grads):
#     f = lambda _: model.loss(X, y)[0]
#     grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
#     print ('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))

#### End: Check the initial loss and to gradient check the network both with and without regularization. ####


#### Start: Use a three-layer Net to overfit 50 training examples.####

# # 3-layer network with 100 units in each hidden layer.
# # Tweak the learning rate and initialization scale.
# # you should be able to overfit the achieve 100% training accuracy within 20 epochs.
# num_train = 50
# small_data = {
#   'X_train': data['X_train'][:num_train],
#   'y_train': data['y_train'][:num_train],
#   'X_val': data['X_val'],
#   'y_val': data['y_val'],
# }

# weight_scale = 1e-2
# learning_rate = 1e-4
# model = FullyConnectedNet([100, 100],
#               weight_scale=weight_scale, dtype=np.float64)
# solver = Solver(model, small_data,
#                 print_every=10, num_epochs=20, batch_size=25,
#                 update_rule='sgd',
#                 optim_config={
#                   'learning_rate': learning_rate,
#                 }
#          )
# solver.train()

# plt.plot(solver.loss_history, 'o')
# plt.title('Training loss history,3-layer')
# plt.xlabel('Iteration')
# plt.ylabel('Training loss')
# plt.show()

#### End: Use a three-layer Net to overfit 50 training examples.####



#### Start: Use a five-layer Net to overfit 50 training examples. ####

# # 5-layer network, all other things are same with the previous part
# num_train = 50
# small_data = {
#   'X_train': data['X_train'][:num_train],
#   'y_train': data['y_train'][:num_train],
#   'X_val': data['X_val'],
#   'y_val': data['y_val'],
# }

# learning_rate = 1e-3
# weight_scale = 3e-2
# model = FullyConnectedNet([100, 100, 100, 100],
#                 weight_scale=weight_scale, dtype=np.float64)
# solver = Solver(model, small_data,
#                 print_every=10, num_epochs=20, batch_size=25,
#                 update_rule='sgd',
#                 optim_config={
#                   'learning_rate': learning_rate,
#                 }
#          )
# solver.train()

# plt.plot(solver.loss_history, 'o')
# plt.title('Training loss history,5-layer')
# plt.xlabel('Iteration')
# plt.ylabel('Training loss')
# plt.show()

#### End: Use a five-layer Net to overfit 50 training examples. ####



#### Start: SGD+Momentum  (a update rule) ####

# # Stochastic gradient descent with momentum, a update rule that make deep networks converge faster
# # Implement the SGD+momentum update rule in the function sgd_momentum
# # you should see errors less than 1e-8

# from cs231n.optim import sgd_momentum

# N, D = 4, 5
# w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
# dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
# v = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)

# config = {'learning_rate': 1e-3, 'velocity': v}
# next_w, _ = sgd_momentum(w, dw, config=config)

# expected_next_w = np.asarray([
#   [ 0.1406,      0.20738947,  0.27417895,  0.34096842,  0.40775789],
#   [ 0.47454737,  0.54133684,  0.60812632,  0.67491579,  0.74170526],
#   [ 0.80849474,  0.87528421,  0.94207368,  1.00886316,  1.07565263],
#   [ 1.14244211,  1.20923158,  1.27602105,  1.34281053,  1.4096    ]])
# expected_velocity = np.asarray([
#   [ 0.5406,      0.55475789,  0.56891579, 0.58307368,  0.59723158],
#   [ 0.61138947,  0.62554737,  0.63970526,  0.65386316,  0.66802105],
#   [ 0.68217895,  0.69633684,  0.71049474,  0.72465263,  0.73881053],
#   [ 0.75296842,  0.76712632,  0.78128421,  0.79544211,  0.8096    ]])

# print('next_w error: ', rel_error(next_w, expected_next_w))
# print('velocity error: ', rel_error(expected_velocity, config['velocity']))

#### End: SGD+Momentum  (a update rule) ####



#### Start: a six-layer network with both SGD and SGD+momentum ####

# # Should see the SGD+momentum update rule converge faster

# num_train = 4000
# small_data = {
#   'X_train': data['X_train'][:num_train],
#   'y_train': data['y_train'][:num_train],
#   'X_val': data['X_val'],
#   'y_val': data['y_val'],
# }

# solvers = {}

# for update_rule in ['sgd', 'sgd_momentum']:
#   print('running with ', update_rule)
#   model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2)

#   solver = Solver(model, small_data,
#                   num_epochs=5, batch_size=100,
#                   update_rule=update_rule,
#                   optim_config={
#                     'learning_rate': 1e-3,
#                   },
#                   verbose=True)
#   solvers[update_rule] = solver
#   solver.train()
#   print()
# plt.subplot(3, 1, 1)
# plt.title('Training loss')
# plt.xlabel('Iteration')

# plt.subplot(3, 1, 2)
# plt.title('Training accuracy')
# plt.xlabel('Epoch')

# plt.subplot(3, 1, 3)
# plt.title('Validation accuracy')
# plt.xlabel('Epoch')

# for update_rule, solver in solvers.items():
# # for update_rule, solver in solvers.iteritems():
#   plt.subplot(3, 1, 1)
#   plt.plot(solver.loss_history, 'o', label=update_rule)

#   plt.subplot(3, 1, 2)
#   plt.plot(solver.train_acc_history, '-o', label=update_rule)

#   plt.subplot(3, 1, 3)
#   plt.plot(solver.val_acc_history, '-o', label=update_rule)

# for i in [1, 2, 3]:
#   plt.subplot(3, 1, i)
#   plt.legend(loc='upper center', ncol=4)
# plt.gcf().set_size_inches(15, 15)
# plt.show()

#### End: a six-layer network with both SGD and SGD+momentum ####




# Test Start
# RMSProp and Adam
# Both are update rules that set per-parameter learning rates by using a running average of the second moments of gradients.
# implement the RMSProp update rule in the rmsprop function and implement the Adam update rule in the adam function, all in the file optim.py


#### Start: Test RMSProp implementation ####

# # you should see errors less than 1e-7
# from cs231n.optim import rmsprop

# N, D = 4, 5
# w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
# dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
# cache = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)

# config = {'learning_rate': 1e-2, 'cache': cache}
# next_w, _ = rmsprop(w, dw, config=config)

# expected_next_w = np.asarray([
#   [-0.39223849, -0.34037513, -0.28849239, -0.23659121, -0.18467247],
#   [-0.132737,   -0.08078555, -0.02881884,  0.02316247,  0.07515774],
#   [ 0.12716641,  0.17918792,  0.23122175,  0.28326742,  0.33532447],
#   [ 0.38739248,  0.43947102,  0.49155973,  0.54365823,  0.59576619]])
# expected_cache = np.asarray([
#   [ 0.5976,      0.6126277,   0.6277108,   0.64284931,  0.65804321],
#   [ 0.67329252,  0.68859723,  0.70395734,  0.71937285,  0.73484377],
#   [ 0.75037008,  0.7659518,   0.78158892,  0.79728144,  0.81302936],
#   [ 0.82883269,  0.84469141,  0.86060554,  0.87657507,  0.8926    ]])

# print ('next_w error: ', rel_error(expected_next_w, next_w))
# print ('cache error: ', rel_error(expected_cache, config['cache']))

#### End: Test RMSProp implementation ####


#### Start: Test of Adam implementation ####

# # you should see errors around 1e-7 or less

# from cs231n.optim import adam

# N, D = 4, 5
# w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
# dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
# m = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)
# v = np.linspace(0.7, 0.5, num=N*D).reshape(N, D)

# config = {'learning_rate': 1e-2, 'm': m, 'v': v, 't': 5}
# next_w, _ = adam(w, dw, config=config)

# expected_next_w = np.asarray([
#   [-0.40094747, -0.34836187, -0.29577703, -0.24319299, -0.19060977],
#   [-0.1380274,  -0.08544591, -0.03286534,  0.01971428,  0.0722929],
#   [ 0.1248705,   0.17744702,  0.23002243,  0.28259667,  0.33516969],
#   [ 0.38774145,  0.44031188,  0.49288093,  0.54544852,  0.59801459]])
# expected_v = np.asarray([
#   [ 0.69966,     0.68908382,  0.67851319,  0.66794809,  0.65738853,],
#   [ 0.64683452,  0.63628604,  0.6257431,   0.61520571,  0.60467385,],
#   [ 0.59414753,  0.58362676,  0.57311152,  0.56260183,  0.55209767,],
#   [ 0.54159906,  0.53110598,  0.52061845,  0.51013645,  0.49966,   ]])
# expected_m = np.asarray([
#   [ 0.48,        0.49947368,  0.51894737,  0.53842105,  0.55789474],
#   [ 0.57736842,  0.59684211,  0.61631579,  0.63578947,  0.65526316],
#   [ 0.67473684,  0.69421053,  0.71368421,  0.73315789,  0.75263158],
#   [ 0.77210526,  0.79157895,  0.81105263,  0.83052632,  0.85      ]])

# print ('next_w error: ', rel_error(expected_next_w, next_w))
# print ('v error: ', rel_error(expected_v, config['v']))
# print ('m error: ', rel_error(expected_m, config['m']))

#### End: Test of Adam implementation ####




#### Start:Run the following to train a pair of deep networks using these new update rules: RMSProp and Adam ####

# # Uncomment former 'six-layer network with both SGD and SGD+momentum' to let the solvers and small_data available
# learning_rates = {'rmsprop': 1e-4, 'adam': 1e-3}
# for update_rule in ['adam', 'rmsprop']:
#   print('running with ', update_rule)
#   model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2)

#   solver = Solver(model, small_data,
#                   num_epochs=5, batch_size=100,
#                   update_rule=update_rule,
#                   optim_config={
#                     'learning_rate': learning_rates[update_rule]
#                   },
#                   verbose=True)
#   solvers[update_rule] = solver
#   solver.train()
#   print()

# plt.figure(7)
# plt.subplot(3, 1, 1)
# plt.title('Training loss')
# plt.xlabel('Iteration')

# plt.subplot(3, 1, 2)
# plt.title('Training accuracy')
# plt.xlabel('Epoch')

# plt.subplot(3, 1, 3)
# plt.title('Validation accuracy')
# plt.xlabel('Epoch')

# for update_rule, solver in solvers.items(): # Alter point
# # for update_rule, solver in solvers.iteritems():
#   plt.subplot(3, 1, 1)
#   plt.plot(solver.loss_history, 'o', label=update_rule)

#   plt.subplot(3, 1, 2)
#   plt.plot(solver.train_acc_history, '-o', label=update_rule)

#   plt.subplot(3, 1, 3)
#   plt.plot(solver.val_acc_history, '-o', label=update_rule)

# for i in [1, 2, 3]:
#   plt.subplot(3, 1, i)
#   plt.legend(loc='upper center', ncol=4)
# plt.gcf().set_size_inches(15, 15)
# plt.show()

#### End:Run the following to train a pair of deep networks using these new update rules: RMSProp and Adam ####





#### Start:Train a good model ####

# # Store your best model in the 'best_model' variable.
# # at least 50% accuracy on the validation set using a fully-connected net
# # prefer that you spend your effort working on convolutional nets rather than fully-connected nets

# best_model = None
# ################################################################################
# # TODO: Train the best FullyConnectedNet that you can on CIFAR-10. You might   #
# # batch normalization and dropout useful. Store your best model in the         #
# # best_model variable.                                                         #
# ################################################################################
# pass
# best_acc = 0
# for learning_rate in [1e-3, 1e-4]:
#     for reg in [1e-2, 1e-3, 1e-4]:
#         model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2, reg = reg)

#         solver = Solver(model, data,
#                       num_epochs=10, batch_size=200,
#                       update_rule='rmsprop',
#                       optim_config={
#                         'learning_rate': learning_rate
#                       },
#                       verbose=False)
#         solver.train()
#         print ('learning_rate = %f, reg = %f, best val loss = %f' %(learning_rate, reg, solver.best_val_acc))
#         if solver.best_val_acc > best_acc:
#             best_acc = solver.best_val_acc
#             best_model = model
# ################################################################################
# #                              END OF YOUR CODE                                #
# ################################################################################

#### End:Train a good model ####




#### Start: Test your best model ####

# y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
# y_val_pred = np.argmax(best_model.loss(data['X_val']), axis=1)
# print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())
# print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())

# # WARNING: It takes a long time to run
# # My own result, take as the reference
# # learning_rate = 0.001000, reg = 0.010000, best val loss = 0.496000
# # learning_rate = 0.001000, reg = 0.001000, best val loss = 0.511000
# # learning_rate = 0.001000, reg = 0.000100, best val loss = 0.505000
# # learning_rate = 0.000100, reg = 0.010000, best val loss = 0.449000
# # learning_rate = 0.000100, reg = 0.001000, best val loss = 0.428000
# # learning_rate = 0.000100, reg = 0.000100, best val loss = 0.444000
# # Validation set accuracy:  0.511
# # Test set accuracy:  0.467

#### End: Test your best model ####



