import os, time, shutil
import numpy as np
import theano, lasagne
import theano.tensor as T
import lasagne.layers as ll
import lasagne.nonlinearities as ln
import nn
from lasagne.init import Normal
from theano.sandbox.rng_mrg import MRG_RandomStreams
from shortcuts import convlayer
from objective import categorical_crossentropy_ssl_separated, categorical_crossentropy
import shortcuts
import math
from layers.merge import ConvConcatLayer, MLPConcatLayer
from lasagne.layers import dnn
from sklearn.metrics import confusion_matrix

'''
parameters
'''
filename_script = os.path.basename(os.path.realpath(__file__))
outfolder = os.path.join("results-ssl", os.path.splitext(filename_script)[0])
outfolder += '.'
outfolder += str(int(time.time()))

if not os.path.exists(outfolder):
    os.makedirs(outfolder)
    sample_path = os.path.join(outfolder, 'sample')
    os.makedirs(sample_path)
logfile = os.path.join(outfolder, 'logfile.log')
shutil.copy(os.path.realpath(__file__), os.path.join(outfolder, filename_script))

num_labelled = 2000
ssl_para_seed = 1234
ssl_data_seed = 1
rng_data = np.random.RandomState(ssl_data_seed)
print('ssl_para_seed %d, num_labelled %d' % (ssl_para_seed, num_labelled))

num_classes = 2
train_batch_size = 100
valid_batch_size = 100

batch_size_g = 100
batch_size_l_c = min(100, num_labelled)
batch_size_u_c = max(100, int(10000 / num_labelled))
batch_size_u_d = 400
batch_size_l_d = max(int(num_labelled / 100), 1)  # 2000 / 100=20
# z_generated = num_classes
z_generated = 50
n_z = 100
eval_epoch = 1

# C
alpha_decay = 1e-4
alpha_labeled = 1.
alpha_unlabeled_entropy = .3
alpha_average = 1e-3
alpha_cla = 1.
# G
# n_z = 100
alpha_cla_g = .1
epoch_cla_g = 300
# D
noise_D_data = .3
noise_D = .5
# optimization
b1_g = .5  # mom1 in Adam
b1_d = .5
b1_c = .5

# threshold
alpha_stage1 = 30
alpha_stage2 = 100 if num_labelled >= 50 else 100
num_epochs = 300 if num_labelled >= 50 else 400
optim_flag = 1 if num_labelled >= 50 else 0


lr = 1e-2
anneal_lr_epoch = 200
anneal_lr_every_epoch = 1
anneal_lr_factor = .995
# data dependent

gen_final_non = ln.sigmoid
dim_input = (64, 64)
in_channels = 1
colorImg = False
generation_scale = False
weight_d2 = 1e-4
weight_d3 = 1e-4

rng = np.random.RandomState(ssl_para_seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

'''
data
'''
train_data_pulsar_path = np.load("./file_info_trans64/pulsar_path_trans64.npy")[:480]
train_data_rfi_path = np.load("./file_info_trans64/rfi_path_trans64.npy")[:10920]

test_data_pulsar_path = np.load("./file_info_trans64/pulsar_path_trans64.npy")[718:]
test_data_rfi_path = np.load("./file_info_trans64/rfi_path_trans64.npy")[41459:]

train_data_path = np.concatenate((train_data_pulsar_path, train_data_rfi_path))
test_data_path = np.concatenate((test_data_pulsar_path, test_data_rfi_path))



train_inds = rng_data.permutation(train_data_path.shape[0])
train_data_path = train_data_path[train_inds]

test_inds = rng_data.permutation(test_data_path.shape[0])
test_data_path = test_data_path[test_inds]

train_data = []
train_label = []

for i in train_data_path:
    data, label = np.load(i)
    train_data.append(data)
    train_label.append(label)

train_data = np.array(train_data)
train_label = np.array(train_label)


test_data = []
test_label = []
for i in test_data_path:
    data, label = np.load(i)
    test_data.append(data)
    test_label.append(label)

test_data = np.array(test_data)
test_label = np.array(test_label)


train_data = train_data.reshape((train_data.shape[0], 1, 64, 64))
test_data = test_data.reshape((test_data.shape[0], 1, 64, 64))

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

print(train_data.shape)
print(test_data.shape)

train_label = np.int32(train_label)
test_label = np.int32(test_label)

x_unlabelled = train_data.copy()

x_labelled = []
y_labelled = []
for j in range(num_classes):
    if j == 0:
        x_labelled.append(train_data[train_label == j][:(160 * 19)])
        y_labelled.append(train_label[train_label == j][: (160 * 19)])

    else:
        x_labelled.append(train_data[train_label == j][:160])
        y_labelled.append(train_label[train_label == j][:160])

x_labelled = np.concatenate(x_labelled, axis=0)
y_labelled = np.concatenate(y_labelled, axis=0)
print(y_labelled[3000:3100])

print('x_labelled:', x_labelled.shape)
print('y_labelled:', y_labelled.shape)
print('x_unlabeled:', x_unlabelled.shape)
del train_data

test_num_bathces = math.ceil(test_data.shape[0] / valid_batch_size)


n_batches_train_u_c = int(train_data_path.shape[0] / batch_size_u_c)
n_batches_train_l_c = int(x_labelled.shape[0] / batch_size_l_c)
n_batches_train_u_d = int(train_data_path.shape[0] / batch_size_u_d)
n_batches_train_l_d = int(x_labelled.shape[0] / batch_size_l_d)
n_batches_train_g = int(train_data_path.shape[0] / batch_size_g)
print('******************')
print(n_batches_train_u_c)
print(n_batches_train_l_c)
print(n_batches_train_u_d)
print(n_batches_train_l_d)
print(n_batches_train_g)

'''
models
'''
# symbols
sym_z_image = T.tile(theano_rng.uniform((z_generated, n_z)), (num_classes, 1))
sym_z_rand = theano_rng.uniform(size=(batch_size_g, n_z))
sym_x_u = T.tensor4()
sym_x_u_d = T.tensor4()
sym_x_u_g = T.tensor4()
sym_x_l = T.tensor4()

sym_y = T.ivector()
sym_y_g = T.ivector()
sym_x_eval = T.tensor4()
sym_lr = T.scalar()
sym_alpha_cla_g = T.scalar()
sym_alpha_unlabel_entropy = T.scalar()
sym_alpha_unlabel_average = T.scalar()


shared_unlabel = theano.shared(x_unlabelled, borrow=True)
slice_x_u_g = T.ivector()
slice_x_u_d = T.ivector()
slice_x_u_c = T.ivector()


########### Classifier
cla_in_x = ll.InputLayer(shape=(None, in_channels) + dim_input)
cla_layers = [cla_in_x]
cla_layers.append(convlayer(l=cla_layers[-1], bn=True, dr=0.5, ps=2, n_kerns=32, d_kerns=(5, 5),
                            pad='valid', stride=1, W=Normal(0.05), nonlinearity=ln.rectify,
                            name='cla-1'))

cla_layers.append(convlayer(l=cla_layers[-1], bn=True, dr=0.5, ps=2, n_kerns=64, d_kerns=(3, 3),
                            pad='valid', stride=1, W=Normal(0.05), nonlinearity=ln.rectify,
                            name='cla-3'))

cla_layers.append(convlayer(l=cla_layers[-1], bn=True, dr=0.5, ps=2, n_kerns=128, d_kerns=(3, 3),
                            pad='valid', stride=1, W=Normal(0.05),nonlinearity=ln.rectify,
                            name='cla-5'))

cla_layers.append(convlayer(l=cla_layers[-1], bn=True, dr=0.5, ps=2, n_kerns=256, d_kerns=(3, 3),
                            pad='valid', stride=1, W=Normal(0.05),nonlinearity=ln.rectify,
                            name='cla-5'))
cla_layers.append(convlayer(l=cla_layers[-1], bn=True, dr=0, ps=1, n_kerns=256, d_kerns=(3, 3),
                            pad='same', stride=1, W=Normal(0.05),nonlinearity=ln.rectify,
                            name='cla-6'))

cla_layers.append(ll.GlobalPoolLayer(cla_layers[-1]))
cla_layers.append(ll.DenseLayer(cla_layers[-1], num_units=num_classes, W=lasagne.init.Normal(1e-2, 0),
                                nonlinearity=ln.softmax, name='cla-6'))


################# Generator
gen_in_z = ll.InputLayer(shape=(None, n_z))
gen_in_y = ll.InputLayer(shape=(None,))
gen_layers = [gen_in_z]

gen_layers.append(MLPConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-5'))
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=512*4*4, nonlinearity=ln.linear, name='gen-6'), g=None, name='gen-61'))

gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (-1, 512, 4, 4), name='gen-7'))

gen_layers.append(ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-8'))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (None, 256, 8, 8), filter_size=(4,4), stride=(2, 2), W=Normal(0.05), nonlinearity=nn.relu, name='gen-11'), g=None, name='gen-12'))

gen_layers.append(ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-9'))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (None, 128, 16, 16), filter_size=(4,4), stride=(2, 2), W=Normal(0.05), nonlinearity=nn.relu, name='gen-11'), g=None, name='gen-12'))

gen_layers.append(ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-10'))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (None, 64, 32, 32), filter_size=(4,4), stride=(2, 2), W=Normal(0.05), nonlinearity=nn.relu, name='gen-11'), g=None, name='gen-12'))

gen_layers.append(ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-11'))
gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (None, 1, 64, 64), filter_size=(4,4), stride=(2, 2), W=Normal(0.05), nonlinearity=gen_final_non, name='gen-31'), train_g=True, init_stdv=0.1, name='gen-32'))



########## Discriminators
dis_in_x = ll.InputLayer(shape=(None, in_channels) + dim_input)
dis_in_y = ll.InputLayer(shape=(None,))
dis_layers = [dis_in_x]

dis_layers.append(ll.DropoutLayer(dis_layers[-1], p=0.2, name='dis-00'))
dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-01'))
dis_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 32, filter_size=4, stride=(2,2), pad=1, W=Normal(0.02), nonlinearity=nn.lrelu, name='dis-02'), name='dis-03'))

dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-20'))
dis_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 64, filter_size=4, stride=(2,2), pad=1, W=Normal(0.02), nonlinearity=nn.lrelu, name='dis-02'), name='dis-03'))

dis_layers.append(ll.DropoutLayer(dis_layers[-1], p=0.2, name='dis-23'))
dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-30'))
dis_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 128, filter_size=4, stride=(2,2), pad=1, W=Normal(0.02), nonlinearity=nn.lrelu, name='dis-02'), name='dis-03'))

dis_layers.append(ll.DropoutLayer(dis_layers[-1], p=0.2, name='dis-23'))
dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-40'))
dis_layers.append(nn.batch_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 256, filter_size=4, stride=(2,2), pad=1, W=Normal(0.02), nonlinearity=nn.lrelu, name='dis-02'), name='dis-03'))

dis_layers.append(ll.ReshapeLayer(dis_layers[-1], (-1, 256*4*4), name='dis-03'))

dis_layers.append(MLPConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-70'))

dis1 = [nn.DenseLayer(dis_layers[-1], num_units=1, nonlinearity=ln.sigmoid, name='dis-19')]
dis2 = [nn.DenseLayer(dis_layers[-1], num_units=1, nonlinearity=ln.sigmoid, name='dis-19')]
dis3 = [nn.DenseLayer(dis_layers[-1], num_units=1, nonlinearity=ln.sigmoid, name='dis-19')]

dis1_layers = dis_layers + dis1
dis2_layers = dis_layers + dis2
dis3_layers = dis_layers + dis3


'''
objectives
'''
gen_out_x = ll.get_output(layer_or_layers=gen_layers[-1], inputs={gen_in_y: sym_y_g, gen_in_z: sym_z_rand},
                          deterministic=False)

cla_out_y_l = ll.get_output(cla_layers[-1], sym_x_l, deterministic=False)
cla_out_y = ll.get_output(layer_or_layers=cla_layers[-1], inputs=sym_x_u, deterministic=False)
cla_out_y_g = ll.get_output(cla_layers[-1], {cla_in_x: gen_out_x}, deterministic=False)
cla_out_y_g_hard = cla_out_y_g.argmax(axis=1)

cla_out_y_d = ll.get_output(cla_layers[-1], {cla_in_x: sym_x_u_d}, deterministic=False)
cla_out_y_d_hard = cla_out_y_d.argmax(axis=1)


dis_out_p = ll.get_output(layer_or_layers=dis1_layers[-1],
                          inputs={dis_in_x: T.concatenate([sym_x_l, sym_x_u_d], axis=0),
                                  dis_in_y: T.concatenate([sym_y, cla_out_y_d_hard], axis=0)},
                          deterministic=False)


cla_out_y_hard = cla_out_y.argmax(axis=1)
dis_out_p_c = ll.get_output(layer_or_layers=dis1_layers[-1],
                            inputs={dis_in_x: sym_x_u, dis_in_y: cla_out_y_hard},
                            deterministic=False)


dis_out_p_g = ll.get_output(layer_or_layers=dis1_layers[-1], inputs={dis_in_x: gen_out_x, dis_in_y: sym_y_g},
                            deterministic=False)


dis_out_p_g_c = ll.get_output(layer_or_layers=dis1_layers[-1], inputs={dis_in_x: gen_out_x, dis_in_y: cla_out_y_g_hard},
                              deterministic=False)


dis2_out_p_g_c = ll.get_output(layer_or_layers=dis2_layers[-1],
                               inputs={dis_in_x: gen_out_x, dis_in_y: cla_out_y_g_hard},
                               deterministic=False)


dis2_out_p_c = ll.get_output(layer_or_layers=dis2_layers[-1],
                             inputs={dis_in_x: sym_x_u, dis_in_y: cla_out_y_hard},
                             deterministic=False)


dis2_out_p_g = ll.get_output(layer_or_layers=dis2_layers[-1],
                             inputs={dis_in_x: gen_out_x, dis_in_y: sym_y_g},
                             deterministic=False)


dis3_out_p_c = ll.get_output(layer_or_layers=dis3_layers[-1],
                             inputs={dis_in_x: sym_x_u, dis_in_y: cla_out_y_hard},
                             deterministic=False)


dis3_out_p_g = ll.get_output(layer_or_layers=dis3_layers[-1],
                             inputs={dis_in_x: gen_out_x, dis_in_y: sym_y_g},
                             deterministic=False)


image = ll.get_output(gen_layers[-1], {gen_in_y: sym_y_g, gen_in_z: sym_z_image}, deterministic=False)

cla_out_y_eval = ll.get_output(cla_layers[-1], sym_x_eval, deterministic=True)
accurracy_eval = (lasagne.objectives.categorical_accuracy(cla_out_y_eval, sym_y))
accurracy_eval = accurracy_eval.mean()


bce = lasagne.objectives.binary_crossentropy
############
dis_cost_p = bce(dis_out_p, T.ones(dis_out_p.shape)).mean()

dis_cost_p_c = bce(dis_out_p_c, T.zeros(dis_out_p_c.shape)).mean()

dis_cost_p_g = bce(dis_out_p_g, T.zeros(dis_out_p_g.shape)).mean()


dis_cost_p_g_c = bce(dis_out_p_g_c, T.zeros(dis_out_p_g_c.shape)).mean()

############

dis2_cost_p_c = bce(dis2_out_p_c, T.zeros(dis2_out_p_c.shape)).mean()

dis2_cost_p_g = bce(dis2_out_p_g, T.zeros(dis2_out_p_g.shape)).mean()

dis2_cost_p_g_c = bce(dis2_out_p_g_c, T.ones(dis2_out_p_g_c.shape)).mean()

############
dis3_cost_p_c = bce(dis3_out_p_c, T.ones(dis3_out_p_c.shape)).mean()

dis3_cost_p_g = bce(dis3_out_p_g, T.zeros(dis3_out_p_g.shape)).mean()

############
gen_cost_p_g = bce(dis_out_p_g, T.ones(dis_out_p_g.shape)).mean()
gen2_cost_p_g = bce(dis2_out_p_g, T.ones(dis2_out_p_g.shape)).mean()
gen3_cost_p_g_c = bce(dis3_out_p_g, T.ones(dis3_out_p_g.shape)).mean()

############
cla_cost_p_c = bce(dis_out_p_c, T.ones(dis_out_p_c.shape))  # C fools D
cla2_cost_p_c = bce(dis2_out_p_c, T.ones(dis2_out_p_c.shape))
cla3_cost_p_c = bce(dis3_out_p_c, T.zeros(dis3_out_p_c.shape))


p = cla_out_y.max(axis=1)
cla_cost_p_c = (cla_cost_p_c * p).mean()
cla2_cost_p_c = (cla2_cost_p_c * p).mean()
cla3_cost_p_c = (cla3_cost_p_c * p).mean()

weight_decay_classifier = lasagne.regularization.regularize_layer_params_weighted({cla_layers[-1]: 1},
                                                                                  lasagne.regularization.l2)

cla_cost_cla = categorical_crossentropy_ssl_separated(predictions_l=cla_out_y_l, targets=sym_y, predictions_u=cla_out_y,
                                                      weight_decay=weight_decay_classifier, alpha_labeled=alpha_labeled,
                                                      alpha_unlabeled=sym_alpha_unlabel_entropy,
                                                      alpha_average=sym_alpha_unlabel_average,
                                                      alpha_decay=alpha_decay)



cla_cost_cla_g = categorical_crossentropy(predictions=cla_out_y_g, targets=sym_y_g)

'''##################D1###################'''

dis_cost = dis_cost_p + .5 * dis_cost_p_g + .5 * dis_cost_p_c + .5 * dis_cost_p_g_c
dis_cost_list = [dis_cost]

'''##################D2###################'''

dis2_cost = .5 * dis2_cost_p_g + .5 * dis2_cost_p_c + .5 * dis2_cost_p_g_c
dis2_cost_list = [dis2_cost]

'''##################D3###################'''

dis3_cost = .5 * dis3_cost_p_g + .5 * dis3_cost_p_c
dis3_cost_list = [dis3_cost]

'''##################G###################'''
gen_cost = .5 * gen_cost_p_g + weight_d2 * gen2_cost_p_g + weight_d3 * gen3_cost_p_g_c
gen_cost_list = [gen_cost]

'''##################C###################'''
# first stage
cla_cost1 = cla_cost_cla
cla_cost_list1 = [cla_cost1]

# second stage
cla_cost2 = .5 * cla_cost_p_c + alpha_cla * cla_cost_cla + weight_d2 * cla2_cost_p_c + weight_d3 * cla3_cost_p_c
cla_cost_list2 = [cla_cost2]

# third stage
cla_cost3 = .5 * cla_cost_p_c + alpha_cla * (cla_cost_cla + sym_alpha_cla_g * cla_cost_cla_g) +\
            weight_d2 * cla2_cost_p_c + weight_d3 * cla3_cost_p_c
cla_cost_list3 = [cla_cost3]

dis_params = ll.get_all_params(dis1_layers, trainable=True)
dis_grads = T.grad(dis_cost, dis_params)
dis_updates = lasagne.updates.adam(loss_or_grads=dis_grads, params=dis_params, learning_rate=0.0002, beta1=b1_d)

########## updates of D2
dis2_params = ll.get_all_params(dis2_layers, trainable=True)
dis2_grads = T.grad(dis2_cost, dis2_params)
dis2_updates = lasagne.updates.adam(loss_or_grads=dis2_grads, params=dis2_params, learning_rate=0.0002, beta1=b1_d)

########## updates of D3
dis3_params = ll.get_all_params(dis3_layers[-1], trainable=True)
dis3_grads = T.grad(dis3_cost, dis3_params)
dis3_updates = lasagne.updates.adam(loss_or_grads=dis3_grads, params=dis3_params, learning_rate=0.0002, beta1=b1_d)

# updates of G
gen_params = ll.get_all_params(gen_layers, trainable=True)
gen_grads = T.grad(gen_cost, gen_params)
gen_updates = lasagne.updates.adam(gen_grads, gen_params, beta1=b1_g, learning_rate=0.0002)

# updates of C
cla_params = ll.get_all_params(cla_layers, trainable=True)

# first stage
cla_grads1 = T.grad(cla_cost1, cla_params)
cla_updates1_ = lasagne.updates.adam(cla_grads1, cla_params, beta1=b1_c, learning_rate=sym_lr)

# second stage
cla_grads2 = T.grad(cla_cost2, cla_params)
cla_updates2_ = lasagne.updates.adam(cla_grads2, cla_params, beta1=b1_c, learning_rate=sym_lr)

# third stage
cla_grads3 = T.grad(cla_cost3, cla_params)
cla_updates3_ = lasagne.updates.adam(cla_grads3, cla_params, beta1=b1_c, learning_rate=sym_lr)



######## avg
avg_params = lasagne.layers.get_all_params(cla_layers)
cla_param_avg = []
for param in avg_params:
    value = param.get_value(borrow=True)
    cla_param_avg.append(theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                       broadcastable=param.broadcastable,
                                       name=param.name))


cla_avg_updates = [(a, a + 0.01 * (p - a)) for p, a in zip(avg_params, cla_param_avg)]
cla_avg_givens = [(p, a) for p, a in zip(avg_params, cla_param_avg)]
cla_updates1 = list(cla_updates1_.items()) + cla_avg_updates
cla_updates2 = list(cla_updates2_.items()) + cla_avg_updates
cla_updates3 = list(cla_updates3_.items()) + cla_avg_updates



train_batch_dis = theano.function(inputs=[sym_x_l, sym_y, sym_y_g,
                                          slice_x_u_c, slice_x_u_d],
                                  outputs=dis_cost_list, updates=dis_updates,
                                  givens={sym_x_u: shared_unlabel[slice_x_u_c],
                                          sym_x_u_d: shared_unlabel[slice_x_u_d]})

train_batch_dis2 = theano.function(inputs=[sym_y_g, slice_x_u_c],
                                   outputs=dis2_cost_list, updates=dis2_updates,
                                   givens={sym_x_u: shared_unlabel[slice_x_u_c]})

train_batch_dis3 = theano.function(inputs=[sym_y_g, slice_x_u_c],
                                   outputs=dis3_cost_list, updates=dis3_updates,
                                   givens={sym_x_u: shared_unlabel[slice_x_u_c]})

train_batch_gen = theano.function(inputs=[sym_y_g],
                                  outputs=gen_cost_list, updates=gen_updates)

print('pass-1')

train_batch_cla1 = theano.function(inputs=[sym_x_l, sym_y, slice_x_u_c,
                                           sym_lr, sym_alpha_unlabel_entropy, sym_alpha_unlabel_average],
                                   outputs=cla_cost_list1, updates=cla_updates1,
                                   givens={sym_x_u: shared_unlabel[slice_x_u_c]})

train_batch_cla2 = theano.function(inputs=[sym_x_l, sym_y, slice_x_u_c,
                                           sym_lr, sym_alpha_unlabel_entropy, sym_alpha_unlabel_average],
                                   outputs=cla_cost_list2, updates=cla_updates2,
                                   givens={sym_x_u: shared_unlabel[slice_x_u_c]})

train_batch_cla3 = theano.function(inputs=[sym_x_l, sym_y, sym_y_g, slice_x_u_c, sym_alpha_cla_g,
                                           sym_lr, sym_alpha_unlabel_entropy, sym_alpha_unlabel_average],
                                   outputs=cla_cost_list3, updates=cla_updates3,
                                   givens={sym_x_u: shared_unlabel[slice_x_u_c]})

print('pass-2')
sym_index = T.iscalar()

generate = theano.function(inputs=[sym_y_g], outputs=image)

# avg
evaluate = theano.function(inputs=[sym_x_eval, sym_y], outputs=[accurracy_eval], givens=cla_avg_givens)
predict = theano.function(inputs=[sym_x_eval], outputs=[cla_out_y_eval], givens=cla_avg_givens)


np.random.seed(0)
p = np.array([0.95, 0.05])
a = [0, 1]

index_all = []
for i in range(100):
    index = np.random.choice(a, p=p.ravel())
    index_all.append(index)

print('index_all:', index_all)
print('OK')
batch_size_eval = 100
con_mat = []
for epoch in range(1, 200):
    start = time.time()
    p_l = rng.permutation(x_labelled.shape[0])
    x_labelled = x_labelled[p_l]
    y_labelled = y_labelled[p_l]

    p_u = rng.permutation(x_unlabelled.shape[0]).astype('int32')
    p_u_d = rng.permutation(x_unlabelled.shape[0]).astype('int32')
    p_u_g = rng.permutation(x_unlabelled.shape[0]).astype('int32')

    g1 = [0.] * len(gen_cost_list)
    c1 = [0.] * len(cla_cost_list3)

    for i in range(n_batches_train_u_c):

        from_u_c = i * batch_size_u_c
        to_u_c = (i + 1) * batch_size_u_c

        i_c = i % n_batches_train_l_c
        from_l_c = i_c * batch_size_l_c
        to_l_c = (i_c + 1) * batch_size_l_c

        i_d = i % n_batches_train_l_d
        from_l_d = i_d * batch_size_l_d
        to_l_d = (i_d + 1) * batch_size_l_d

        i_d_ = i % n_batches_train_u_d
        from_u_d = i_d_ * batch_size_u_d
        to_u_d = (i_d_ + 1) * batch_size_u_d

        sample_y = np.int32(index_all)

        if optim_flag == 1:
            d1_b = train_batch_dis(x_labelled[from_l_d:to_l_d], y_labelled[from_l_d:to_l_d], sample_y,
                                   p_u[from_u_c:to_u_c], p_u_d[from_u_d:to_u_d])
            d2_b = train_batch_dis2(sample_y, p_u[from_u_c:to_u_c])
            d3_b = train_batch_dis3(sample_y, p_u[from_u_c:to_u_c])

        else:
            d1_b = train_batch_dis(x_labelled[from_l_d:to_l_d], y_labelled[from_l_d:to_l_d], sample_y,
                                   p_u[from_u_c:to_u_c], p_u_d[from_u_d:to_u_d])

            for k in range(2):
                d2_b = train_batch_dis2(sample_y, p_u[from_u_c:to_u_c])

            if epoch < alpha_stage1:
                d3_b = train_batch_dis3(sample_y, p_u[from_u_c:to_u_c])

        g1_b = train_batch_gen(sample_y)

        if epoch < alpha_stage1:
            c1_b = train_batch_cla1(x_labelled[from_l_c:to_l_c], y_labelled[from_l_c:to_l_c],
                                    p_u[from_u_c:to_u_c], lr, alpha_unlabeled_entropy, alpha_average)

        elif epoch >= alpha_stage1 and epoch < alpha_stage2:

            c1_b = train_batch_cla2(x_labelled[from_l_c:to_l_c], y_labelled[from_l_c:to_l_c],
                                    p_u[from_u_c:to_u_c], lr, alpha_unlabeled_entropy, alpha_average)

        else:

            c1_b = train_batch_cla3(x_labelled[from_l_c:to_l_c], y_labelled[from_l_c:to_l_c],
                                    sample_y, p_u[from_u_c:to_u_c], alpha_cla_g,
                                    lr, alpha_unlabeled_entropy, alpha_average)

        for j in range(len(g1)):
            g1[j] += g1_b[j]

        for j in range(len(c1)):
            c1[j] += c1_b[j]

    for i in range(len(g1)):
        g1[i] /= n_batches_train_u_c

    for i in range(len(c1)):
        c1[i] /= n_batches_train_u_c

    if (epoch > anneal_lr_epoch) and (epoch % anneal_lr_every_epoch) == 0:
        lr = lr * anneal_lr_factor

    t = time.time() - start
    print('time:', t)
    line = "*Epoch=%d LR=%.5f\n" % (epoch, lr) + \
           "GenLosses:" + str(g1) + "\nClaLosses:" + str(c1)
    print('pass-4')
    print(line)

    with open(logfile, 'a') as f:
        f.write(line + '\n')

    if epoch % 1 == 0:
        start = time.time()
        ground_truth_test = []
        label_test = []
        for i in range(test_num_bathces):
            from_l_c = i * batch_size_eval
            to_l_c = (i + 1) * batch_size_eval

            batch_data = test_data[from_l_c:to_l_c]
            batch_label = test_label[from_l_c:to_l_c]

            predict_label = predict(batch_data)
            label_batch = np.argmax(predict_label, axis=-1)[0]

            label_test.extend(label_batch)
            ground_truth_test.extend(batch_label)

        Confu_matir = confusion_matrix(ground_truth_test, label_test)

        Recall_NN = Confu_matir[1][1] / (Confu_matir[1][1] + Confu_matir[1][0])
        Presion_NN = Confu_matir[1][1] / (Confu_matir[1][1] + Confu_matir[0][1])
        f1_CNN = (2 * Recall_NN * Presion_NN) / (Recall_NN + Presion_NN)
        acc = (Confu_matir[0][0] + Confu_matir[1][1]) / (
                Confu_matir[0][0] + Confu_matir[1][1] + Confu_matir[0][1] + Confu_matir[1][0])

        print(Confu_matir)
        print('Test_Recall:', Recall_NN)
        print('Test_Presion:', Presion_NN)
        print('Test_F1-Score:', f1_CNN)
        print('Test_ACC:', acc)
        print('elapse time:', time.time() - start)


    if epoch % 1 == 0:
        print('Saving images')
        tail = '-' + str(epoch) + '.png'
        ran_y = np.int32(np.repeat(np.arange(num_classes), z_generated))
        x_gen = generate(ran_y)
        x_gen = x_gen.reshape((z_generated * num_classes, -1))
        print(x_gen.shape)
        image = shortcuts.mat_to_img(x_gen.T, dim_input, colorImg=colorImg, scale=generation_scale,
                                     save_path=os.path.join(sample_path, 'sample' + tail))
    print('#############################################')

