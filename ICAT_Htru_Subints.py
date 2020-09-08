import os, time, shortcuts, shutil
import numpy as np
import theano, lasagne
import theano.tensor as T
import lasagne.layers as ll
import lasagne.nonlinearities as ln
import nn
from lasagne.init import Normal
from theano.sandbox.rng_mrg import MRG_RandomStreams
from objective import categorical_crossentropy_ssl_separated_two, categorical_crossentropy
import math
from sklearn.metrics import confusion_matrix
from shortcuts import convlayer
from layers.merge import ConvConcatLayer, MLPConcatLayer
from lasagne.layers import dnn

filename_script = os.path.basename(os.path.realpath(__file__))
outfolder = os.path.join("results_ICAT_Subints", os.path.splitext(filename_script)[0])
outfolder += '.'
outfolder += str(int(time.time()))

if not os.path.exists(outfolder):
    os.makedirs(outfolder)
    sample_path = os.path.join(outfolder, 'sample')
    os.makedirs(sample_path)
logfile = os.path.join(outfolder, 'logfile.log')
shutil.copy(os.path.realpath(__file__), os.path.join(outfolder, filename_script))

n_z = 100
num_classes = 2
z_generated = 50
train_batch_size = 100
n = 1
batch_size_g = 100 * n
lr = 5e-2
dim_input = (64, 64)
in_channels = 1
colorImg = False
generation_scale = False
seed=1234
rng=np.random.RandomState(seed)
theano_rng=MRG_RandomStreams(rng.randint(2 ** 15))
valid_batch_size = 200
batch_size_eval = 200
gen_final_non=ln.sigmoid
anneal_lr_every_epoch = 1
anneal_lr_factor = .995


train_data_pulsar_path = np.load("./file_info_trans64/pulsar_path_trans64.npy")[:480]
train_data_rfi_path = np.load("./file_info_trans64/rfi_path_trans64.npy")[:10920] 

test_data_pulsar_path = np.load("./file_info_trans64/pulsar_path_trans64.npy")[718:]
test_data_rfi_path = np.load("./file_info_trans64/rfi_path_trans64.npy")[41459:]

train_data_path = np.concatenate((train_data_pulsar_path, train_data_rfi_path))
test_data_path = np.concatenate((test_data_pulsar_path, test_data_rfi_path))

train_data = []
train_label = []
for i in train_data_path:
    data, label = np.load(i)
    train_data.append(data)
    train_label.append(label)

train_data = np.array(train_data)
train_label = np.array(train_label)

# valid_data = []
# valid_label = []
# for i in valid_data_path:
#     data, label = np.load(i)
#     valid_data.append(data)
#     valid_label.append(label)
#
# valid_data = np.array(valid_data)
# valid_label = np.array(valid_label)

test_data = []
test_label = []
for i in test_data_path:
    data, label = np.load(i)
    test_data.append(data)
    test_label.append(label)

test_data = np.array(test_data)
test_label = np.array(test_label)


train_data = train_data.reshape((train_data.shape[0], 1, 64, 64))
# valid_data = valid_data.reshape((valid_data.shape[0], 1, 64, 64))
test_data = test_data.reshape((test_data.shape[0], 1, 64, 64))


print('train_data:', train_data.shape)
# print('valid_data:', valid_data.shape)
print('test_data:', test_data.shape)

train_label = np.int32(train_label)
# valid_label = np.int32(valid_label)
test_label = np.int32(test_label)

ssl_data_seed = 1
rng_data = np.random.RandomState(ssl_data_seed)
train_inds = rng_data.permutation(train_data.shape[0])
train_data = train_data[train_inds]
train_label = train_label[train_inds]

valid_inds = rng_data.permutation(test_data.shape[0])
test_data = test_data[valid_inds]
test_label = test_label[valid_inds]


train_num_bathces = math.ceil(train_data.shape[0] / train_batch_size)
# vaild_num_bathces = math.ceil(valid_data.shape[0] / valid_batch_size)
test_num_bathces = math.ceil(test_data.shape[0] / valid_batch_size)

'''
models
'''
sym_x_l = T.tensor4()
sym_y = T.ivector()
sym_lr = T.scalar()
sym_x_eval = T.tensor4()
sym_y_g = T.ivector()
sym_z_image = T.tile(theano_rng.uniform((z_generated, n_z)), (num_classes, 1))
sym_z_rand = theano_rng.uniform(size=(batch_size_g, n_z))

###########
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

#################
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




##########
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

dis1_layers = dis_layers + dis1
dis2_layers = dis_layers + dis2


gen_out_x = ll.get_output(layer_or_layers=gen_layers[-1], inputs={gen_in_y: sym_y_g, gen_in_z: sym_z_rand},
                          deterministic=False)

cla_out_y_l = ll.get_output(cla_layers[-1], sym_x_l, deterministic=False)
cla_out_y_g = ll.get_output(cla_layers[-1], {cla_in_x: gen_out_x}, deterministic=False)
cla_out_y_g_hard = cla_out_y_g.argmax(axis=1)


dis1_out_p = ll.get_output(layer_or_layers=dis1_layers[-1], inputs={dis_in_x: sym_x_l, dis_in_y: sym_y},
                          deterministic=False)   # D(x, y)

dis1_out_p_g = ll.get_output(layer_or_layers=dis1_layers[-1], inputs={dis_in_x: gen_out_x, dis_in_y: sym_y_g},
                            deterministic=False)  # D(x', y)

dis1_out_p_g_c = ll.get_output(layer_or_layers=dis1_layers[-1], inputs={dis_in_x: gen_out_x, dis_in_y: cla_out_y_g_hard},
                            deterministic=False)

dis2_out_p_g_c = ll.get_output(layer_or_layers=dis2_layers[-1], inputs={dis_in_x: gen_out_x, dis_in_y: cla_out_y_g_hard},
                               deterministic=False)

dis2_out_p_g = ll.get_output(layer_or_layers=dis2_layers[-1], inputs={dis_in_x: gen_out_x, dis_in_y: sym_y_g},
                             deterministic=False)

image = ll.get_output(gen_layers[-2], {gen_in_y: sym_y_g, gen_in_z: sym_z_image}, deterministic=False)

bce = lasagne.objectives.binary_crossentropy


dis_cost_p = bce(dis1_out_p, T.ones(dis1_out_p.shape)).mean()
dis_cost_p_g = bce(dis1_out_p_g, T.zeros(dis1_out_p_g.shape)).mean()
dis_cost_p_g_c = bce(dis1_out_p_g_c, T.zeros(dis1_out_p_g_c.shape)).mean()

dis2_cost_p_g = bce(dis2_out_p_g, T.zeros(dis2_out_p_g.shape)).mean()
dis2_cost_p_g_c = bce(dis2_out_p_g_c, T.ones(dis2_out_p_g_c.shape)).mean()

gen_cost_p_g = bce(dis1_out_p_g, T.ones(dis1_out_p_g.shape)).mean()
gen2_cost_p_g = bce(dis2_out_p_g, T.ones(dis2_out_p_g.shape)).mean()

cla_cost_p_c = bce(dis1_out_p_g_c, T.ones(dis1_out_p_g_c.shape)).mean()
cla2_cost_p_c = bce(dis2_out_p_g_c, T.zeros(dis2_out_p_g_c.shape)).mean()

alpha_decay = 1e-4
alpha_labeled = 1.
alpha_cla = 1.
weight_d2 = 1e-4

weight_decay_classifier = lasagne.regularization.regularize_layer_params_weighted({cla_layers[-1]: 1},
                                                                                  lasagne.regularization.l2)

cla_cost_cla = categorical_crossentropy_ssl_separated_two(predictions_l=cla_out_y_l, targets=sym_y,
                                                      weight_decay=weight_decay_classifier, alpha_labeled=alpha_labeled,
                                                      alpha_decay=alpha_decay)

cla_cost_cla_g = categorical_crossentropy(predictions=cla_out_y_g, targets=sym_y_g)


dis_cost = dis_cost_p + dis_cost_p_g + 5e-1 * dis_cost_p_g_c
dis_cost_list = [dis_cost, dis_cost_p, dis_cost_p_g, 5e-1 * dis_cost_p_g_c]


dis2_cost = 0.5 * dis2_cost_p_g + .5 * dis2_cost_p_g_c
dis2_cost_list = [dis2_cost, 0.5 * dis2_cost_p_g, .5 * dis2_cost_p_g_c]

gen_cost = gen_cost_p_g + 0.05 * gen2_cost_p_g
gen_cost_list = [gen_cost, gen_cost_p_g, 0.05 * gen2_cost_p_g]

cla_cost1 = cla_cost_cla
cla_cost_list1 = [cla_cost1, cla_cost_cla]

cla_cost2 = cla_cost_cla + 2e-1*cla_cost_p_c + 2e-1 * cla2_cost_p_c
cla_cost_list2 = [cla_cost2, cla_cost_cla, cla_cost_cla_g, 2e-1*cla_cost_p_c, 2e-1 * cla2_cost_p_c]

cla_cost3 = cla_cost_cla + 1e-1 *cla_cost_cla_g + 2e-1 *cla_cost_p_c + 2e-1 * cla2_cost_p_c
cla_cost_list3 = [cla_cost3, cla_cost_cla, 1e-1 *cla_cost_cla_g, 2e-1 *cla_cost_p_c, 2e-1 * cla2_cost_p_c]


dis_params = ll.get_all_params(dis1_layers, trainable=True)
dis_grads = T.grad(dis_cost, dis_params)
dis_updates = lasagne.updates.adam(loss_or_grads=dis_grads, params=dis_params, learning_rate=0.0002, beta1=0.5, beta2=0.999)

dis2_params = ll.get_all_params(dis2_layers, trainable=True)
dis2_grads = T.grad(dis2_cost, dis2_params)
dis2_updates = lasagne.updates.adam(loss_or_grads=dis2_grads, params=dis2_params, learning_rate=0.0002, beta1=0.5, beta2=0.999)

gen_params = ll.get_all_params(gen_layers, trainable=True)
gen_grads = T.grad(gen_cost, gen_params)
gen_updates = lasagne.updates.adam(loss_or_grads=gen_grads, params=gen_params, learning_rate=0.0002, beta1=0.5, beta2=0.999)

cla_params = ll.get_all_params(cla_layers, trainable=True)

cla_grads1 = T.grad(cla_cost1, cla_params)
cla_grads2 = T.grad(cla_cost2, cla_params)
cla_grads3 = T.grad(cla_cost3, cla_params)

cla_updates1_ = lasagne.updates.adam(cla_grads1, cla_params, learning_rate=sym_lr, beta1=0.5, beta2=0.999)  # learning_rate=3e-3
cla_updates2_ = lasagne.updates.adam(cla_grads2, cla_params, learning_rate=sym_lr, beta1=0.5, beta2=0.999)
cla_updates3_ = lasagne.updates.adam(cla_grads3, cla_params, learning_rate=sym_lr, beta1=0.5, beta2=0.999)

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

print('pass-2')
train_batch_dis = theano.function(inputs=[sym_x_l, sym_y, sym_y_g],
                                  outputs=dis_cost_list, updates=dis_updates)

train_batch_dis2 = theano.function(inputs=[sym_y_g],
                                   outputs=dis2_cost_list, updates=dis2_updates)

train_batch_gen = theano.function(inputs=[sym_y_g],
                                  outputs=gen_cost_list, updates=gen_updates)


train_batch_cla1 = theano.function(inputs=[sym_x_l, sym_y, sym_lr],
                                   outputs=cla_cost_list1, updates=cla_updates1)

train_batch_cla2 = theano.function(inputs=[sym_x_l, sym_y, sym_y_g, sym_lr],
                                   outputs=cla_cost_list2, updates=cla_updates2)

train_batch_cla3 = theano.function(inputs=[sym_x_l, sym_y, sym_y_g, sym_lr],
                                   outputs=cla_cost_list3, updates=cla_updates3)


sym_z_image = T.tile(theano_rng.uniform((z_generated, n_z)), (num_classes, 1))


image = ll.get_output(gen_layers[-1], {gen_in_y: sym_y_g, gen_in_z: sym_z_image}, deterministic=False)  #
generate = theano.function(inputs=[sym_y_g], outputs=image)

cla_out_y_eval = ll.get_output(cla_layers[-1], sym_x_eval, deterministic=True)
predict = theano.function(inputs=[sym_x_eval], outputs=[cla_out_y_eval], givens=cla_avg_givens)

alpha_stage1 = 50
alpha_stage2 = 200
con_mat = []
for epoch in range(1, 300):
    start = time.time()
    d1 = [0.] * len(dis_cost_list)
    d2 = [0.] * len(dis2_cost_list)
    g1 = [0.] * len(gen_cost_list)
    c = [0.] * len(cla_cost_list2)

    for i in range(train_num_bathces):
        from_l_c = i * train_batch_size
        to_l_c = (i+1) * train_batch_size

        # sample_y = np.int32(np.repeat(np.arange(num_classes), batch_size_g / num_classes))
        batch_data = train_data[from_l_c:to_l_c]
        batch_label = train_label[from_l_c:to_l_c]

        sample_y = np.array(list(batch_label) * n)

        d1_b = train_batch_dis(batch_data, batch_label, sample_y)
        for j in range(len(d1)):
            d1[j] += d1_b[j]
        #
        d2_b = train_batch_dis2(sample_y)
        for j in range(len(d2)):
            d2[j] += d2_b[j]

        g1_b = train_batch_gen(sample_y)
        for j in range(len(g1)):
            g1[j] += g1_b[j]

        if epoch < alpha_stage1:
            c1_b = train_batch_cla1(batch_data, batch_label, lr)
            c1_b += [0, 0, 0]

        elif epoch > alpha_stage1 and epoch < alpha_stage2:
            c1_b = train_batch_cla2(batch_data, batch_label, sample_y, lr)

        else:
            c1_b = train_batch_cla3(batch_data, batch_label, sample_y, lr)

        for j in range(len(c)):
            c[j] += c1_b[j]

    for i in range(len(d1)):
        d1[i] /= train_num_bathces

    for i in range(len(d2)):
        d2[i] /= train_num_bathces

    for i in range(len(g1)):
        g1[i] /= train_num_bathces

    for i in range(len(c)):
        c[i] /= train_num_bathces

    line = "*Epoch=%d Time=%.5f LR=%.3f\n" % (epoch, time.time() - start, lr) + "DisLosses1: " + str(d1) + "\nDisLosses2: " + str(d2) \
           + "\nGenLosses: " + str(g1) + "\nClaLosses: " + str(c)

    print(line)

    with open(logfile, 'a') as f:
        f.write(line + "\n")

    # ground_truth_valid = []
    # label_valid = []
    # for i in range(vaild_num_bathces):
    #     from_l_c = i * batch_size_eval
    #     to_l_c = (i + 1) * batch_size_eval
    #
    #     batch_data = valid_data[from_l_c:to_l_c]
    #     batch_label = valid_label[from_l_c:to_l_c]
    #
    #     predict_label = predict(batch_data)
    #     label_batch = np.argmax(predict_label, axis=-1)[0]
    #
    #     label_valid.extend(label_batch)
    #     ground_truth_valid.extend(batch_label)
    #
    # Confu_matir = confusion_matrix(ground_truth_valid, label_valid)
    #
    # Recall_NN = Confu_matir[1][1] / (Confu_matir[1][1] + Confu_matir[1][0])
    # Presion_NN = Confu_matir[1][1] / (Confu_matir[1][1] + Confu_matir[0][1])
    # f1_CNN = (2 * Recall_NN * Presion_NN) / (Recall_NN + Presion_NN)
    # acc = (Confu_matir[0][0] + Confu_matir[1][1]) / (
    #         Confu_matir[0][0] + Confu_matir[1][1] + Confu_matir[0][1] + Confu_matir[1][0])
    #
    # print(Confu_matir)
    # print('Valid_Recall:', Recall_NN)
    # print('Valid_Presion:', Presion_NN)
    # print('Valid_F1-Score:', f1_CNN)
    # print('Valid_ACC:', acc)
    # print('************************')

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
    print('#############################################')

    if (epoch > alpha_stage2) and (epoch % anneal_lr_every_epoch) == 0:
        lr = lr * anneal_lr_factor

    if epoch % 1 == 0:
        print('Saving images')
        tail = '-' + str(epoch) + '.png'
        ran_y = np.int32(np.repeat(np.arange(num_classes), z_generated))
        x_gen = generate(ran_y)
        x_gen = x_gen.reshape((z_generated * num_classes, -1))

        image = shortcuts.mat_to_img(x_gen.T, dim_input, colorImg=colorImg, scale=generation_scale,
                                     save_path=os.path.join(sample_path, 'sample' + tail))
