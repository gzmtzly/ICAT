import os, time, shutil
import numpy as np
import theano, lasagne
import theano.tensor as T
import lasagne.layers as ll
import lasagne.nonlinearities as ln
from lasagne.layers import dnn
import nn
from lasagne.init import Normal
from theano.sandbox.rng_mrg import MRG_RandomStreams
import svhn_data
from objective import categorical_crossentropy_ssl_separated_two, categorical_crossentropy
import math
from layers.merge import ConvConcatLayer, MLPConcatLayer

num_labelled = 20000
n = 2
batch_size_g = 100 * n

filename_script = os.path.basename(os.path.realpath(__file__))
outfolder = os.path.join("results_1234", os.path.splitext(filename_script)[0])
outfolder += '.'
outfolder += str(int(time.time()))

outfolder += '_labeled'  
outfolder += str(num_labelled)
outfolder += str(73200)

outfolder += '_generated'
outfolder += str(batch_size_g)

if not os.path.exists(outfolder):
   os.makedirs(outfolder)
   sample_path = os.path.join(outfolder, 'sample')
   os.makedirs(sample_path)
logfile = os.path.join(outfolder, 'logfile.log')
shutil.copy(os.path.realpath(__file__), os.path.join(outfolder, filename_script))

'''
parameters
'''
seed=135
rng=np.random.RandomState(seed)
theano_rng=MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))
ssl_data_seed = 1
# dataset
data_dir = './data/svhn'

lr = 3e-3
train_batch_size = 100
num_classes = 10
dim_input = (32, 32)
in_channels = 3
batch_size_g = 200
colorImg=True
generation_scale=True
z_generated=num_classes
gen_final_non=ln.tanh

# evaluation
batch_size_eval = 200
n_z = 100


'''
data
'''
def rescale(mat):
    return np.transpose(np.cast[theano.config.floatX]((-127.5 + mat)/127.5),(3,2,0,1))
train_x, train_y = svhn_data.load('./data', 'train')
eval_x, eval_y = svhn_data.load('./data', 'test')

train_y = np.int32(train_y)
eval_y = np.int32(eval_y)
train_x = rescale(train_x)
eval_x = rescale(eval_x)


print(train_x.shape, eval_x.shape)

train_num_bathces = math.ceil(train_x.shape[0] / train_batch_size)        
n_batches_eval = math.ceil(eval_x.shape[0] / batch_size_eval)            

print('train_num_bathces:', train_num_bathces)
print('n_batches_eval:', n_batches_eval)

'''
models
'''
sym_x_l = T.tensor4()
sym_y = T.ivector()
sym_x_eval = T.tensor4()
sym_y_g = T.ivector()
sym_z_image = T.tile(theano_rng.uniform((z_generated, n_z)), (num_classes, 1))
sym_z_rand = theano_rng.uniform(size=(batch_size_g, n_z))


##################################### classifier

cla_in_x = ll.InputLayer(shape=(None, in_channels) + dim_input)
cla_layers = [cla_in_x]
cla_layers.append(ll.DropoutLayer(cla_layers[-1], p=0.2, name='cla-00'))
cla_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(cla_layers[-1], 128, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='cla-02'), name='cla-03'))
cla_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(cla_layers[-1], 128, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='cla-11'), name='cla-12'))
cla_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(cla_layers[-1], 128, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='cla-21'), name='cla-22'))
cla_layers.append(dnn.MaxPool2DDNNLayer(cla_layers[-1], pool_size=(2, 2)))
cla_layers.append(ll.DropoutLayer(cla_layers[-1], p=0.5, name='cla-23'))
cla_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(cla_layers[-1], 256, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='cla-31'), name='cla-32'))
cla_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(cla_layers[-1], 256, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='cla-41'), name='cla-42'))
cla_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(cla_layers[-1], 256, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='cla-51'), name='cla-52'))
cla_layers.append(dnn.MaxPool2DDNNLayer(cla_layers[-1], pool_size=(2, 2)))
cla_layers.append(ll.DropoutLayer(cla_layers[-1], p=0.5, name='cla-53'))
cla_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(cla_layers[-1], 512, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu, name='cla-61'), name='cla-62'))
cla_layers.append(ll.batch_norm(ll.NINLayer(cla_layers[-1], num_units=256, W=Normal(0.05), nonlinearity=nn.lrelu, name='cla-71'), name='cla-72'))
cla_layers.append(ll.batch_norm(ll.NINLayer(cla_layers[-1], num_units=128, W=Normal(0.05), nonlinearity=nn.lrelu, name='cla-81'), name='cla-82'))
cla_layers.append(ll.GlobalPoolLayer(cla_layers[-1], name='cla-83'))
cla_layers.append(ll.batch_norm(ll.DenseLayer(cla_layers[-1], num_units=num_classes, W=Normal(0.05), nonlinearity=ln.softmax, name='cla-91'), name='cla-92'))

################################## generator
gen_in_z = ll.InputLayer(shape=(None, n_z))
gen_in_y = ll.InputLayer(shape=(None,))
gen_layers = [gen_in_z]
gen_layers.append(MLPConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-00'))
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=4*4*512, W=Normal(0.05), nonlinearity=nn.relu, name='gen-01'), g=None, name='gen-02'))
gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (-1,512,4,4), name='gen-03'))
gen_layers.append(ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-10'))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (None,256,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu, name='gen-11'), g=None, name='gen-12'))
gen_layers.append(ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-20'))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (None,128,16,16), (5,5), W=Normal(0.05), nonlinearity=nn.relu, name='gen-21'), g=None, name='gen-22'))
gen_layers.append(ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-30'))
gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (None,3,32,32), (5,5), W=Normal(0.05), nonlinearity=gen_final_non, name='gen-31'), train_g=True, init_stdv=0.1, name='gen-32'))


################################### discriminators
dis_in_x = ll.InputLayer(shape=(None, in_channels) + dim_input)
dis_in_y = ll.InputLayer(shape=(None,))
dis_layers = [dis_in_x]
dis_layers.append(ll.DropoutLayer(dis_layers[-1], p=0.2, name='dis-00'))
dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-01'))
dis_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 32, (3,3), pad=1, W=Normal(0.05),
                                                    nonlinearity=nn.lrelu, name='dis-02'), name='dis-03'))

dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-20'))
dis_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 32, (3,3), pad=1, stride=2, W=Normal(0.05),
                                                    nonlinearity=nn.lrelu, name='dis-21'), name='dis-22'))

dis_layers.append(ll.DropoutLayer(dis_layers[-1], p=0.2, name='dis-23'))
dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-30'))
dis_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 64, (3,3), pad=1, W=Normal(0.05),
                                                    nonlinearity=nn.lrelu, name='dis-31'), name='dis-32'))

dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-40'))
dis_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 64, (3,3), pad=1, stride=2, W=Normal(0.05),
                                                    nonlinearity=nn.lrelu, name='dis-41'), name='dis-42'))

dis_layers.append(ll.DropoutLayer(dis_layers[-1], p=0.2, name='dis-43'))
dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-50'))
dis_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 128, (3,3), pad=0, W=Normal(0.05),
                                                    nonlinearity=nn.lrelu, name='dis-51'), name='dis-52'))

dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-60'))
dis_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 128, (3,3), pad=0, W=Normal(0.05),
                                                    nonlinearity=nn.lrelu, name='dis-61'), name='dis-62'))

dis_layers.append(ll.GlobalPoolLayer(dis_layers[-1], name='dis-63'))
dis_layers.append(MLPConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-70'))


dis1 = [nn.weight_norm(ll.DenseLayer(dis_layers[-1], num_units=1, W=Normal(0.05),
                                     nonlinearity=ln.sigmoid, name='dis-71'), train_g=True, init_stdv=0.1, name='dis-72')]
dis2 = [nn.weight_norm(ll.DenseLayer(dis_layers[-1], num_units=1, W=Normal(0.05),
                                     nonlinearity=ln.sigmoid, name='dis-71'), train_g=True, init_stdv=0.1, name='dis-72')]

dis1_layers = dis_layers + dis1
dis2_layers = dis_layers + dis2

cla_out_y_eval = ll.get_output(cla_layers[-1], sym_x_eval, deterministic=True)
accurracy_eval = (lasagne.objectives.categorical_accuracy(cla_out_y_eval, sym_y))
accurracy_eval = accurracy_eval.mean()


gen_out_x = ll.get_output(layer_or_layers=gen_layers[-1], inputs={gen_in_y: sym_y_g, gen_in_z: sym_z_rand},
                          deterministic=False)

cla_out_y_l = ll.get_output(cla_layers[-1], sym_x_l, deterministic=False)   
cla_out_y_g = ll.get_output(cla_layers[-1], {cla_in_x: gen_out_x}, deterministic=False) 
cla_out_y_g_hard = cla_out_y_g.argmax(axis=1)


dis1_out_p = ll.get_output(layer_or_layers=dis1_layers[-1], inputs={dis_in_x: sym_x_l, dis_in_y: sym_y},
                          deterministic=False) 

dis1_out_p_g = ll.get_output(layer_or_layers=dis1_layers[-1], inputs={dis_in_x: gen_out_x, dis_in_y: sym_y_g},
                            deterministic=False)  

dis1_out_p_g_c = ll.get_output(layer_or_layers=dis1_layers[-1], inputs={dis_in_x: gen_out_x, dis_in_y: cla_out_y_g_hard},
                            deterministic=False)

dis2_out_p_g_c = ll.get_output(layer_or_layers=dis2_layers[-1], inputs={dis_in_x: gen_out_x, dis_in_y: cla_out_y_g_hard},
                               deterministic=False)

dis2_out_p_g = ll.get_output(layer_or_layers=dis2_layers[-1], inputs={dis_in_x: gen_out_x, dis_in_y: sym_y_g},
                             deterministic=False)

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


dis_cost = dis_cost_p + .5 * dis_cost_p_g + .5 * dis_cost_p_g_c
dis_cost_list = [dis_cost, dis_cost_p, .5 * dis_cost_p_g, .5 * dis_cost_p_g_c]


dis2_cost = 0.5 * dis2_cost_p_g + .5 * dis2_cost_p_g_c
dis2_cost_list = [dis2_cost, 0.5 * dis2_cost_p_g, .5 * dis2_cost_p_g_c]



gen_cost = gen_cost_p_g + 5e-2 * gen2_cost_p_g
gen_cost_list = [gen_cost, gen_cost_p_g, 5e-2 * gen2_cost_p_g]




cla_cost1 = cla_cost_cla
cla_cost_list1 = [cla_cost1, cla_cost_cla]

cla_cost2 = cla_cost_cla + 5e-2*cla_cost_p_c + 5e-2 * cla2_cost_p_c
cla_cost_list2 = [cla_cost2, cla_cost_cla, cla_cost_cla_g, 5e-2*cla_cost_p_c, 5e-2 * cla2_cost_p_c]



cla_cost3 = cla_cost_cla + 1e-1 *cla_cost_cla_g + 5e-2 *cla_cost_p_c + 5e-2 * cla2_cost_p_c
cla_cost_list3 = [cla_cost3, cla_cost_cla, cla_cost_cla_g, 5e-2 *cla_cost_p_c, 5e-2 * cla2_cost_p_c]

dis_params = ll.get_all_params(dis1_layers, trainable=True)
dis_grads = T.grad(dis_cost, dis_params)
dis_updates = lasagne.updates.adam(loss_or_grads=dis_grads, params=dis_params, learning_rate=0.0005, beta1=0.5, beta2=0.999)

dis2_params = ll.get_all_params(dis2_layers, trainable=True)
dis2_grads = T.grad(dis2_cost, dis2_params)
dis2_updates = lasagne.updates.adam(loss_or_grads=dis2_grads, params=dis2_params, learning_rate=0.0005, beta1=0.5, beta2=0.999)


gen_params = ll.get_all_params(gen_layers, trainable=True)
gen_grads = T.grad(gen_cost, gen_params)
gen_updates = lasagne.updates.adam(loss_or_grads=gen_grads, params=gen_params, learning_rate=0.0005, beta1=0.5, beta2=0.999)


cla_params = ll.get_all_params(cla_layers, trainable=True)


cla_grads1 = T.grad(cla_cost1, cla_params)

cla_grads2 = T.grad(cla_cost2, cla_params)

cla_grads3 = T.grad(cla_cost3, cla_params)

cla_updates1_ = lasagne.updates.adam(cla_grads1, cla_params, learning_rate=3e-3, beta1=0.9, beta2=0.999) 
cla_updates2_ = lasagne.updates.adam(cla_grads2, cla_params, learning_rate=3e-3, beta1=0.9, beta2=0.999)
cla_updates3_ = lasagne.updates.adam(cla_grads3, cla_params, learning_rate=3e-3, beta1=0.5, beta2=0.999)

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

train_batch_cla1 = theano.function(inputs=[sym_x_l, sym_y],
                                   outputs=cla_cost_list1, updates=cla_updates1)

train_batch_cla2 = theano.function(inputs=[sym_x_l, sym_y, sym_y_g],
                                   outputs=cla_cost_list2, updates=cla_updates2)

train_batch_cla3 = theano.function(inputs=[sym_x_l, sym_y, sym_y_g],
                                   outputs=cla_cost_list3, updates=cla_updates3)

sym_z_image = T.tile(theano_rng.uniform((z_generated, n_z)), (num_classes, 1))



image = ll.get_output(gen_layers[-1], {gen_in_y: sym_y_g, gen_in_z: sym_z_image}, deterministic=False)

generate = theano.function(inputs=[sym_y_g], outputs=image)
evaluate = theano.function(inputs=[sym_x_eval, sym_y], outputs=[accurracy_eval], givens=cla_avg_givens)
predict = theano.function(inputs=[sym_x_eval], outputs=[cla_out_y_eval], givens=cla_avg_givens)


alpha_stage1 = 10
alpha_stage2 = 250
svhn_acc = []
eval_epoch = 1
for epoch in range(1, 350):
    start = time.time()
    d1 = [0.] * len(dis_cost_list)
    d2 = [0.] * len(dis2_cost_list)
    g1 = [0.] * len(gen_cost_list)
    c = [0.] * len(cla_cost_list2)

    for i in range(train_num_bathces):
        from_l_c = i * train_batch_size
        to_l_c = (i+1) * train_batch_size

        sample_y = np.int32(np.repeat(np.arange(num_classes), batch_size_g / num_classes))
        batch_data = train_x[from_l_c:to_l_c]
        batch_label = train_y[from_l_c:to_l_c]
        # sample_y = np.array(list(batch_label) * n)

        d1_b = train_batch_dis(batch_data, batch_label, sample_y)
        for j in range(len(d1)):
            d1[j] += d1_b[j]

        d2_b = train_batch_dis2(sample_y)
        for j in range(len(d2)):
            d2[j] += d2_b[j]

        g1_b = train_batch_gen(sample_y)
        for j in range(len(g1)):
            g1[j] += g1_b[j]


        if epoch < alpha_stage1:
            c1_b = train_batch_cla1(batch_data, batch_label)
            c1_b += [0, 0, 0]

        elif epoch > alpha_stage1 and epoch < alpha_stage2:
            c1_b = train_batch_cla2(batch_data, batch_label, sample_y)

        else:
            c1_b = train_batch_cla3(batch_data, batch_label, sample_y)

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

    label = []
    for i in range(n_batches_eval):
        predict_label = predict(eval_x[i * batch_size_eval:(i + 1) * batch_size_eval])
        label_batch = np.argmax(predict_label, axis=-1)[0]
        label.extend(label_batch)

    acc = np.sum(eval_y == label) / len(label)

    line = "*Epoch=%d Time=%.5f Acc=%4f \n" % (epoch, time.time() - start, acc) + "DisLosses1: " + str(d1) + "\nDisLosses2: " + str(d2) \
           + "\nGenLosses: " + str(g1) + "\nClaLosses: " + str(c)

    with open(logfile, 'a') as f:
        f.write(line + "\n\n")

    print(line)


    print('#############################################')
