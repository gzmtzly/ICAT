import theano.tensor as T
import theano
import lasagne
def categorical_crossentropy(predictions, targets, epsilon=1e-6):
    predictions = T.clip(predictions, epsilon, 1-epsilon)
    num_cls = predictions.shape[1]
    if targets.ndim == predictions.ndim - 1:
        targets = theano.tensor.extra_ops.to_one_hot(targets, num_cls)
    elif targets.ndim != predictions.ndim:
        raise TypeError('rank mismatch between target and predictions')
    return lasagne.objectives.categorical_crossentropy(predictions, targets).mean()

def categorical_crossentropy2(predictions, targets, epsilon=1e-6):
    predictions = T.clip(predictions, epsilon, 1-epsilon)
    num_cls = predictions.shape[1]
    if targets.ndim == predictions.ndim - 1:
        targets = theano.tensor.extra_ops.to_one_hot(targets, num_cls)
    elif targets.ndim != predictions.ndim:
        raise TypeError('rank mismatch between target and predictions')
    return lasagne.objectives.categorical_crossentropy(predictions, targets)

def entropy(predicitons):
    return categorical_crossentropy(predicitons, predicitons)

def categorical_crossentropy_of_mean(prediction):
    num_cls = prediction.shape[1]
    uniform_targets = T.ones((1, num_cls)) / num_cls
    return categorical_crossentropy(prediction.mean(axis=0, keepdims=True), uniform_targets)

def categorical_crossentropy_ssl_separated(predictions_l, targets, predictions_u, weight_decay, alpha_labeled=1.,
                                           alpha_unlabeled=.3, alpha_average=1e-3, alpha_decay=1e-4):
    ce_loss = categorical_crossentropy(predictions_l, targets)
    en_loss = entropy(predictions_u)
    av_loss = categorical_crossentropy_of_mean(predictions_u)
    return alpha_labeled*ce_loss + alpha_unlabeled*en_loss + alpha_average*av_loss + alpha_decay*weight_decay

def categorical_crossentropy_ssl_separated_two(predictions_l, targets, weight_decay, alpha_labeled=1.,
                                           alpha_average=1e-3, alpha_decay=1e-4):
    ce_loss = categorical_crossentropy(predictions_l, targets)
    return alpha_labeled*ce_loss + alpha_decay*weight_decay

