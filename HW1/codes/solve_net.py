from utils import LOG_INFO, onehot_encoding, calculate_acc
import numpy as np
import draw

iter_cnt = 0



def data_iterator(x, y, batch_size, shuffle=True):
    indx = list(range(len(x)))
    if shuffle:
        np.random.shuffle(indx)

    for start_idx in range(0, len(x), batch_size):
        end_idx = min(start_idx + batch_size, len(x))
        yield x[indx[start_idx: end_idx]], y[indx[start_idx: end_idx]]


def train_net(model, loss, config, inputs, labels, batch_size, disp_freq):

    iter_counter = 0
    loss_list = []
    acc_list = []

    training_acc = 0
    training_loss = 0

    for input, label in data_iterator(inputs, labels, batch_size):
        target = onehot_encoding(label, 10)
        iter_counter += 1


        global iter_cnt
        iter_cnt = iter_cnt + 1

        config['iterations'] = iter_cnt

        # forward net
        output = model.forward(input)
        # calculate loss
        loss_value = loss.forward(output, target)
        # generate gradient w.r.t loss
        grad = loss.backward(output, target)
        # backward gradient

        model.backward(grad)
        # update layers' weights
        model.update(config)

        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        acc_list.append(acc_value)

        training_acc += acc_value
        training_loss += loss_value
        # print(iter_cnt)

        # draw.plot.add_training(iter_cnt, np.mean(loss_value), acc_value)


        if iter_counter % disp_freq == 0:
            msg = '  Training iter %d, batch loss %.4f, batch acc %.4f' % (iter_counter, np.mean(loss_list), np.mean(acc_list))
            loss_list = []
            acc_list = []
            LOG_INFO(msg)

    training_acc /= (inputs.shape[0] / config['batch_size'])
    training_loss /= (inputs.shape[0] / config['batch_size'])

    # history_loss.append(training_loss)
    # if history_loss.__len__() >= config['patience'] and training_loss > np.min(history_loss[-config['patience']:]):
    #     history_loss.clear()
    #     config['learning_rate'] = config['learning_rate'] * config['factor']
    #
    # draw.plot.add_learning_rate(iter_cnt, config['learning_rate'])

    return training_acc


def test_net(model, loss, inputs, labels, batch_size):
    loss_list = []
    acc_list = []

    for input, label in data_iterator(inputs, labels, batch_size, shuffle=False):
        target = onehot_encoding(label, 10)
        output = model.forward(input)
        loss_value = loss.forward(output, target)
        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        acc_list.append(acc_value)

    msg = '    Testing, total mean loss %.5f, total acc %.5f' % (np.mean(loss_list), np.mean(acc_list))
    LOG_INFO(msg)

    return np.mean(loss_list), np.mean(acc_list)