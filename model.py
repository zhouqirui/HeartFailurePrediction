import pickle
import numpy as np
import paddle
import paddle.fluid as fluid
import sys

USE_CUDA = False
CLASS_DIM = 2
EMB_DIM = 128
BATCH_SIZE = 64

def load_data(sequences, labels):
    s = pickle.load(open(sequences, 'rb'))
    l = pickle.load(open(labels, 'rb'))
    return s, l

def split_data(sequences, labels, train=.75, test=.15, validation=.1):
    sequences = np.array(sequences)
    labels = np.array(labels)
    size = len(labels)
    index = np.random.permutation(size)
    nTest = int(size * test)
    nValid = int(size * validation)

    testIndex = index[:nTest]
    validIndex = index[nTest:nTest+nValid] if validation!=0 else None
    trainIndex = index[nTest+nValid:] if validation!=0 else index[nTest:]

    train_x = sequences[trainIndex]
    train_y = labels[trainIndex]
    test_x = sequences[testIndex]
    test_y = labels[testIndex]
    valid_x = sequences[validIndex] if validation!=0 else None
    valid_y = labels[validIndex] if validation!=0 else None

    return train_x, train_y, test_x, test_y, valid_x, valid_y

def build_GRU_model(input, inputDimSize, hiddenDimSize, dropout=True):
    emb = fluid.layers.embedding(input=input, size=[inputDimSize, EMB_DIM])
    x = fluid.layers.fc(input=emb, size=hiddenDimSize * 3)
    gru = fluid.layers.dynamic_gru(input = x, size = hiddenDimSize)
    pool = fluid.layers.sequence_pool(gru, 'max')
    if dropout:
        pool = fluid.layers.dropout(pool, 0.5)
    prediction = fluid.layers.fc(pool, CLASS_DIM, act='softmax')
    return prediction

def build_stacked_LSTM_model(input, inputDimSize, hiddenDimSize, stacked_num, dropout=True):
    emb = fluid.layers.embedding(input=input, size=[inputDimSize, EMB_DIM], is_sparse=True)
    fc1 = fluid.layers.fc(input=emb, size=hiddenDimSize*4)
    lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hiddenDimSize*4)
    inputs = [fc1, lstm1]
    for i in range(2, stacked_num+1):
        fc = fluid.layers.fc(input=inputs, size=hiddenDimSize*4)
        lstm, cell = fluid.layers.dynamic_lstm(input=fc, size=hiddenDimSize*4, is_reverse=(i%2==0))
        inputs = [fc, lstm]
    fc_last = fluid.layers.sequence_pool(input=inputs[0], pool_type='max')
    lstm_last = fluid.layers.sequence_pool(input=inputs[1], pool_type='max')
    prediction = fluid.layers.fc(input=[fc_last, lstm_last], size=CLASS_DIM, act='softmax')
    return prediction



def train(train_x, train_y, test_x, test_y, valid_x, valid_y, epochs):
    startup_program = fluid.default_startup_program()
    main_program = fluid.default_main_program()

    def train_reader():
        for i in range(len(train_x)):
            yield train_x[i], train_y[i]
    def test_reader():
        for i in range(len(test_x)):
            yield test_x[i], test_y[i]

    train_reader = paddle.batch(train_reader, batch_size = BATCH_SIZE)
    test_reader = paddle.batch(test_reader, batch_size = BATCH_SIZE)

    sequence = fluid.layers.data(name='sequence', shape=[1], dtype='int',lod_level=1)
    label = fluid.layers.data(name='label', shape=[1], dtype='int')

    # prediction = build_GRU_model(sequence, 4893, 100) ## For GRU model
    prediction = build_stacked_LSTM_model(sequence, 4893, 100, 3)
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    test_program = main_program.clone(for_test=True)
    # optimizer = fluid.optimizer.AdadeltaOptimizer(learning_rate=0.001) ## For GRU model
    optimizer = fluid.optimizer.Adagrad(0.002)
    optimizer.minimize(avg_loss)


    def train_test(train_test_program, train_test_feed, train_test_reader):
        acc_l = []
        avg_loss_l = []
        for test_data in train_test_reader():
            acc_np, avg_loss_np = exe.run(program=train_test_program, feed=train_test_feed.feed(test_data), fetch_list=[acc, avg_loss])
            acc_l.append(float(acc_np))
            avg_loss_l.append(float(avg_loss_np))
        acc_val_mean = np.array(acc_l).mean()
        avg_loss_val_mean = np.array(avg_loss_l).mean()
        return avg_loss_val_mean, acc_val_mean
    
    place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()
    exe = fluid.Executor(place)

    feeder = fluid.DataFeeder(feed_list=[sequence, label], place = place)
    exe.run(startup_program)

    ## The next line could be useless... Will come back later
    epochs = [i for i in range(epochs)]

    l = []
    step = 0

    for e in epochs:
        for step_id, data in enumerate(train_reader()):
            metrics = exe.run(main_program, feed=feeder.feed(data), fetch_list=[avg_loss, acc])
            if step % 100 == 0:
                print("Pass {}, Epoch {}, Cost {}".format(step, e, metrics[0]))
            step += 1
        avg_loss_val, acc_val = train_test(test_program, feeder, test_reader)
        print("Test with Epoch {}, avg_cost: {}, acc: {}".format(e, avg_loss_val, acc_val))

        l.append([e, avg_loss_val, acc_val])

        fluid.io.save_inference_model('./model/{}'.format(e), feeded_var_names = [sequence.name], target_vars = [prediction], executor = exe)

        best = sorted(l, key = lambda x:float(x[1]))[0]
        print('Best pass is {}, avg cost is {}'.format(best[0], best[1]))
        print('Accuracy is {}%.\n'.format(float(best[2]) * 100))
    print('Finished training.\n')


def infer(path_to_model, x, y):
    place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    lod = []
    for i in x:
        lod.append([x for x in i])
    base_shape = [[len(i) for i in lod]]
    tensor = fluid.create_lod_tensor(lod, base_shape, place)

    infer_program, feeded_var_names, target_var = fluid.io.load_inference_model(dirname=path_to_model, executor=exe)
    results = exe.run(infer_program, feed={feeded_var_names[0]:tensor}, fetch_list = target_var)
    predict = results[0]
    p = []
    
    for i in range(len(predict)):
        p.append(np.argmax(predict[i]))
    return p, y

def compute(predict, true_y):
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(predict)):
        if predict[i] == true_y[i] and predict[i] == 1:
            TP += 1
        elif predict[i] == true_y[i] and predict[i] == 0:
            TN += 1
        elif predict[i] != true_y[i] and predict[i] == 1:
            FP += 1
        elif predict[i] != true_y[i] and predict[i] == 0:
            FN += 1
    
    precision = TP /(TP + FP)
    sensitivity = TP / (TP + FN)
    specificity = TN / (FP + TN)
    print('The precision is {}.'.format(precision))
    print('The prediction has a sensitivity of {}, and a specificity of {}.'.format(sensitivity, specificity))



if __name__ == '__main__':
    epoch_num = int(sys.argv[1])
    sequences, labels = load_data('sequences', 'labels')
    train_x, train_y, test_x, test_y, valid_x, valid_y = split_data(sequences, labels)

    train(train_x, train_y, test_x, test_y, valid_x, valid_y, epoch_num)
    num = int(input('Load the model from epoch: '))
    predict, true_y = infer('./model/{}'.format(num), valid_x, valid_y)
    compute(predict, true_y)