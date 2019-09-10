import pickle
import numpy as np
import paddle
import paddle.fluid as fluid

USE_CUDA = False
CLASS_DIM = 2
EMB_DIM = 128

def load_data(sequences, labels):
    s = pickle.load(open(sequences, 'rb'))
    l = pickle.load(open(labels, 'rb'))
    return s, l

def split_data(sequences, labels, train=.75, test=.15, validation=.1):
    sequences = np.array(sequences)
    # print(type(sequences))
    # print(sequences.shape)
    labels = np.array(labels)
    # print(labels.shape)
    size = len(labels)
    index = np.random.permutation(size)
    nTest = int(size * test)
    nValid = int(size * validation)

    testIndex = index[:nTest]
    validIndex = index[nTest:nTest+nValid]
    trainIndex = index[nTest+nValid:]

    train_x = sequences[trainIndex]
    train_y = labels[trainIndex]
    test_x = sequences[testIndex]
    test_y = labels[testIndex]
    valid_x = sequences[validIndex]
    valid_y = labels[validIndex]

    return train_x, train_y, test_x, test_y, valid_x, valid_y

def build_model(input, inputDimSize, embDimSize, hiddenDimSize):
    print(input)
    emb = fluid.layers.embedding(input=input, size=[inputDimSize, EMB_DIM])
    print(emb)
    # print(fluid.layers.Print(emb, message = 'Show'))
    x = fluid.layers.fc(input=emb, size=hiddenDimSize * 3)
    print(x)
    gru = fluid.layers.dynamic_gru(input = x, size = hiddenDimSize)
    print(gru)
    pool = fluid.layers.sequence_pool(gru, 'max')
    print(pool)
    # model = fluid.layers.softmax(gru)
    model = fluid.layers.fc(pool, CLASS_DIM, act='softmax')

    return model



def train(train_x, train_y, test_x, test_y, valid_x, valid_y, epochs):
    startup_program = fluid.default_startup_program()
    main_program = fluid.default_main_program()

    BATCH_SIZE = 128
    # train_reader = paddle.batch(paddle.reader.shuffle(train_x, buf_size=500), batch_size = BATCH_SIZE)
    # test_reader = paddle.batch(test_x, batch_size = BATCH_SIZE)

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

    prediction = build_model(sequence, 4893, 500, 100)
    print(prediction)
    print(label)
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    ## test_program & optimizer
    test_program = main_program.clone(for_test=True)
    optimizer = fluid.optimizer.AdadeltaOptimizer(learning_rate=0.001)
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
    print(type(feeder))
    exe.run(startup_program)

    ## The next line could be useless... Will come back later
    epochs = [i for i in range(epochs)]

    l = []
    step = 0

    for e in epochs:

        # print(type(train_reader))
        # print(list(train_reader())[0])

        for step_id, data in enumerate(train_reader()):
            # print(step_id)
            metrics = exe.run(main_program, feed=feeder.feed(data), fetch_list=[avg_loss, acc])
            if step % 100 == 0:
                print("Pass {}, Epoch {}, Cost {}".format(step, e, metrics[0]))
            step += 1
        avg_loss_val, acc_val = train_test(test_program, feeder, test_reader)
        print("Test with Epoch {}, avg_cost: {}, acc: {}".format(e, avg_loss_val, acc_val))

        l.append([e, avg_loss_val, acc_val])

        # fluid.io.save_inference_model('model.h', )

        best = sorted(l, key = lambda x:float(x[1]))[0]
        print('Best pass is {}, avg cost is {}'.format(best[0], best[1]))
        print('Accuracy is {}%'.format(float(best[2]) * 100))




if __name__ == '__main__':
    sequences, labels = load_data('sequences', 'labels')
    train_x, train_y, test_x, test_y, valid_x, valid_y = split_data(sequences, labels)
    print(len(train_x), len(train_y), len(test_x), len(test_y))


    train(train_x, train_y, test_x, test_y, valid_x, valid_y, 20)
    # except:
    #     print('Error')
    # train_reader = paddle.batch(paddle.reader.shuffle(train_x, buf_size=500), batch_size = BATCH_SIZE)
    # test_reader = paddle.batch(test_x, batch_size = BATCH_SIZE)

    # sequence = fluid.layers.data(name='sequence', shape=[4893, None], dtype='float')
    # label = fluid.layers.data(name='label', shape=[1], dtype='int32')
    # model = build_model(sequence, 4893, 100, 100)

    # cost = fluid.layers.classification_cost(input = model, label = label)