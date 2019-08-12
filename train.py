import os, sys
import argparse
import numpy as np
from collections import OrderedDict
from dataset.load import Load
from network.cnn import *

def main(args):
    message = OrderedDict({
        "Network": args.network,
        "data": args.data,
        "epoch":args.n_epoch,
        "batch_size": args.batch_size,
        "Optimizer":args.opt,
        "learning_rate":args.lr})
        #"l2_norm": args.l2_norm,
        #"Augmentation": args.aug})

    data = Load(args.data)
    (X_train, y_train), (X_test, y_test) = data.load()

    # 配列の整形と，色の範囲を0-255 -> 0-1に変換
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # 正解ラベルをダミー変数に変換
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    network = args.network(data.output_dim)
    model = network.build()

    model.compile(loss='categorical_crossentropy', optimizer=args.opt, metrics=['accuracy'])
    hist = model.fit(X_train, y_train, batch_size=args.batch_size, verbose=1, epochs=args.n_epoch, validation_split=0.1)
    
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',default="mnist",choices=["mnist","cifar10","cifar100","kuzushiji"], help='select dataset')
    parser.add_argument('--network',default=LeNet,choices=[LeNet], help='select dataset')
    parser.add_argument('--opt',default='sgd', choices=["sgd","Momentum","Adadelta","Adagrad","Adam","RMSProp"])
    parser.add_argument('--n_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()
    main(args)