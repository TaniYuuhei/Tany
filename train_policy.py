import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import optimizers, serializers
import chainer.functions as F

from pydlshogi.common import *
from pydlshogi.network.policy import PolicyNetwork
from pydlshogi.features import *
from pydlshogi.read_kifu import *

import argparse
import random
import pickle
import os
import re

import logging

# 引数の定義
parser = argparse.ArgumentParser()
parser.add_argument('kifulist_train', type=str, help='train kifu list')
parser.add_argument('kifulist_test', type=str, help='test kifu list')
parser.add_argument('--batchsize', '-b', type=int, default=32,
                    help='Number of positions in each mini-batch')
parser.add_argument('--test_batchsize', type=int, default=512,
                    help='Number of positions in each test mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=1,
                    help='Number of epoch times')
parser.add_argument('--model', type=str, default='model/model_policy', 
                    help='model file name')
parser.add_argument('--state', type=str, default='model/state_policy', 
                    help='state file name')
parser.add_argument('--initmodel', '-m', default='',
                    help='Inirialize the model from given file')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='Resume the optimization from snapshot')
parser.add_argument('--log', default=None, help='log file path')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--eval_interval', '-i', type=int, default=1000, help='eval interval')
args = parser.parse_args()

# ログ出力設定
logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)

# モデル構築
model = PolicyNetwork()
model.to_gpu()

# 最適化手法SGD
optimizer = optimizers.SGD(lr=args.lr)
optimizer.setup(model)

# モデルとオプティマイザの状態の読み込み
if args.initmodel:
    logging.info('Load model from {}'.format(args.initmodel))
    serializers.load_npz(args.initmodel, model)
if args.resume:
    logging.info('Load optimizer state from {}'.format(args.resume))
    serializers.load_npz(args.resume, optimizer)

# 棋譜の読み込み
logging.info('read kifu start')

# 保存済みのpickleファイルがある場合、pickleファイルを読み込む
# train data
train_pickle_filename = re.sub(r'\..*?$', '', args.kifulist_train) + '.pickle'
if os.path.exists(train_pickle_filename):
    with open(train_pickle_filename, 'rb') as f:
        positions_train = pickle.load(f)
    logging.info('load train pickle')
else:
    positions_train = read_kifu(args.kifulist_train)

# test data
test_pickle_filename = re.sub(r'\..*?$', '', args.kifulist_test) + '.pickle'
if os.path.exists(test_pickle_filename):
    with open(test_pickle_filename, 'rb') as f:
        positions_test = pickle.load(f)
    logging.info('load test pickle')
else:
    positions_test = read_kifu(args.kifulist_test)

# 保存済みのpickleがない場合、pickleファイルを保存する
if not os.path.exists(train_pickle_filename):
    with open(train_pickle_filename, 'wb') as f:
        pickle.dump(positions_train, f, pickle.HIGHEST_PROTOCOL)
    logging.info('save train pickle')
if not os.path.exists(test_pickle_filename):
    with open(test_pickle_filename, 'wb') as f:
        pickle.dump(positions_test, f, pickle.HIGHEST_PROTOCOL)
    logging.info('save test pickle')
logging.info('read kifu end')

logging.info('train position num = {}'.format(len(positions_train)))
logging.info('test position num = {}'.format(len(positions_test)))

# ミニバッチデータ作成
def mini_batch(positions, i, batchsize):
    mini_batch_data = []
    mini_batch_move = []
    for b in range(batchsize):
        features, move, win = make_features(positions[i + b])
        mini_batch_data.append(features)
        mini_batch_move.append(move)
    return (Variable(cuda.to_gpu(np.array(mini_batch_data, dtype=np.float32))),
            Variable(cuda.to_gpu(np.array(mini_batch_move, dtype=np.int32))))

def mini_batch_for_test(positions, batchsize):
    mini_batch_data = []
    mini_batch_move = []
    for b in range(batchsize):
        features, move, win = make_features(random.choice(positions))
        mini_batch_data.append(features)
        mini_batch_move.append(move)
    return (Variable(cuda.to_gpu(np.array(mini_batch_data, dtype=np.float32))),
            Variable(cuda.to_gpu(np.array(mini_batch_move, dtype=np.int32))))

# 学習ループ
logging.info('start training')
itr = 0
sum_loss = 0
for e in range(args.epoch):
    # 訓練データのシャッフル
    # → シャッフルを行うと学習制度が上がる
    positions_train_shuffled = random.sample(positions_train, len(positions_train))

    itr_epoch = 0
    sum_loss_epoch = 0

    # ミニバッチ単位のループ
    for i in range(0, len(positions_train_shuffled) - args.batchsize, args.batchsize):
        
        # 順伝播
        x, t = mini_batch(positions_train_shuffled, i, args.batchsize)
        
        y = model(x)

        # 勾配を初期化
        model.cleargrads()

        # 損失計算
        loss = F.softmax_cross_entropy(y, t)

        # 誤差逆伝播
        loss.backward()
        
        # 勾配を使用しニューラルネットワークのパラメータ更新
        optimizer.update()

        # 一定間隔おきに評価
        itr += 1
        sum_loss += loss.data
        itr_epoch += 1
        sum_loss_epoch += loss.data

        if optimizer.t % args.eval_interval == 0:
            x, t = mini_batch_for_test(positions_test, args.test_batchsize)
            y = model(x)
            logging.info('epoch = {}, iteration = {}, loss = {}, accuracy = {}'
                        .format(optimizer.epoch + 1, optimizer.t, sum_loss / itr, F.accuracy(y, t).data))
            itr = 0
            sum_loss = 0
    
    # エポック単位ですべてのテストデータを評価
    logging.info('validate test data')
    itr_test = 0
    sum_test_accuracy = 0
    for i in range(0, len(positions_test) - args.batchsize, args.batchsize):
        x, t = mini_batch(positions_test, i, args.batchsize)
        y = model(x)
        itr_test += 1
        sum_test_accuracy += F.accuracy(y, t).data
        logging.info('epoch = {}, iteration = {}, train loss avr = {}, test accuracy = {}'
                    .format(optimizer.epoch + 1, optimizer.t, sum_loss_epoch / itr_epoch, sum_test_accuracy / itr_test))
    
    # 1エポック完了後オプティマイザに次のエポックの処理を通知
    optimizer.new_epoch()

# モデルとオプティマイザの状態の保存
logging.info('save the model')
serializers.save_npz(args.model, model)
logging.info('save the optimizer')
serializers.save_npz(args.state, optimizer)
