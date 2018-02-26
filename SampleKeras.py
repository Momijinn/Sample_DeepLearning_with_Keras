#!/usr/bin/python
# -*- Coding: utf-8 -*-
'''
3この入力を行い,4つに分類(0 ,1 2, 3)をするプログラム
'''
import numpy as np

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model

from keras.utils.vis_utils import model_to_dot
import matplotlib.pyplot as plt


DataNum = 1000
DataInput = 3
DataOutput = DataInput + 1 #0もラベル付するためDataInput + 1になる

#乱数で0 1を出力しこれは配列化,このセットを1000コ生成
Data = np.random.randint(low=0, high=2, size=(DataNum, DataInput))

#データラベルを作成
#kerasだとちょっと工夫が必要
#ラベル2であるとき -> [0. 0. 1. 0.] と変換
DataLabel = np.sum(Data, axis=1)
DataLabel = np_utils.to_categorical(DataLabel)


#--------------------------------------------------------------------------------------------------
#モデルの生成
model = Sequential() #初期化的なやつ
model.add(Dense(64, input_dim=DataInput)) #入力(input_dim)から隠れ層に何個渡すか DataInputに依存

#活性化
model.add(Activation('relu'))

#出力
model.add(Dense(DataOutput, activation='softmax'))

#コンパイル
model.compile('rmsprop', 'categorical_crossentropy', metrics=['accuracy'])

#学習
#epochs:学習回数
#validation_split:バリデーションデータの割合
#shuffle:データのシャッフル
#batch_size:バッチ数
hist = model.fit(Data, DataLabel, epochs=3, validation_split=0.3, batch_size=1, shuffle=True)
#--------------------------------------------------------------------------------------------------
print("----------------------")


#未知のデータを生成
Test = Data = np.random.randint(low=0, high=2, size=(1, DataInput))

#未知ラベルの生成
TestLabel = np.zeros(DataOutput)
TestLabel[np.sum(Test, axis=1)] = 1
print("AnswerLabel:",TestLabel)
print("AnswerVal:", TestLabel.argmax())
print("----------------------")

#numpyを分かりやすく表示させるための一行
np.set_printoptions(precision=6, suppress=True)


##推定
#テストデータ(Test)のみmodelにいれて結果を出力
#batch_size:バッチサイズ
#verbose:進行メッセージの表示 0=非表示, 1=表示
Result = model.predict(Test, batch_size=1, verbose=0)
print("Score:", Result)
print("ScoreVal:", Result.argmax() )
print("----------------------")

#モデルの図の生成
#show_shapes:データ数を表示させるか
plot_model(model, show_shapes=True, to_file='model.png')

#モデルの保存
JsonString = model.to_json()
open("model_result_data.json", 'w').write(JsonString)
model.save_weights("model_result_data.h5")
print("Save Model..")



# #accとlossのグラフ生成
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# # lossのグラフ
# plt.plot(loss, marker='.', label='loss')
# plt.plot(val_loss, marker='.', label='val_loss')
# plt.legend(loc='best', fontsize=10)
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()

# acc = hist.history['acc']
# val_acc = hist.history['val_acc']

# # accuracyのグラフ
# plt.plot(acc, marker='.', label='acc')
# plt.plot(val_acc, marker='.', label='val_acc')
# plt.legend(loc='best', fontsize=10)
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.show()