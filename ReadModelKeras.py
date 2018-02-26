#!/usr/bin/python
# -*- Coding: utf-8 -*-
'''
学習済みデータを読み込んで推定するプログラム
'''
from keras.models import model_from_json
import numpy as np

#学習されたモデル
model_json = "model_result_data.json"
model_h5 = "model_result_data.h5"

#入力データ数と出力データ数
DataInput = 3
DataOutput = DataInput + 1

#numpyを分かりやすく表示させるための一行
np.set_printoptions(precision=6, suppress=True)

#-----------------------------------------------
#学習済みモデルのロード
json_string = open(model_json).read()
model = model_from_json(json_string)
model.compile('rmsprop', 'categorical_crossentropy', metrics=['accuracy'])
model.load_weights(model_h5)
#--------------------------------------------
print("------------------------------------")

#未知のデータを生成
Test = Data = np.random.randint(low=0, high=2, size=(1, DataInput))
TestLabel = np.zeros(DataOutput)
TestLabel[np.sum(Test, axis=1)] = 1
print("AnswerLabel:",TestLabel)
print("AnswerVal:", TestLabel.argmax())
print("----------------------")


#推定
Result = model.predict(Test, batch_size=1, verbose=0)
print("Score:", Result)
print("ScoreVal:", Result.argmax() )
print("----------------------")

