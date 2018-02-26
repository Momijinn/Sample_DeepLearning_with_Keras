Sample_DeepLearning_with_Keras
====
Kerasを使用した簡単なディープラーニングのサンプルプログラム

## Description
Kerasを利用した簡単なデータによるディープラーニングのプログラム

また学習済みモデルを保存しロードするプログラム付き

SampleKeras.pyでは、3個のデータを入力とし、学習を経て最後に4つ出力を行うプログラムです

ReadModelKeras.pyでは、SampleKeras.pyで作成された学習済みモデルをロードし評価を行うプログラムです


</br>

詳細は下記のサイトを参考

http://www.autumn-color.com/?p=1357

## Demo
```bash
$ python SampleKeras.py
C:\Users\******\AppData\Local\Programs\Python\Python36\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
Train on 700 samples, validate on 300 samples
Epoch 1/3
2018-02-26 16:24:34.352121: I C:\tf_jenkins\workspace\rel-win\M\windows\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
700/700 [==============================] - 2s 3ms/step - loss: 0.9826 - acc: 0.6014 - val_loss: 0.7124 - val_acc: 0.9067
Epoch 2/3
700/700 [==============================] - 2s 2ms/step - loss: 0.5290 - acc: 0.9314 - val_loss: 0.3556 - val_acc: 1.0000
Epoch 3/3
700/700 [==============================] - 2s 2ms/step - loss: 0.2376 - acc: 0.9929 - val_loss: 0.1397 - val_acc: 1.0000
----------------------
AnswerLabel: [1. 0. 0. 0.]
AnswerVal: 0
----------------------
Score: [[0.945631 0.051894 0.00243  0.000045]]
ScoreVal: 0
----------------------
Save Model..


$ python .\ReadModelKeras.py
C:\Users\*****\AppData\Local\Programs\Python\Python36\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
2018-02-26 17:31:54.097406: I C:\tf_jenkins\workspace\rel-win\M\windows\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
------------------------------------
AnswerLabel: [0. 1. 0. 0.]
AnswerVal: 1
----------------------
Score: [[0.002073 0.922902 0.073967 0.001058]]
ScoreVal: 1
----------------------
```


## Requirement
* python 3系
* Keras

## Usage
* 学習するプログラム

    ```bash
    $ python SampleKeras.py
    ```

* 学習モデルのロードからの使用プログラム

    ```bash
    $ python ReadModelKeras.py
    ```


## Licence
This software is released under the MIT License, see LICENSE.

## Author
[Twitter](https://twitter.com/momijinn_aka)

[Blog](http://www.autumn-color.com/)