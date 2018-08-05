# 『ゼロから作る Deep Learning 2』


## 1章 ニューラルネットワークの復習

前作『ゼロから作る Deep Learning』の内容の復習。



## 環境セットアップ

### 要件

* インタプリタ
    * Python 3系 (venvで環境切り出し)
* ライブラリ
    * NumPy
    * Matplotlib
    * CuPy (オプション。非NVIDIA GPUなのでインストールできない)

    
### 構築手順

### インタプリタとvenvの設定

PyCharm を使うのでその設定

* 本体メニューの「PyCharm」
* 「Preferences」
* 「Project Interpreter」
* 「Add...」
* 「New environment」

で Base Interpreter で Python 3.x 系を選択してプロジェクトルートに `venv` ディレクトリに virtualenv の設定を配置。

### ライブラリ

* インタプリタの設定にモジュールのインストールリストが出ているので
* `numpy` と `matplotlib` を選択してインストール

環境が非NVIDIA GPUなのでCuPyはインストールに失敗するのでインストールしない。


### ライブラリのバージョンの指定

`pip freeze` の出力を保存。

```
$ pip freeze > requirements.txt
```




## 参考

* 本紙
    * [oreilly-japan/deep-learning-from-scratch-2](https://github.com/oreilly-japan/deep-learning-from-scratch-2) 
