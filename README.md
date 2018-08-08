# 『ゼロから作る Deep Learning 2』


## 1章 ニューラルネットワークの復習

前作『ゼロから作る Deep Learning』の内容の復習。

前作の内容をサマライズした内容は省略するとして、差分や、要復習だった部分を書く。

* 1.3.4 計算グラフ
    * 分岐ノード
        * 逆伝播のときは、上流からの勾配の和になる
    * Repeatノード
        * 分岐ノードの一般化。逆伝播のときは上流からの勾配の総和を取る
    * Sumノード
        * Repeatノードと順伝播と逆伝播が逆になっているノード
    * MatMulノード
        * 結論は分かったが、導出過程の説明が理解が浅いように思うので、再度確認が必要
* 1.3.5.1 Sigmoid 関数の微分の導出過程
    * 詳細は「付録 A」参照


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
