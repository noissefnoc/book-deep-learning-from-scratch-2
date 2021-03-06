# 『ゼロから作る Deep Learning 2』


## 1章 ニューラルネットワークの復習

前作『ゼロから作る Deep Learning』の内容の復習。

前作の内容をサマライズした内容は省略するとして、差分や、要復習だった部分を書く。


### 1.3.4 計算グラフ

* 分岐ノード
    * 逆伝播のときは、上流からの勾配の和になる
* Repeatノード
    * 分岐ノードの一般化。逆伝播のときは上流からの勾配の総和を取る
* Sumノード
    * Repeatノードと順伝播と逆伝播が逆になっているノード
* MatMulノード
    * 結論は分かったが、導出過程の説明が理解が浅いように思うので、再度確認が必要

### 1.3.5.1 Sigmoid 関数の微分の導出過程

* 詳細は「付録 A」参照
* 重みの更新のステップ
    * 訓練データの中からランダムに複数のデータを選び出す
    * 誤差逆伝播方式により、各重みパラメータに関する損失関数の勾配を求める
    * 勾配を使って重みパラメータを更新する
    * 上記3ステップを必要な回数繰り返す

### 1.4.3 学習用のソースコード

一般的な学習の構成

* ハイパーパラメータの設定
* データの読み込み、モデルとオプティマイザの生成
* データのシャッフル
* 勾配を求め、パラメータを更新
* 定期的に学習経過を出力


### 1.4.4 Trainer クラス

* `1.4.3 学習用のソースコード` をクラスにまとめたもの
* `ch01.train_custom_loop` を `common.trainer` にまとめ
* `fit()` を学習用のインターフェースと持たす


### 1.5.1 ビット精度

経験的に精度を落とすことなく、以下の事項が知られている

* (64ビットではなく)32ビット浮動小数の利用で充分
* 推論に限って言えば16ビット浮動小数で充分
* しかし、H/W側が提供しているのが32ビット演算なので、基本的に32ビットを使う
* 保存のときだけ容量削減のため16ビットで実施する


### 1.5.2 GPU (CuPy)

* numpy 互換インターフェースのGPU利用版
* 切り替えについては `common.config`、`common.np`、`common.layers`あたりを参照。
* 4章からのコードで使えるとのことだが、対応GPUを持っていないので試せないと思う。


### 1.6 章のまとめ

ニューラルネットワークの実装では以下を作るとよい

* 構成要素の `Layer` クラス (I/Fとして `forward` と `backward` を持つ)
* 学習のための `Trainer` クラス (I/Fとして `fit` を持つ)
    

## 2章 自然言語と単語の分散表現

### 2.1 自然言語処理とは

本書で扱うのは

* シソーラスによる手法 (2.2 と 付録B)
* カウントベースの方法 (2.3)
* 推論ベースの手法 (word2vec) (次章)

シソーラスとカウントベースはやったことがあるので、概ねパスするかも。


### 2.2 シソーラスによる手法

コードを伴う部分は WordNet のコーパスを使って付録Bでやるとのことだが、パス


### 2.3 カウントベースの方法

一旦英文が対象


#### 2.3.3 分布仮説 (distributional hypothesis)

* 「単語の意味は、周囲の単語によって形成される」という考え方
* (例) "drink" の近くには飲み物が来やすい
* 「コンテキスト」：注目する単語に対して、その周囲に存在する単語を指す
* 「ウィンドウサイズ」：注目する単語を中心にして、周囲の語をどの程度見るか


#### 2.3.4 共起行列 (cooccurrence matrix)

* ウィンドウサイズ内で出現した単語を行列にしたもの
* コーパスの単語が縦横に入り、値は出現回数になる


#### 2.3.5 ベクトル間の類似度

* コサイン類似度
* ゼロ除算を避けるためにごく小さい値(1e-8)を分母に足す
* 上記や浮動小数演算により、誤差が生じるので UnitTest のさいには `assertAlmostEquel` を使う


#### 2.3.6 類似単語のランキング表示

* コサイン類似度の単語ランキングを作成する


#### 2.4.2 次元削減

* 次元削減：ベクトルの必要な情報を残したまま、データ量を削減すること
* 疎なベクトル：ベクトル中の多くの要素が0であるベクトルのこと
* 特異値分解(SVD)：任意の行列を3つの行列の積へと変換


#### 2.4.5 PTBデータセットでの評価

PTBデータセットで共起行列を作って正の相互情報量を出し、SVDで単語ベクトルを次元圧縮する。


### 2.6 まとめ

カウントベースの単語の類似度を測るための手法について学んだ。

次の章で簡易的な `word2vec` を実装する。


## 3章 word2vec

#### 3.1.1 カウントベース手法の問題点

* 計算量が多いこと(例：SVDは n x n 行列に対して計算量がO(n^3)かかる)
* コーパス全体のデータを使わないといけない


#### 3.1.2 推論ベースの手法の概要

* コンテキストを入力。出力を各単語の出現確率とするようなモデルを作成する
* コンテキストをベクトル表現できれば(one-hot表現が紹介されている)計算可能になりそう


### 3.2 シンプルな word2vec

word2vecには

* continuous bag-of-words(CBOW)
* skip-gram

の2モデルがあり、まずはCBOWの方を先に説明する(skip-gramは3.5.2)

#### 3.2.1 CBOWモデルの推論処理

* 入力層が複数
* 中間層が1層(全結合による変換処理の値が平均されたもの)
* 出力層(スコア算出層で、Softmaxをまだかけていない)

ものから構成される。


#### 3.2.3 word2vecの重みと分散表現

入力と出力側の両方に重みがあるがどちらを使うか、という問題があるが、word2vec(特にskip-gramモデル)の場合は入力側の重みだけを使う。


### 3.3 学習データの準備

#### 3.3.1 コンテキストとターゲット

* word2vecで用いるニューラルネットワークの
    * 入力：コンテキスト。正解を囲む単語
    * 正解ラベル：コンテキストに囲まれた中央の単語

とする


### 3.4 CBOW モデルの実装

CBOWモデルの実装。 

optimizer として特段説明なしに Adam が出てくるが、構造自体はこれまでのニューラルネットの実装と同じ。


### 3.5 word2vec に関する補足

#### 3.5.1 CBOW モデルと確率

負の対数尤度の相加平均。


#### 3.5.2 skip-gram モデル

* CBOWモデル
    * 複数のコンテキストから中央の単語(ターゲット)を推測する
    * 学習は高速
* skip-gramモデル
    * 中央の単語(ターゲット)から周囲の複数ある単語(コンテキスト)を推測する
    * 精度は高い


### 3.6 まとめ

* 推論ベースの方法は、副産物として単語の分散表現が得られる
* word2vecは重みの再学習ができるため、単語の分散表現の更新や追加が効率的に行える


## 4章 word2vec の高速化

### 4.1 word2vec の改良その1

one-hot表現では行列積演算が必要になり、計算量が多い。

one-hotベクタとの行列積の結果は、該当行の抜き出しになるので、行列のインデックスアクセスで代用する。

この処理を行う層を `Embedding Layer` と呼ぶ

* 順伝播：インデックスアクセス
* 逆伝播：前の層から伝わってきた勾配の総和


### 4.2 word2vec の改良その2

問題を多値分類から二値分類に変換し、各レイヤでの計算量を抑える。

そのために Negative Sampling をはじめとした手法を適応していく。


### 4.3 改良版 word2vec の学習

`4.3.2 CBOW モデルの学習コード` のコードは本書中で「学習には多くの時間(半日程度)が必要になります」と記載があり、CPUモードで Macbook Pro 13" (late 2016)で3時間程度かかった。 


### 4.5 まとめ

* Embedding レイヤは単語の分散表現を格納し、順伝播において該当する単語ID のベクトルを抽出する
* Negative Sampling は負例をサンプリングする手法であり、これを利用することで、多値分類を二値分類として扱うことができる
* word2vecの特徴
    * 語彙数と計算量が比例関係にあるので、近似計算などで高速化する
    * 単語の分散表現は、単語の意味が埋め込まれたものであり、似たコンテキストで使われる単語は単語ベクトルの空間上で近い性質を持つ
    * 単語の分散表現によって、類似問題をベクトルの加算と減算に置き換えることができる
    * 転移学習の点で特に重要であり、単語の分散表現はさまざまな自然言語処理のタスクに利用できる


## 5章 リカレントニューラルネットワーク(RNN)

* フィードフォワード
    * これまでに説明のあったネットワークのタイプ
    * 流れが一方向のネットワーク
    * 単純で仕組みも理解をしやすく、それなりの問題にも対応できる
    * 時系列データをうまく扱えないという問題がある(時系列データのパターンを十分に学習できない)
* リカレントニューラルネットワーク
    * 時系列データを扱える


### 5.1.1 word2vec を確立の視点から眺める

CBOWモデルではコンテキストの単語(指定の単語の両周囲の語)から単語が決まると考えるモデル。

コンテキストを想定単語の左側の語(その後より前に出現した語)で推測を行うモデルを考える。ここからの発展で言語モデルについて考える。


### 5.1.2 言語モデル

言語モデルは単語の並びに対して確率を与える。並びがよく起こるもの(自然なもの)ほど高い確率を出す。

一番右の語からそれ以前の語が出現した後の事後確率として分解していってこれを条件付き言語モデルと呼ぶ。


### 5.1.3 CBOW モデルを言語モデルに？

`5.1.1` で CBOW　モデルを両サイドで挟んでいる語をコンテキストとして考えていたのを、左二つ前に変換した話があった。

それと同じように考えると言語モデルで考えた並びに近似ができる。これは直前2つの単語だけに依存して次の単語が決まるモデルなので、2階マルコフ連鎖になる。


CBOW モデルはコンテキストのサイズを大きくしてもコンテキスト内の単語の並びが無視されるという問題がある。

CBOW モデルのネットワークレイヤーを考え直すと、中間層は単語ベクトルの和を取るので、単語の順によらず同じ結果になる。

解決方法としては

* 中間層として結合した単語ベクトルを使う(Neural Probablistic Language Modelなど)
* リカレントニューラルネットワーク(RNN)を使う

RNNはコンテキストの情報を記憶するメカニズムがある。


## 5.2 RNN とは


### 5.2.1 循環するニューラルネットワーク

ループする経路を持つことにより、過去の情報を記憶しながら、最新のデータへと更新される仕組みが作れる。


### 5.2.2 ループの展開

RNN のループを展開すると隠れ層が次の層の入力に入る構成に分解できる。


### 5.2.3 Backpropagation Through Time

時間方向に展開したニューラルネットワークの誤差逆伝播法。

長い時系列データを学習する場合、時間サイズの大きさに比例して

* コンピュータの計算リソースが増加
* 逆伝播時の勾配が不安定になる

というデメリットがある。


### 5.2.4 Truncated BPTT

逆伝播のときだけ、適当な塊の区間だけで逆伝播を行うようにする。
※ 順伝播のときは通常通り伝播させる


### 5.2.5 Truncated BPTT のミニバッチ学習

これまでミニバッチ学習のときにはデータをランダムに取得していたが、シーケンシャルにデータを与える必要があるため、データの開始インデックスをずらした入力を使う。


## 5.3 RNN の実装

### 5.3.1 RNN レイヤの実装

順伝播は計算グラフ通り。


### 5.3.2 Time RNN レイヤの実装

Time RNN レイヤは T(任意の数)個の RNN レイヤから構成される。

各RNNレイヤの入力を x_0, x_1, ..., x_t-1、出力が h_0, h_1, ..., h_t-1 として、それぞれをベクトル表記にしてそれぞれ `xs` , `hs` とする。

Time RNN レイヤはそのレイヤを複数結合して構成され、レイヤ間の隠れ状態を `h` とする。隠れ状態を伝えるか伝えないかは選べるような実装にする。


## 5.4 時系列データを扱うレイヤの実装

### 5.4.1 RNNLM の全体図

以下のレイヤ構成をしている。

* Embedding：単語IDを単語の分散表現へと変換
* RNN
* Affine
* Softmax

このレイヤ構成で、これまで入力された単語を元に次に出現する単語を予測できる。


### 5.4.2 Time レイヤの実装

* Time Affine：AffineレイヤをT個用意して各時刻のデータ(x_0, ..., x_t-1)を処理する。AffineレイヤをT個生成する実装ではなく、行列計算としてまとめて処理する実装が効率的
* Time Softmax with Loss：損失Lは L = (L_0 + ... + L_t-1) / T になる


## 5.5 RNNLM の学習と評価

### 5.5.1 RNNLM の実装

SimpleRnnlmという構成を考え、RNNとAffineレイヤの初期値としてXavierの初期値を利用する。

Xavierの初期値は前層のノードの個数をnとした場合、1 / (n^-1/2) の標準偏差を持つ分布。


### 5.5.2 言語モデルの評価

パープレキシティを評価に使う。パープレキシティは確率の逆数。

RNN で単語の発生確率が出るので、その逆数を取ると、発生確率が高ければ高いほどパープレキシティが低くなる。


### 5.5.3 RNNLM の学習コード

PTB データセットを利用して RNNLM の学習を行う。

現在の RNNLM の実装では

* 精度が出ない
* 大きいデータセットに対応できない

という問題があるため、先頭から1,000個の単語のみを対象とし、学習によってパープレキシティが下がっていくことを観察するために使う。

コードの構成は

* ハイパーパラメータの設定
* 学習データの読み込み
* モデルの生成
    * ミニバッチの各サンプルの読み込み開始位置を計算(スライドして使うので)
    * ミニバッチの取得
    * 勾配計算してパラメータを更新(ここで最小化問題にしたいので、高くなるほどよい確率ではなく、小さくなるほどよいパープレキシティを採用している)
    * エポックごとにパープレキシティを評価する
    

### 5.5.4 RNNLM の Trainer クラス

`common/trainer.py` の `RnnlmTrainer` にある。使い方は前節と同じ。


### 5.6 まとめ

RNN はデータを循環させることで「隠れ状態」を記憶する(循環するので「前の状態」が少なくとも分かるはず)。

実際には RNN ではうまくいかない例があるので、RNN 以外のレイヤとして

* LSTM レイヤ
* GRU レイヤ

を次章以降で説明していく。


## 6章 ゲート付き RNN

## 6.1 RNN の問題点

RNN は時系列データの長期の依存関係を学習するのには向いていない。

理由としては BPTT (Backpropagation Through Time) において勾配消失/勾配爆発がおこるため。


### 6.1.1 RNN の復習

5章の復習なので省略。


### 6.1.2 勾配消失もしくは勾配爆発

現在のシンプルな RNN レイヤでは時間を遡るに従って、勾配消失ないしは勾配爆発が起こる。その理由は次節。


### 6.1.3 勾配消失もしくは勾配爆発の原因

* tanh の逆関数の値が1.0以下しか取りえない
* MatMul の勾配変動が、重みによって単調増加するか単調減少するかは変化するが、指数関数的に変化する


### 6.1.4 勾配爆発への対策

* 勾配爆発への対策は勾配クリッピング(gradients clipping)で対応する
    * パラメーター群のL2ノルムが閾値を超えた場合、パラメータ群のL2ノルム分の閾値を掛け直す


### 6.2 勾配消失とLSTM

勾配爆発は勾配クリッピングで対応できたものの、勾配消失にはまだ対応できていない。

勾配消失への対応策としてはLSTMとGRUがあり、それぞれ以下の箇所で説明する。

* LSTM：この節で詳細説明
* GRU：付録Cにて説明


### 6.2.1 LSTMのインターフェース

計算グラフの簡略化のため `tanh(h_t-1 W_h + x_t W_x + b)` を長方形の `tanh` で表現する。

RNNとLSTMを比較するとLSTMはRNNに `c` (セル)という LSTM 用の記憶回路が追加されている。

記憶セルの特徴は LSTMレイヤ内だけでデータの受け渡しをする(＝他のレイヤへは出力しない)。そのため、他レイヤへの出力は隠れ状態ベクトルの `h` のみで変わらない。


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
