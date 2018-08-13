## 「1章 ニューラルネットワークの復習」でつまづいたところ

### numpy

想像以上に numpy のリファレンスがよく分かっていないので、本書中で紹介のあった

* [100 numpy exercises](http://www.labri.fr/perso/nrougier/teaching/numpy.100/)
* [rougier/numpy-100](https://github.com/rougier/numpy-100)

をやっていく必要がありそう。


#### numpy の単体テスト

numpy の `ndarray` などのアサーションは numpy の `numpy.testing` ライブラリで実施する。

* 参照：[Test Support](https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.testing.html)


#### numpy の Type Hinting

探してみたが、まだ issue が open のまま。

* [Type hinting / annotation (PEP 484) for ndarray, dtype, and ufunc #7370](https://github.com/numpy/numpy/issues/7370)

まだ決定版はないのか。
