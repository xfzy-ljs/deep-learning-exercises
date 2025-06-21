from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
# 以下は三層構成のFNNです。
#データのインポート
(x_train, y_train), (x_test, y_test)=mnist.load_data()

#インポートしたデータの形を確認
print('x_train.shape:', x_train.shape)
print('x_test.shape:', x_test.shape)
print('y_train.shape:', y_train.shape)
print('y_test.shape:', y_test.shape)

#スケール変換(60000*28*28のテンソルを60000*784に変換、つまり平面をベクトルに変換すること)
x_train = x_train.reshape(60000,784)
x_train = x_train/255.
x_test = x_test.reshape(10000,784)
x_test = x_test/255.

#クラスラベルデータをネットワークに対応するように変形する
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#モデルの構築準備
model = Sequential()

#中間層の追加
model.add(
    Dense(
        units=64,
        input_shape=(784,),
        activation='relu'
    )
)

#出力層の追加
model.add(
    Dense(
        units=10,
        activation='softmax'
    )
)

#データの学習
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
tsb=TensorBoard(log_dir=r'D:\project\Yamaguchi University\Research\Practice\logs')
history_adam=model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.2,
    callbacks=[tsb]
)