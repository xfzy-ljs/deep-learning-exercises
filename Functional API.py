from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import TensorBoard

#データのインポート
(x_train, y_train), (x_test, y_test)=mnist.load_data()

#スケール変換(60000*28*28のテンソルを60000*784に変換、つまり平面をベクトルに変換すること)
x_train = x_train.reshape(60000,784)
x_train = x_train/255.
x_test = x_test.reshape(10000,784)
x_test = x_test/255.

#クラスラベルデータをネットワークに対応するように変形する
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
tsb=TensorBoard(log_dir=r'D:\project\Yamaguchi University\Research\Practice\Functional\logs')

#Functional　APIによるモデルの構築 
input = Input(shape=(784,))
middle = Dense(units=64, activation='relu')(input)
output = Dense(units=10, activation='softmax')(middle)
model = Model(inputs=[input], outputs=[output])

#モデルのコンパイル
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#MINSTのデータセットを学習する
model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.2,
    callbacks=[tsb]
)