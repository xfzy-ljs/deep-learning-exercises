import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from tensorflow.keras.callbacks import TensorBoard

#GPU設置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU 显存按需分配已启用")
    except RuntimeError as e:
        print("显存配置失败:", e)
else:
    print("未检测到GPU")

#データのインポート
(x_train, y_train), (x_test, y_test)=cifar10.load_data()

# データの大きさを確認
print('x_train.shape :', x_train.shape)
print('x_test.shape :', x_test.shape)
print('y_train.shape :', y_train.shape)
print('y_test.shape :', y_test.shape)

#正則化(预处理)
x_train = x_train/255.
x_test = x_test/255.

#クラスラベルデータをネットワークに対応するように変形する one-hot
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
model=Sequential()

#畳み込み層 
model.add(
    Conv2D(
        filters=32,
        input_shape=(32,32,3),
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'
    )
)
model.add(
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'
    )
)

#プーリング層
model.add(MaxPooling2D(pool_size=(2,2)))

#Dropout層追加
model.add(Dropout(0.25))

#畳み込み層とプーリング層追加
model.add(
    Conv2D(
        filters=64,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'
    )
)
model.add(
    Conv2D(
        filters=64,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'
    )
)
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#プーリング層追加号のモデルの出力形式(４次元のテンソル)
model.output_shape

#Flatten layer追加(展開用、全結合層は２次元のテンソルしか入力に取ることができません)
model.add(Flatten())
model.output_shape

#全結合層の追加
model.add(Dense(units=512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10,activation='softmax'))

#モデルのコンパイル
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
tsb=TensorBoard(log_dir=r'D:\project\Yamaguchi University\Research\Practice\CNN\logs')
history_model1=model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.2,
    callbacks=[tsb]
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n 测试集准确率: {test_acc:.4f}")
# use tensorboard --logdir "D:\project\Yamaguchi University\Research\Practice\CNN\logs" to check the graphs