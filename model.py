import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense,Rescaling
from tensorflow.keras.callbacks import TensorBoard

#GPU設置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU起動")
    except RuntimeError as e:
        print("失敗:", e)
else:
    print("GPUが探知できません")

#基本情報
data_dir=r'D:\project\Yamaguchi University\Research\Practice\CNN\advance issues\raw_data'

img_size = (32,32)
batch_size = 32
epochs = 20

#データのインポート
categories = ["emphysema", "GGO", "honeycomb", "infiltration", "nodular", "normal"]
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,
    seed=123,
    class_names=categories
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,
    seed=123,
    class_names=categories
)

#正則化
normalization_layer = Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

model=Sequential()

#畳み込み層 
model.add(
    Conv2D(
        filters=32,
        input_shape=(img_size[0],img_size[1],3),
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

#Flatten layer追加(展開用、全結合層は２次元のテンソルしか入力に取ることができません)
model.add(Flatten())

#全結合層の追加
model.add(Dense(units=512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(categories),activation='softmax'))

#モデルのコンパイル
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
tsb=TensorBoard(log_dir=r'D:\project\Yamaguchi University\Research\Practice\CNN\advance issues\logs')

history_model1=model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=[tsb]
)

train_loss, train_acc = model.evaluate(train_ds)
val_loss, val_acc = model.evaluate(val_ds)
print(f"\n 训练集准确率: {train_acc:.4f} 训练集损失: {train_loss:.4f}")
print(f"\n 验证集准确率: {val_acc:.4f} 验证集损失: {val_loss:.4f}")
# use tensorboard --logdir "D:\project\Yamaguchi University\Research\Practice\CNN\logs" to check the graphs
