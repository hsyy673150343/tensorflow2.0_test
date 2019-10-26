import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


'''下载 IMDB 数据集'''
imdb = keras.datasets.imdb
# 参数 num_words=10000 保留了训练数据中最常出现的 10,000 个单词。为了保持数据规模的可管理性，低频词将被丢弃。
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

'''探索数据'''
# print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
# print(train_data[0])
# # 以下代码显示了第一条和第二条评论的中单词数量。由于神经网络的输入必须是统一的长度，我们稍后需要解决这个问题。
# print(len(train_data[0]), len(train_data[1]))

'''将整数转换回单词'''
# 这里我们将创建一个辅助函数来查询一个包含了整数到字符串映射的字典对象：
# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()
# print(word_index)
# 保留第一个索引
word_index = {k:(v+3) for k,v in word_index.items()}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# print(reverse_word_index)

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
# 使用 decode_review 函数来显示首条评论的文本：
# print(decode_review(train_data[0]))

'''准备数据'''
# 影评——即整数数组必须在输入神经网络之前转换为张量。这种转换可以通过以下两种方式来完成：

# 将数组转换为表示单词出现与否的由 0 和 1 组成的向量，类似于 one-hot 编码。例如，序列[3, 5]将转换为一个 10,000 维的向量，
# 该向量除了索引为 3 和 5 的位置是 1 以外，其他都为 0。然后，将其作为网络的首层——一个可以处理浮点型向量数据的稠密层。不过，
# 这种方法需要大量的内存，需要一个大小为 num_words * num_reviews 的矩阵。

# 或者，我们可以填充数组来保证输入数据具有相同的长度，然后创建一个大小为 max_length * num_reviews 的整型张量。
# 我们可以使用能够处理此形状数据的嵌入层作为网络中的第一层。
# 在本教程中，我们将使用第二种方法。

# 由于电影评论长度必须相同，我们将使用 pad_sequences 函数来使长度标准化：

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
# 现在让我们看下样本的长度：
print(len(train_data[0]), len(train_data[1]))
# 并检查一下首条评论（当前已经填充）：
print(train_data[0])

'''
构建模型

神经网络由堆叠的层来构建，这需要从两个主要方面来进行体系结构决策：
1.模型里有多少层？
2.每个层里有多少隐层单元（hidden units）？
'''
# 输入形状是用于电影评论的词汇数目（10,000 词）
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 创建一个验证集
'''
在训练时，我们想要检查模型在未见过的数据上的准确率（accuracy）。通过从原始训练数据中分离 10,000 个样本来创建一个验证集。
（为什么现在不使用测试集？我们的目标是只使用训练数据来开发和调整模型，然后只使用一次测试数据来评估准确率（accuracy））。
'''
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


'''
训练模型
 
在训练过程中，监测来自验证集的 10,000 个样本上的损失值（loss）和准确率（accuracy）
'''
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,# 通过在 20 个左右的 epoch 后停止训练来避免过拟合
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


'''评估模型 在测试集上评估模型'''
results = model.evaluate(test_data,  test_labels, verbose=2)

print(results)


'''
创建一个准确率（accuracy）和损失值（loss）随时间变化的图表
model.fit() 返回一个 History 对象，该对象包含一个字典，其中包含训练阶段所发生的一切事件：
'''

history_dict = history.history
print(history_dict.keys())



acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# “bo”代表 "蓝点"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b代表“蓝色实线”
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # 清除数字

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


'''
过拟合：
    模型在训练数据上的表现比在以前从未见过的数据上的表现要更好。
    在此之后，模型过度优化并学习特定于训练数据的表示，而不能够泛化到测试数据。
    
    通过在 20 个左右的 epoch 后停止训练来避免过拟合
'''

