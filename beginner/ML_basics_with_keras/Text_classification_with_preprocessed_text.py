import tensorflow as tf
from tensorflow import keras
import numpy as np

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