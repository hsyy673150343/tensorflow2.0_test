import tensorflow as tf
import tensorflow_datasets as tfds
import os

DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

for name in FILE_NAMES:
    text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL + name)

parent_dir = os.path.dirname(text_dir)

print(parent_dir)

'''将文本加载到数据集中'''
# 迭代整个文件，将整个文件加载到自己的数据集中。
# 每个样本都需要单独标记，所以请使用 tf.data.Dataset.map 来为每个样本设定标签。
# 这将迭代数据集中的每一个样本并且返回（ example, label ）对。
def labeler(example, index):
  return example, tf.cast(index, tf.int64)

labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
  lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
  '''
    a = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
    a.map(lambda x: x + 1)  # ==> [ 2, 3, 4, 5, 6 ]
  '''
  labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
  labeled_data_sets.append(labeled_dataset)


'''将这些标记的数据集合并到一个数据集中，然后对其进行随机化操作。'''
BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000

all_labeled_data = labeled_data_sets[0]

for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False)

# 使用 tf.data.Dataset.take 与 print 来查看 (example, label) 对的外观。numpy 属性显示每个 Tensor 的值。
# for ex in all_labeled_data.take(5):
#   print(ex)

''' 将文本编码成数字 '''
# 机器学习基于的是数字而非文本，所以字符串需要被转化成数字列表。为了达到此目的，我们需要构建文本与整数的一一映射。
'''
--1-- 建立词汇表

首先，通过将文本标记为单独的单词集合来构建词汇表。
        1.迭代每个样本的 numpy 值。
        2.使用 tfds.features.text.Tokenizer 来将其分割成 token。
        3.将这些 token 放入一个 Python 集合中，借此来清除重复项。
        4.获取该词汇表的大小以便于以后使用。
'''
tokenizer = tfds.features.text.Tokenizer()
vocabulary_set = set()
for text_tensor, _ in all_labeled_data:
  some_tokens = tokenizer.tokenize(text_tensor.numpy())
  vocabulary_set.update(some_tokens)

# print(vocabulary_set)
vocab_size = len(vocabulary_set)
# print(vocab_size) # 17178

'''
--2--  样本编码

通过传递 vocabulary_set 到 tfds.features.text.TokenTextEncoder 来构建一个编码器。
编码器的 encode 方法传入一行文本，返回一个整数列表。
'''
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
# 查看输出的样式
example_text = next(iter(all_labeled_data))[0].numpy()
print(example_text)

encoded_example = encoder.encode(example_text)
print(encoded_example)

# tf.py_function 将一个python函数封装到一个TensorFlow操作中，并急切地执行它。
# 现在，在数据集上运行编码器（通过将编码器打包到 tf.py_function 并且传参至数据集的 map 方法的方式来运行）。
def encode(text_tensor, label):
  encoded_text = encoder.encode(text_tensor.numpy())
  return encoded_text, label

def encode_map_fn(text, label):
  return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))

all_encoded_data = all_labeled_data.map(encode_map_fn)

'''
--3-- 将数据集分割为测试集和训练集且进行分支

在数据集被传入模型之前，数据集需要被分批。最典型的是，每个分支中的样本大小与格式需要一致。
但是数据集中样本并不全是相同大小的（每行文本字数并不相同）。
因此，使用 tf.data.Dataset.padded_batch（而不是 batch ）将样本填充到相同的大小。
'''
train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[]))

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[]))

# 现在，test_data 和 train_data 不是（ example, label ）对的集合，而是批次的集合。每个批次都是一对（多样本, 多标签 ），表示为数组。
# sample_text, sample_labels = next(iter(test_data))
# print(sample_text[0], sample_labels[0], sep='\n')

# 由于我们引入了一个新的 token 来编码（填充零），因此词汇表大小增加了一个。
vocab_size += 1

'''
--4-- 建立模型

'''
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 64))
# 下一层是 LSTM 层，它允许模型利用上下文中理解单词含义。
# LSTM 上的双向包装器有助于模型理解当前数据点与其之前和之后的数据点的关系。
'''
__init__(
    layer,
    merge_mode='concat',
    weights=None,
    backward_layer=None,
    **kwargs
)
'''
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
# 一个或多个紧密连接的层
# 编辑 `for` 行的列表去检测层的大小
for units in [64, 64]:
    model.add(tf.keras.layers.Dense(units, activation='relu'))

# 输出层。第一个参数是标签个数。
model.add(tf.keras.layers.Dense(3, activation='softmax'))

# 对于一个 softmax 分类模型来说，通常使用 sparse_categorical_crossentropy 作为其损失函数。
# 你可以尝试其他的优化器，但是 adam 是最常用的。
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

'''
--5-- 训练模型
'''
model.fit(train_data, epochs=3, validation_data=test_data)
eval_loss, eval_acc = model.evaluate(test_data)

print('\nEval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))