import tensorflow as tf
from tensorflow.keras import layers


'''
__init__(
    input_dim,
    output_dim,
    embeddings_initializer='uniform',
    embeddings_regularizer=None,
    activity_regularizer=None,
    embeddings_constraint=None,
    mask_zero=False,
    input_length=None,
    **kwargs
)

Input shape:
2D tensor with shape: (batch_size, input_length).

Output shape:
3D tensor with shape: (batch_size, input_length, output_dim).
'''

'''This layer can only be used as the first layer in a model.'''


embedding_layer = layers.Embedding(1000, 5)
result = embedding_layer(tf.constant([1,2,3]))
print(result.numpy())

result = embedding_layer(tf.constant([[0,1,2],[3,4,5]]))
# When given a batch of sequences as input, an embedding layer returns a 3D floating point tensor,
#  of shape (samples, sequence_length, embedding_dimensionality).
print(result.shape) # (2, 3, 5)