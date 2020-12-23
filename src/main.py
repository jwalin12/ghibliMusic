# Local import
from network import MusicGenerator

# Third-party import
import tensorflow as tf

""" The file used for agent execution.
    
    ...

Runs:
  Training
  Generation
  ...
"""

# NEURAL NETWORK EXECUTION HELPER FUNCTIONS.
@tf.function
def train_step(X, Y):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(X, training=True)
        loss = loss_object(Y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(Y, predictions)

@tf.function
def test_step(X, Y):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(X, training=False)
    t_loss = loss_object(Y, predictions)

    test_loss(t_loss)
    test_accuracy(Y, predictions)





# PRE-EXECUTION.
model = MusicGenerator(..., ...)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')



# MAIN EXECUTION.
EPOCHS = 5

train_ds    = ...
test_ds     = ...

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for X, Y in train_ds:
        train_step(X, Y)

    for test_X, test_Y in test_ds:
        test_step(test_X, test_Y)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )
