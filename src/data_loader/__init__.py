"""
This module is Data Loader for keras training.

it requires label_policy from external code.

DataGetter Support get one data pair from data_idx
DataLoader Support get one batch of data from batch_idx.
  - Support multi process
DataSequence Support get one batch of data from batch_idx.
  - This class inherits from the Keras Data Sequence.
"""