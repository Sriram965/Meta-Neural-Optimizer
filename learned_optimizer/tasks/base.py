import tensorflow as tf
from abc import ABC, abstractmethod


class BaseTask(ABC):

    @abstractmethod
    def sample_theta(self) -> tf.Tensor:
        pass

    @abstractmethod
    def loss(self, theta: tf.Tensor) -> tf.Tensor:
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        pass