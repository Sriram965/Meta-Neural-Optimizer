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
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique human-readable name for this task instance."""
        pass