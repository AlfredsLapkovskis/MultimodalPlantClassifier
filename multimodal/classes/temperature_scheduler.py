from abc import ABC, abstractmethod
import numpy as np


class TemperatureScheduler(ABC):
    @abstractmethod
    def get_temperature(self): 
        pass
    @abstractmethod
    def update(self, iteration):
        pass


class InvExpTemperatureScheduler(TemperatureScheduler):

    def __init__(self, max_temperature, min_temperature, decay_rate, iteration=0):
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.decay_rate = decay_rate

        self.update(iteration)


    def get_temperature(self):
        return self.temperature


    def update(self, iteration):
        self.temperature = (self.max_temperature - self.min_temperature) * np.exp(-((iteration + 1.) / self.decay_rate) ** 2) + self.min_temperature

    
    def __str__(self) -> str:
        return f"{type(self).__name__}(max_temperature={self.max_temperature}, min_temperature={self.min_temperature}, decay_rate={self.decay_rate}, temperature={self.temperature})"