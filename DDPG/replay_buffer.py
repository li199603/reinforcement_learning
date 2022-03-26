import numpy as np

class ReplayBuffer:
    def __init__(self, value_dim, buffer_size, sample_size):
        self.buffer_size = buffer_size
        self.buffer = np.zeros((buffer_size, value_dim))
        self.count = 0
        self.sample_size = sample_size
    
    def store(self, data):
        index = self.count % self.buffer_size
        self.buffer[index] = data
        self.count += 1
    
    def sample(self):
        if self.count < self.sample_size:
            raise Exception("There are not enough data.")
        indicex = np.random.choice(min(self.count, self.buffer_size),
                                   self.sample_size,
                                   False)
        return self.buffer[indicex]
    
if __name__ == "__main__":
    buffer = ReplayBuffer(2, 5, 3)
    for i in range(10):
        data = [i, i]
        buffer.store(data)
        if i >= 3:
            print(buffer.sample())
    
        
        