import pickle
import numpy as np



data = np.random.randn(30, 3, 77, 224, 224).astype(np.float32)
with open('./data/video_train.pkl', 'wb') as file:
    pickle.dump(data, file)
    
data = np.random.randn(10, 3, 77, 224, 224).astype(np.float32)
with open('./data/video_val.pkl', 'wb') as file:
    pickle.dump(data, file)
    
data = np.random.randn(30, 3, 51).astype(np.float32)
with open('./data/kine_train.pkl', 'wb') as file:
    pickle.dump(data, file)
    
data = np.random.randn(10, 3, 51).astype(np.float32)
with open('./data/kine_val.pkl', 'wb') as file:
    pickle.dump(data, file)
    
data = np.random.randint(0, 4, size=30, dtype=np.int64)
with open('./data/label_train.pkl', 'wb') as file:
    pickle.dump(data, file)               
    
data = np.random.randint(0, 4, size=10, dtype=np.int64)
with open('./data/label_val.pkl', 'wb') as file:
    pickle.dump(data, file)

    
