import torch
from sklearn.model_selection import train_test_split

def preprocess_data(X, y, X_scaler, y_scalers):
    Xf = X.reshape(-1, X.shape[-1])
    Xs = X_scaler.transform(Xf).reshape(X.shape)

    ys = []
    for i in range(y.shape[1]):
        ys.append(y_scalers[i].transform(y[:, i:i+1]))
    y = torch.tensor(np.hstack(ys), dtype=torch.float32)

    return torch.tensor(Xs, dtype=torch.float32), y

def split_dataset(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)
