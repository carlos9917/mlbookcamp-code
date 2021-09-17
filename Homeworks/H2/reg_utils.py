# some help functions from Alexey's notebook
import numpy as np
def prepare_X(df,base):
    df_num = df[base]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

#standard lin regression
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]

def split_data(data,seed=42):
    np.random.seed(seed)
    n = len(data)
    n_val = int(0.2 * n) #20 % for validation
    n_test = int(0.2 * n) # 20 % for testing
    n_train = n - (n_val + n_test) # USe the remainder for training
    idx = np.arange(n)
    np.random.shuffle(idx)
    df_shuffled = data.iloc[idx]
    df_train = df_shuffled.iloc[:n_train].copy()
    df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
    df_test = df_shuffled.iloc[n_train+n_val:].copy()
    return df_train,df_val,df_test


def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]


def plot_compare(y_train,y_pred,figname):
    from matplotlib import pyplot as plt
    import seaborn as sns
    fig=plt.figure(figsize=(6, 4))
    sns.histplot(y_train, label='target', color='#222222', alpha=0.6, bins=40)
    sns.histplot(y_pred, label='prediction', color='#aaaaaa', alpha=0.8, bins=40)
    plt.legend()
    plt.ylabel('Frequency')
    plt.xlabel('Log(Price + 1)')
    plt.title('Predictions vs actual distribution')
    plt.show()
    fig.savefig(figname)

def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)
