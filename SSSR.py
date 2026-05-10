import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

# define Lasso regression 
def sparse_reg(Theta, R, alpha=1e-4, tol=1e-5):
    lin_reg = linear_model.Lasso(alpha=alpha)
    Theta_orig = Theta.copy()

    lin_reg.fit(Theta, R)
    label0 = np.ones(Theta.shape[1])
    label1 = np.where(np.abs(lin_reg.coef_) < tol, 0, 1)

    while not np.array_equal(label0, label1):
        label0 = label1.copy()
        Theta = Theta_orig * label0  # 使用原始Theta做掩码
        lin_reg.fit(Theta, R)
        label1 = np.where(np.abs(lin_reg.coef_) < tol, 0, 1)

    pred = lin_reg.predict(Theta_orig).reshape(-1, 1)
    return lin_reg, pred, label1

# define Group Orthogonal Matching Pursuit 
def SSSR(Theta, R, sparsity_level, Para_number):
    R = R.flatten()
    N, p = Theta.shape
    residual = R.copy()
    support_set = []
    coef = np.zeros(p)
    Indexs = list(np.arange(Theta.shape[1]))
    support_set = []
    m = int(len(Theta)/Para_number)

    for _ in range(sparsity_level):
        loss_min = 1000
        index = -1
        for k in Indexs:
            support_cache = support_set.copy()
            support_cache.append(k)
            error = 0
            for j in range(Para_number):
                Target = R[m*j:m*(j+1)].reshape(-1,1)
                Feature = Theta[m*j:m*(j+1),support_cache]
                reg, pred = linear_reg(Feature,Target)
                error = error + rrmse(pred,Target)
            loss = error/Para_number
            if loss < loss_min:
                index = k
                loss_min = loss
        support_set.append(index)
        Indexs.remove(index)
    return support_set

# define linear regression
def linear_reg(Theta, R):
    reg = LinearRegression()
    reg.fit(Theta, R)
    pred = reg.predict(Theta).reshape(-1, 1)
    return reg, pred

def rrmse(x,y):
    return (np.mean((x-y)**2))**0.5/(np.mean(y**2))**0.5

