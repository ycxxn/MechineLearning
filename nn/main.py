from PyOptimizer import CForwardDiff
from function import *

# x = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
x = np.array([0.01]*42)
CFD = CForwardDiff(cost_fun42,x,42)

alpha = 0.1

for i in range(10000):
    d = CFD.GetGrad(x)
    d = [-d[i] for i in range(len(d))]
    x = [x[i]+alpha*d[i] for i in range(len(d))]
    print("iter :{},loss : {}".format(i,cost_fun(x)))

y_pred = predict_3input(x,[0,0,0,0],[0,0,1,1],[0,1,0,1])
print(y_pred)