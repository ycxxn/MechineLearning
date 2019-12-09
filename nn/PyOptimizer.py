import numpy as np 

class CForwardDiff:
    def __init__(self, costfun, x, dim, eps = 1e-5, percent = 1e-5):
        self.__costfun = costfun
        self.__x = x
        self.__dim = dim
        self.__eps = eps
        self.__percent = percent

    def set_costfun(self, costfun):
        self.__costfun = costfun   
    def set_x(self, x):
        self.__x = x
    def set_dim(self, dim):
        self.__dim = dim
    def set_eps(self, eps):
        self.__eps = eps
    def set_percent(self, percent):
        self.__percent = percent

    def GetGrad(self,x):
        dx = [0] * self.__dim
        grad = [0] * self.__dim
        # x = self.__x
        percent = self.__percent
        func = self.__costfun
        
        for i in range(self.__dim):
            dx[i] = x[i] * percent

        ori_x = np.copy(x)
        for i in range(self.__dim):
            for j in range(self.__dim):
                if j == i:
                    x[j] = ori_x[j] + dx[j]
                else:
                    x[j] = ori_x[j]
                
            grad[i] = (func(x)-func(ori_x)) / dx[i]
        
        return grad

class CGradDecent:
    def __init__(self, costfun, x0, dim, Gradient = 'Forward', LineSearch = "FiS", MinNorm = 0.001, MaxIter = 1000):
        self.costfun = costfun
        self.x0 = x0
        self.dim = dim
        self.Gradient = Gradient
        self.LineSearch = LineSearch
        self.MinNorm = MinNorm
        self.MaxIter = MaxIter
    
    def RunOptimizer(self):
        x_ = self.x0
        alpha = 0.01
        if self.Gradient == "Forward":
            grad = CForwardDiff(costfun=self.costfun , x=self.x0, dim=self.dim, percent=0.01)

        for i in range(self.MaxIter):
            grad.set_x(x_)
            d = grad.GetGrad()
            x_ = [x_[i]-alpha*d[i] for i in range(len(d))]
            print(x_,d)



        
# CFD = CForwardDiff(costfun=func1, x=x, dim=2, percent=0.01)
# print(CFD.GetGrad())
# CFD.set_x([2,0.1])
# print(CFD.GetGrad())


# CGD = CGradDecent(costfun=func2, x0=x, dim=2)
# CGD.RunOptimizer()

# for i in range(1000):
#     CFD.set_x(x)
#     d = CFD.GetGrad()
#     x = [x[i]-alpha*d[i] for i in range(2)]
#     print(x,d)
# x = [x[i]*2 for i in range(len(x))]

# print(x)
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = Axes3D(fig)

# x=[0, 0]
# # X, Y value
# x[0] = np.arange(0, 2, 0.01)
# x[1] = np.arange(0, 2, 0.01)
# x[0], x[1] = np.meshgrid(x[0], x[1])    # x-y 平面的网格
# z = func1(x)
# # height value
# ax.plot_surface(x[0], x[1], z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
# plt.show()