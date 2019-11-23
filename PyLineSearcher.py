import test
import time
class CGSSearch:
    def __init__(self, costfun, x = 0, d = 1, eps = 0.01):
        self.__costfun = costfun
        self.__x = x 
        self.__d = d
        self.__eps = eps
        
    def set_costfun(self, costfun):
        self.__costfun = costfun
    def set_x(self, x):
        self.__x = x

    def set_d(self, d):
        self.__d = d

    def set_eps(self, eps):
        self.__eps = eps

    def get_costfun(self):
        return self.__costfun

    def get_x(self):
        return self.__x

    def get_d(self):
        return self.__d
    
    def get_eps(self):
        return self.__eps

    def Phase1(self):
        func = self.__costfun
        phi = 1.618
        x_g = self.__x + self.__eps * (phi**(self.__d-1))
        while func(self.__x) > func(x_g):
            x_g = self.__eps * (phi**(self.__d-1))
            self.__d += 1
            # print(self.__x,x_g)
            # print(func(self.__x),func(x_g))
        return self.__x, round(x_g)

    def Phase2(self,interval_):
        func = self.__costfun
        rho = 0.382

        while(1):
            a = interval_[0]
            b = interval_[1]
            interval = b - a

            if interval < 0.3:
                val = (interval_[1]+interval_[0])/2
                return (val, func(val))
                break

            else:
                x1 = a + rho * interval
                x2 = a + ((1-rho) * interval)
                # print(x1,x2)
                # print(func(x1),func(x2))   
                if func(x1) < func(x2):
                    interval_ = [a,x2]  

                if func(x1) > func(x2):
                    interval_ = [x1,b]
        
    def RunSearch(self):
        print("-----Start Phase1-----")
        interval_ = self.Phase1()
        print(interval_)
        print("-----Start Phase2-----")
        print(self.Phase2(interval_))

func = test.TestLineFun3
CGS = CGSSearch(func,x = 0)
CGS.RunSearch()













# class CFiSearch:
#     def __init__(self, costfun, x = 0, d = 1, eps = 0.01):
#     def set_costfun(self, costfun):
#     def set_x(self, x):
#     def set_d(self, d):
#     def set_eps(self, eps):
#     def Phase1(self):
#     def Phase2(self):
#     def RunSearch(self):