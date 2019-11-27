import test
import time
class CGSSearch:
    def __init__(self, costfun, x = 0, d = 1, eps = 0.01, val_range = 0.3):
        self.__costfun = costfun
        self.__x = x 
        self.__d = d
        self.__eps = eps
        self.__val_range = val_range
        self.max_iter = 0
    def get_range(self):
        return self.__val_range
    def set_costfun(self, costfun):
        self.__costfun = costfun
    def set_x(self, x):
        self.__x = x
    def set_d(self, d):
        self.__d = d
    def set_eps(self, eps):
        self.__eps = eps
    def set_val_range(self, val_range):
        self.__val_range = val_range

    def Phase1(self):
        func = self.__costfun
        phi = 1.618
        x_g = self.__x + self.__eps * (phi**(self.max_iter))
        while(1):
            x_g = self.__eps * (phi**(self.max_iter))
            if func(self.__x) < func(x_g):
                break
            # print(func(self.__x), func(x_g))
            self.max_iter += 1

        return self.__x, round(x_g)
    
    def Phase2(self,interval_):
        max_iter = 0
        func = self.__costfun
        rho = 0.382

        while(1):
            a = interval_[0]
            b = interval_[1]
            interval = b - a

            if interval < self.__val_range:
                val = (interval_[1]+interval_[0])/2
                return (val, func(val),max_iter)

            else:
                x1 = a + rho * interval
                x2 = a + ((1-rho) * interval)
                # print(x1,x2)
                # print(func(x1),func(x2))   
                if func(x1) < func(x2):
                    interval_ = [a,x2]  

                if func(x1) > func(x2):
                    interval_ = [x1,b]

            max_iter+=1

    def RunSearch(self):
        print("-----Phase1-----")
        interval_ = self.Phase1()
        print("Interval : ({},{})".format(interval_[0], interval_[1]))
        print("-----Phase2-----")
        x = self.Phase2(interval_)
        print("Max_iter : "+str(x[2]))
        print(x[:2])

class CFiSearch(CGSSearch):
    def __init__(self, costfun, x = 0, d = 1, eps = 0.01, val_range = 0.3):
        super(CFiSearch, self).__init__(costfun)
        self.__costfun = costfun
        self.__x = x 
        self.__d = d
        self.__eps = eps
        self.__val_range = val_range
        self.max_iter = 0
    def set_costfun(self, costfun):
        self.__costfun = costfun   
    def set_x(self, x):
        self.__x = x
    def set_d(self, d):
        self.__d = d
    def set_eps(self, eps):
        self.__eps = eps
    def set_val_range(self, val_range):
        self.__val_range = val_range
        
    def fib(self,x):
        if x == 1:
            return 1
        if x == 2:
            return 2
        if x > 2:
            return self.fib(x-1)+self.fib(x-2)

    def Phase2(self,interval_):
        fib = self.fib
        n = 1
        # print(self.__eps)
        while fib(n+1) <= 2*(1+2*self.__eps)/self.__val_range:
            n+=1
        self.max_iter = n

        while(1):
            a = interval_[0]
            b = interval_[1]
            interval = b - a

            if n == 1:
                rho = 0.5 - self.__eps
                x1 = a + rho * interval
                x2 = a + ((1-rho) * interval)
                # print(x1,x2)
                # print(func(x1),func(x2))   
                if func(x1) < func(x2):
                    interval_ = [a,x2]  
                if func(x1) > func(x2):
                    interval_ = [x1,b]

                val = (interval_[1]+interval_[0])/2
                return (val, func(val), self.max_iter)

            else:
                rho = 1 - fib(n)/fib(n+1)
                x1 = a + rho * interval
                x2 = a + ((1-rho) * interval)
                # print(x1,x2)
                # print(func(x1),func(x2))   
                if func(x1) < func(x2):
                    interval_ = [a,x2]  

                if func(x1) > func(x2):
                    interval_ = [x1,b] 
            
            n-=1

    def RunSearch(self):
        print("-----Phase1-----")
        interval_ = self.Phase1()
        print("Interval : ({},{})".format(interval_[0], interval_[1]))
        print("-----Phase2-----")
        x = self.Phase2(interval_)
        print("Max_iter : "+str(x[2]))
        print(x[:2])

func = test.TestLineFun1
# CGS = CGSSearch(func)
# CGS.set_eps(0.3)
# # print(CGS.Phase1())
# # CGS.set_eps(0.3)
# CGS.RunSearch()
CFi = CFiSearch(func)
CFi.set_eps(0.05)
CFi.set_val_range(0.3)
CFi.RunSearch()