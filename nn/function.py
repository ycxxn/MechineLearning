import numpy as np 
import math

def activation(x):
    y = [1/(1+math.exp(-x[i])) for i in range(len(x))]
    return y

def cost_fun(x):
    train = np.array([[0,0,0],[1,0,1],[0,1,1],[1,1,0]]) 
    w11 = x[0]
    w12 = x[1]
    b1  = x[2]
    w21 = x[3]
    w22 = x[4]
    b2  = x[5]
    w31 = x[6]
    w32 = x[7]
    b3  = x[8]
    x1 = train[:,0]
    x2 = train[:,1]
    y = train[:,2]
    z1 = activation(np.dot(x1,w11)+np.dot(x2,w12)+b1)
    z2 = activation(np.dot(x1,w21)+np.dot(x2,w22)+b2)
    y_pred = activation(np.dot(z1,w31)+np.dot(z2,w32)+b3)

    err = sum([(y[i]-y_pred[i])**2 for i in range(len(y))])
    return err

def predict(x,in1,in2):
    w11 = x[0]
    w12 = x[1]
    b1  = x[2]
    w21 = x[3]
    w22 = x[4]
    b2  = x[5]
    w31 = x[6]
    w32 = x[7]
    b3  = x[8]
    z1 = activation(np.dot(in1,w11)+np.dot(in2,w12)+b1)
    z2 = activation(np.dot(in1,w21)+np.dot(in2,w22)+b2)
    y_pred = activation(np.dot(z1,w31)+np.dot(z2,w32)+b3)
    return y_pred

def cost_fun16(x):
    train = np.array([
                    [0,0,0,0],
                    [0,0,0,1],
                    [0,1,0,1],
                    [0,1,1,0],
                    [1,0,0,1],
                    [1,0,1,0],
                    [1,1,0,0],
                    [1,1,1,1],
                    ]) 
    w11 = x[0]
    w12 = x[1]
    w13 = x[2]
    b1  = x[3]
    w21 = x[4]
    w22 = x[5]
    w23 = x[6]
    b2  = x[7]
    w31 = x[8]
    w32 = x[9]
    w33 = x[10]
    b3  = x[11]
    w41 = x[12]
    w42 = x[13]
    w43 = x[14]
    b4  = x[15]
    
    x1 = train[:,0]
    x2 = train[:,1]
    x3 = train[:,2]
    y = train[:,3]
    z1 = activation(np.dot(x1,w11)+np.dot(x2,w12)+np.dot(x3,w13)+b1)
    z2 = activation(np.dot(x1,w21)+np.dot(x2,w22)+np.dot(x3,w23)+b2)
    z3 = activation(np.dot(x1,w31)+np.dot(x2,w32)+np.dot(x3,w33)+b3)
    y_pred = activation(np.dot(z1,w41)+np.dot(z2,w42)+np.dot(z3,w43)+b4)

    err = sum([(y[i]-y_pred[i])**2 for i in range(len(y))])
    return err

def cost_fun42(x):
    train = np.array([
                    [0,0,0,0],
                    [0,0,0,1],
                    [0,1,0,1],
                    [0,1,1,0],
                    [1,0,0,1],
                    [1,0,1,0],
                    [1,1,0,0],
                    [1,1,1,1],
                    ]) 
    w11 = x[0]
    w12 = x[1]
    w13 = x[2]
    b1  = x[3]
    w21 = x[4]
    w22 = x[5]
    w23 = x[6]
    b2  = x[7]
    w31 = x[8]
    w32 = x[9]
    w33 = x[10]
    b3  = x[11]
    w41 = x[12]
    w42 = x[13]
    w43 = x[14]
    b4  = x[15]
    w51 = x[12]
    w52 = x[13]
    w53 = x[14]
    b5  = x[15]
    w61 = x[16]
    w62 = x[17]
    w63 = x[18]
    b6  = x[19]
    w71 = x[20]
    w72 = x[21]
    w73 = x[22]
    b7  = x[23]
    w81 = x[24]
    w82 = x[25]
    w83 = x[26]
    b8  = x[27]
    w91 = x[28]
    w92 = x[29]
    w93 = x[30]
    b9  = x[31]
    w101 = x[32]
    w102 = x[33]
    w103 = x[34]
    w104 = x[35]
    w105 = x[36]
    w106 = x[37]
    w107 = x[38]
    w108 = x[39]
    w109 = x[40]
    b10  = x[41]
    
    x1 = train[:,0]
    x2 = train[:,1]
    x3 = train[:,2]
    y = train[:,3]
    z1 = activation(np.dot(x1,w11)+np.dot(x2,w12)+np.dot(x3,w13)+b1)
    z2 = activation(np.dot(x1,w21)+np.dot(x2,w22)+np.dot(x3,w23)+b2)
    z3 = activation(np.dot(x1,w31)+np.dot(x2,w32)+np.dot(x3,w33)+b3)
    z4 = activation(np.dot(x1,w41)+np.dot(x2,w42)+np.dot(x3,w43)+b4)
    z5 = activation(np.dot(x1,w51)+np.dot(x2,w52)+np.dot(x3,w53)+b5)
    z6 = activation(np.dot(x1,w61)+np.dot(x2,w62)+np.dot(x3,w63)+b6)
    z7 = activation(np.dot(x1,w71)+np.dot(x2,w72)+np.dot(x3,w73)+b7)
    z8 = activation(np.dot(x1,w81)+np.dot(x2,w82)+np.dot(x3,w83)+b8)
    z9 = activation(np.dot(x1,w91)+np.dot(x2,w92)+np.dot(x3,w93)+b9)
    y_pred = activation(np.dot(z1,w101)+np.dot(z2,w102)+np.dot(z3,w103)+
                        np.dot(z4,w104)+np.dot(z5,w105)+np.dot(z6,w106)+
                        np.dot(z7,w107)+np.dot(z8,w108)+np.dot(z9,w109)+b10)

    err = sum([(y[i]-y_pred[i])**2 for i in range(len(y))])
    return err

def predict_3input(x,in1,in2,in3):
    w11 = x[0]
    w12 = x[1]
    w13 = x[2]
    b1  = x[3]
    w21 = x[4]
    w22 = x[5]
    w23 = x[6]
    b2  = x[7]
    w31 = x[8]
    w32 = x[9]
    w33 = x[10]
    b3  = x[11]
    w41 = x[12]
    w42 = x[13]
    w43 = x[14]
    b4  = x[15]
    w51 = x[12]
    w52 = x[13]
    w53 = x[14]
    b5  = x[15]
    w61 = x[16]
    w62 = x[17]
    w63 = x[18]
    b6  = x[19]
    w71 = x[20]
    w72 = x[21]
    w73 = x[22]
    b7  = x[23]
    w81 = x[24]
    w82 = x[25]
    w83 = x[26]
    b8  = x[27]
    w91 = x[28]
    w92 = x[29]
    w93 = x[30]
    b9  = x[31]
    w101 = x[32]
    w102 = x[33]
    w103 = x[34]
    w104 = x[35]
    w105 = x[36]
    w106 = x[37]
    w107 = x[38]
    w108 = x[39]
    w109 = x[40]
    b10  = x[41]
    z1 = activation(np.dot(in1,w11)+np.dot(in2,w12)+np.dot(in3,w13)+b1)
    z2 = activation(np.dot(in1,w21)+np.dot(in2,w22)+np.dot(in3,w23)+b2)
    z3 = activation(np.dot(in1,w31)+np.dot(in2,w32)+np.dot(in3,w33)+b3)
    z4 = activation(np.dot(in1,w41)+np.dot(in2,w42)+np.dot(in3,w43)+b4)
    z5 = activation(np.dot(in1,w51)+np.dot(in2,w52)+np.dot(in3,w53)+b5)
    z6 = activation(np.dot(in1,w61)+np.dot(in2,w62)+np.dot(in3,w63)+b6)
    z7 = activation(np.dot(in1,w71)+np.dot(in2,w72)+np.dot(in3,w73)+b7)
    z8 = activation(np.dot(in1,w81)+np.dot(in2,w82)+np.dot(in3,w83)+b8)
    z9 = activation(np.dot(in1,w91)+np.dot(in2,w92)+np.dot(in3,w93)+b9)
    y_pred = activation(np.dot(z1,w101)+np.dot(z2,w102)+np.dot(z3,w103)+
                        np.dot(z4,w104)+np.dot(z5,w105)+np.dot(z6,w106)+
                        np.dot(z7,w107)+np.dot(z8,w108)+np.dot(z9,w109)+b10)
    return y_pred