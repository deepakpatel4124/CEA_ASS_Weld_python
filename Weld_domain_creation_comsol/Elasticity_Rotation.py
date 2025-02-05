import numpy as np

Eul=[0,0,0]


# C=[250e9, 250e9, 250e9, 155e9, 155e9, 155e9, 125e9, 125e9, 125e9]
# C=[227e9, 244e9, 218e9, 109e9, 140e9, 146e9, 107e9, 119e9, 80e9]
C=[229e9, 229e9, 229e9, 131e9, 131e9, 131e9, 102e9, 102e9, 102e9]


def Rot_E (C,Eul):
    c11 = C[0]
    c22 = C[1]
    c33 = C[2]
    c12 = C[3]
    c13 = C[4]
    c23 = C[5]
    c44 = C[6]
    c55 = C[7]
    c66 = C[8]
    
    C=[[c11, c12, c13, 0, 0, 0],
       [c12, c22, c23, 0, 0, 0],
       [c13, c23, c33, 0, 0, 0],
       [0,   0,   0,  c44,0, 0],
       [0,   0,   0,   0, c55,0],
       [0,   0,   0,   0, 0,c66]]
    
    EuR=np.array(Eul)*(np.pi/180)
    
    R11=(np.cos(EuR[0])*np.cos(EuR[2]))-(np.cos(EuR[1])*np.sin(EuR[0])*np.sin(EuR[2]))
    R12=(np.cos(EuR[1])*np.cos(EuR[0])*np.sin(EuR[2]))+(np.sin(EuR[0])*np.cos(EuR[2]))
    R13=np.sin(EuR[1])*np.sin(EuR[2])
    R21=-(np.cos(EuR[0])*np.sin(EuR[2]))-(np.cos(EuR[1])*np.sin(EuR[0])*np.cos(EuR[2]))
    R22=(np.cos(EuR[1])*np.cos(EuR[0])*np.cos(EuR[2]))-(np.sin(EuR[0])*np.sin(EuR[2]))
    R23=np.sin(EuR[1])*np.cos(EuR[2])
    R31=np.sin(EuR[1])*np.sin(EuR[0])
    R32=-(np.sin(EuR[1])*np.cos(EuR[0]))
    R33=np.cos(EuR[1])
    
    
    T_sigma=[[R11**2, R21**2, R31**2, 2*R11*R21, 2*R21*R31, 2*R31*R11],
              [R12**2, R22**2, R32**2, 2*R12*R22, 2*R22*R32, 2*R32*R12],
              [R13**2, R23**2, R33**2, 2*R13*R23, 2*R23*R33, 2*R33*R13],
              [R11*R12, R21*R22, R31*R32, R11*R22+R12*R21, R21*R32+R31*R22, R31*R12+R32*R11],
              [R12*R13, R22*R23, R33*R32, R23*R12+R13*R22, R22*R33+R32*R23, R32*R13+R12*R33],
              [R11*R13, R21*R23, R31*R33, R13*R21+R11*R23, R23*R31+R21*R33, R33*R11+R31*R13]]
    
    T_epsilon=[[R11**2, R21**2, R31**2, R11*R21, R21*R31, R31*R11],
              [R12**2, R22**2, R32**2, R12*R22, R22*R32, R32*R12],
              [R13**2, R23**2, R33**2, R13*R23, R23*R33, R33*R13],
              [2*R11*R12, 2*R21*R22, 2*R31*R32, R11*R22+R12*R21, R21*R32+R31*R22, R31*R12+R32*R11],
              [2*R12*R13, 2*R22*R23, 2*R33*R32, R23*R12+R13*R22, R22*R33+R32*R23, R32*R13+R12*R33],
              [2*R11*R13, 2*R21*R23, 2*R31*R33, R13*R21+R11*R23, R23*R31+R21*R33, R33*R11+R31*R13]]
    
    T_sigma_inv=np.linalg.inv(T_sigma)
    
    #Dot=np.dot(T_sigma,T_sigma_inv)
    
    C_Rot=np.matmul(T_sigma_inv, np.matmul(C,T_epsilon))
    return C_Rot

C_Rot=Rot_E (C,Eul)

# Converting the matrix to a list of strings and flattening the matrix
C_Rot_list = [f"{np.round(C_Rot[i, j], 4):.0f}" for i in range(6) for j in range(6)]


print(np.round(C_Rot,4)/1e9)

# print(C_Rot_list)
         
         
