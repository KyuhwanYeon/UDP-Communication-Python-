# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 10:38:06 2018

@author: Kyuhwan
"""
###########################################
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import shelve
import time
import sys
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from keras import backend as K
K.clear_session()
#############################################

##############################
import socket
import struct
import numpy as np
import array
from Scheduler import perpetualTimer
from Scaler import MinMaxScaler_MinMax


import matplotlib.animation as animation
from matplotlib import style

#from main import PredSpdarr_MLP
#from main import PredSpdarr_RNN
#%% Server Class
global checkData

global Arg_Input
Arg_Input = np.zeros((1,5))





#%%

#%%
###########################################################################    

model = load_model('ModelArchive/RNNmodel_0227_MidanA.h5')
model2 = load_model('ModelArchive/MLPmodel_0227_MidanA.h5')
graph = tf.get_default_graph()
graph2 = tf.get_default_graph()
SaveVariables = shelve.open('ModelArchive/Variables_0227_MidanA.out','r')

for key in SaveVariables:
    globals()[key]=SaveVariables[key]
SaveVariables.close()
print('------------------Models are loaded------------------')
###########################################################################   


#%%
class NN_UDP:
    
    
    def __init__(self, ip="", port=10001,timeout=-1):
        self.sendDataSize= int(60)
        self.recvcount= 0
        self.SendData = np.zeros((28)) #228/8

        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        print('------------------Socket is created------------------')
        self.DataSize = np.zeros(5)
        self.Input = np.zeros((1,7))
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)
#        self.model=load_model('ModelArchive/RNNmodel_0122_Amsa.h5')
#        self.model2 = load_model('ModelArchive/MLPmodel_0122_Amsa.h5')
#        self.graph = tf.get_default_graph()
#        self.graph2 = tf.get_default_graph()
#        SaveVariables = shelve.open('ModelArchive/Variables_0122_Amsa.out','r')
#
#        for key in SaveVariables:
#            globals()[key]=SaveVariables[key]
#            SaveVariables.close()
        print('------------------Models are loaded------------------')
        
        if (timeout > 0):
            self.sock.settimeout(timeout)
    def recv(self):
        global checkData
        self.data, addr = self.sock.recvfrom(512)
        checkData=self.data
        
        
        ui32_startByte = self.data[0:4]
        ui8_HeaderType = self.data[4:5]
        ui32_SenderUID = self.data[5:9]
        i64_tmTime = self.data[9:17]   
        ui32_DataSize = self.data[17:21]   
        ui8_reserved = self.data[21:32]
        Char_StrMsgId = self.data[32:162]
        ui32_MsgSize = self.data[162:166]
        ui8_PackageId = self.data[166:167]
        ui8_PackageCount = self.data[167:168]
        
        Decoded_ui32_startByte = struct.unpack('<I' , ui32_startByte)
        Decoded_ui8_HeaderType = struct.unpack('<B' , ui8_HeaderType)
        Decoded_ui32_SenderUID = struct.unpack('<I' , ui32_SenderUID)
        Decoded_i64_tmTime = struct.unpack('<q' , i64_tmTime)
        
        
        Decoded_ui32_DataSize = struct.unpack('<I' , ui32_DataSize)
        
        
        Decoded_ui8_reserved = struct.unpack('<BBBBBBBBBBB' , ui8_reserved)
        
        
        Decoded_Char_StrMsgId = struct.unpack('<cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc' , Char_StrMsgId)
        
        
        Decoded_ui32_MsgSize = struct.unpack('<I' , ui32_MsgSize)
        Decoded_ui8_PackageId = struct.unpack('<B' , ui8_PackageId)
        Decoded_ui8_PackageCount = struct.unpack('<B' , ui8_PackageCount)  
        self.DataSize[0] = Decoded_ui32_DataSize[0]
        #########Default Header Size: 168 Bytes!#########################################
        #################################################################################
        ################    Distinguish Input ID ########################################
        ################    1 for Road Position, 2 for In vehicle Info ##################
        #################################################################################
#        print('start byte: ', Decoded_ui32_startByte)
#        print('HeaderType: ', Decoded_ui8_HeaderType)
#        print('SenderUID: ', Decoded_ui32_SenderUID)
#        print('tmTime: ', Decoded_i64_tmTime)
#        print('Datasize: ', Decoded_ui32_DataSize)
#        print('reserved: ', Decoded_ui8_reserved)
#        print('StrMsgID: ', Decoded_Char_StrMsgId)
#        print('MsgSize: ', Decoded_ui32_MsgSize)
#        print('PackageID: ', Decoded_ui8_PackageId)
#        print('PackageCount: ', Decoded_ui8_PackageCount)
        
        
#        Sin = self.data[168:176]
#        TestDecodedID=(Decoded_Char_StrMsgId[5].decode('utf-8'))
#        if TestDecodedID == '1':
#            DecodedSin = struct.unpack('<d' , Sin)         
#            print('DecodedSin: ', DecodedSin)
#        
# 
        
        
        Decoded_ID=(Decoded_Char_StrMsgId[5].decode('utf-8'))
        if Decoded_ID == '1':
            Data2_1 = self.data[168:176]
            RoadPos = struct.unpack('<d' , Data2_1) 
            Data2_2 = self.data[176:184]
            EngineSpeed= struct.unpack('<d' , Data2_2)
            Data2_3 = self.data[184:192]
            APS = struct.unpack('<d' , Data2_3)
            Data2_4= self.data[192:200]
            MasterCylPres = struct.unpack('<d' , Data2_4)
            Data2_5= self.data[200:208]
            RadDis = struct.unpack('<d' , Data2_5)
            Data2_6= self.data[208:216]
            RadSpd = struct.unpack('<d' , Data2_6)
            Data2_7= self.data[216:224]
            Speed = struct.unpack('<d' , Data2_7)    
#            print('EngSpd: ', EngineSpeed) 
#            print('APS: ', APS)           
#            print('MasterCylPres: ', MasterCylPres)
#            print('RadDis: ', RadDis) 
#            print('RadSpd: ', RadSpd)
#            print('Speed: ', Speed)
            RoadPos = RoadPos[0]
            EngineSpeed=EngineSpeed[0]
            APS=APS[0]
            MasterCylPres=MasterCylPres[0]
            RadDis=RadDis[0]
            RadSpd=RadSpd[0]
            Speed=Speed[0]
            if RadDis>146:
                RadDis= 0
            
            self.Input[0,0] = EngineSpeed
            self.Input[0,1] = APS
            self.Input[0,2] = MasterCylPres
            self.Input[0,3] = RoadPos
            self.Input[0,4] = RadDis
            self.Input[0,5] = RadSpd
            self.Input[0,6] = Speed

        global Arg_Input
        Arg_Input[0,0]=self.Input[0,0] 
        Arg_Input[0,1]=self.Input[0,1] 
        Arg_Input[0,2]=self.Input[0,2] 
        Arg_Input[0,3]=self.Input[0,3] 
        Arg_Input[0,4]=self.Input[0,6] 
        self.recvcount= 1
        
        
        
        
    def send(self):
        if self.recvcount ==1:
            global checkData
            global PredSpdarr_MLP
            global PredSpdarr_RNN
            
            PredSpdarr=np.reshape(PredSpdarr_RNN,15)
            checkData= PredSpdarr
            print(PredSpdarr)

            
            
#            Encoded_PredSpdarr=array.array('d', PredSpdarr).tostring()
#            Encoded_PredSpdarr=bytes('ddddddddddddddd',PredSpdarr)
            bytearr_senddata=bytearray(self.SendData)
            bytearr_recv=bytearray(self.data) 
            bytearr_senddata[0:168] = bytearr_recv[0:168]
            Encoded_dataSize =  struct.pack('I',self.sendDataSize)
            Encoded_StrID =  struct.pack('cccccccccc',b'o',b'u',b't',b'p',b'u',b't',b'_',b'r',b'a',b'w')
            
            bytearr_senddata[32:42] =Encoded_StrID
            bytearr_senddata[17:21]  = Encoded_dataSize
            bytearr_senddata[162:166] = Encoded_dataSize
            PredSpd1 =  struct.pack('f',PredSpdarr[0])
            PredSpd2 =  struct.pack('f',PredSpdarr[1])
            PredSpd3 =  struct.pack('f',PredSpdarr[2])
            PredSpd4 =  struct.pack('f',PredSpdarr[3])
            PredSpd5 =  struct.pack('f',PredSpdarr[4])
            PredSpd6 =  struct.pack('f',PredSpdarr[5])
            PredSpd7 =  struct.pack('f',PredSpdarr[6])
            PredSpd8 =  struct.pack('f',PredSpdarr[7])
            PredSpd9 =  struct.pack('f',PredSpdarr[8])
            PredSpd10 =  struct.pack('f',PredSpdarr[9])
            PredSpd11 =  struct.pack('f',PredSpdarr[10])
            PredSpd12 =  struct.pack('f',PredSpdarr[11])
            PredSpd13 =  struct.pack('f',PredSpdarr[12])
            PredSpd14 =  struct.pack('f',PredSpdarr[13])
            PredSpd15 =  struct.pack('f',PredSpdarr[14])
            
            bytearr_senddata[168:172] = PredSpd1
            bytearr_senddata[172:176] = PredSpd2
            bytearr_senddata[176:180] = PredSpd3
            bytearr_senddata[180:184] = PredSpd4
            bytearr_senddata[184:188] = PredSpd5
            bytearr_senddata[188:192] = PredSpd6
            bytearr_senddata[192:196] = PredSpd7
            bytearr_senddata[196:200] = PredSpd8
            bytearr_senddata[200:204] = PredSpd9
            bytearr_senddata[204:208] = PredSpd10
            bytearr_senddata[208:212] = PredSpd11
            bytearr_senddata[212:216] = PredSpd12
            bytearr_senddata[216:220] = PredSpd13
            bytearr_senddata[220:224] = PredSpd14
            bytearr_senddata[224:228] = PredSpd15
            


#            print(bytearr_senddata[17])
#            print(bytearr_senddata[162])                        
#            
            
            
#            print(Encoded_PredSpdarr)
#            bytearr_senddata[168:288] = Encoded_PredSpdarr
            EncodedSendData=bytes(bytearr_senddata)
            self.sock.sendto(EncodedSendData,('192.168.0.2',5000))
            
            
#            
#        print(self.Input)
    def PrintInput(self):
        print('RoadPos: ', self.Input[0,3])
        print('EngSpd: ', self.Input[0,0]) 
        print('APS: ', self.Input[0,1])           
        print('MasterCylPres: ', self.Input[0,2])
        print('RadDis: ', self.Input[0,4]) 
        print('RadSpd: ', self.Input[0,5])
        print('Speed: ', self.Input[0,6])
    def GetInput(self):
        global Arg_Input
        Arg_Input=self.Input
#        return Arg_Input
#        print(self.Input)
#%%
        
        
#%%

Predict_range = Test_MLP_Y.shape[1] # 15 (150m)
RNNInputRange = TestX.shape[1] # 20 (200m)
NumFeature =  TestX.shape[2]
global PrePos
PrePos=0
global Posarr
Posarr = []
global PredSpdarr_MLP
PredSpdarr_MLP=np.zeros((1,Predict_range))

global Input_RNN
Input_RNN=np.zeros((1,RNNInputRange,NumFeature))
global PredSpdarr_RNN
PredSpdarr_RNN=np.zeros((1,Predict_range))




global Posarr_Pred
Posarr_Pred = np.zeros((1,Predict_range))
global Spdarr
Spdarr = []
global count
count = 0



global RNN_Plot_Flag
RNN_Plot_Flag = 0      
def Get_Output_NN():

    global PrePos
    global PredSpdarr_MLP
    global Posarr_Pred
    global Spdarr
    global count
    global Input_RNN
    global PredSpdarr_RNN
    global RNN_Plot_Flag
#    Raw_Input=UDP_Communication.Arg_Input
    CurPos=Arg_Input[0,3]
    if CurPos-PrePos >10:
        
        
        Norm_Input = MinMaxScaler_MinMax(Arg_Input,Minarr,Maxarr)
        Norm_Input = np.reshape(Norm_Input,[1,len(Maxarr)] )
        global graph
        with graph.as_default():
            Output_MLP= model2.predict(Norm_Input)    
        PrePos=CurPos
        Posarr.append(PrePos)        
        PredSpdarr_MLP =Output_MLP*maxspeed
        for i in range(Predict_range):
            Posarr_Pred[0,i]=CurPos+(i+1)*10
        Spdarr.append(Arg_Input[0,4])
        
        
        ##############################################

        
        if count <20:
            Input_RNN[:,count,:] = Norm_Input
            count = count+1
        else :
            
            Input_RNN[:,RNNInputRange-1,:]= Norm_Input
            global graph2
            with graph2.as_default():
                Output_RNN=model.predict(Input_RNN)
            PredSpdarr_RNN =Output_RNN*maxspeed
            Input_RNN=np.roll(Input_RNN,-1,axis=1)
            RNN_Plot_Flag=1
            UDP_Server.send

        
        
        


#%%
Offline_True = Measured_vel[:]*maxspeed            
            
            
#%%
plt.close("all")
fig = plt.figure(1)
ax1 = fig.add_subplot(1,1,1)   
ax1.grid()    
def animate(i):
    global RNN_Plot_Flag


#    ax1.clear()
    ax1.plot(Posarr, Spdarr,color="black",label='Truth' ,zorder=10)

    
     #MLP Plot
    ax1.plot(Posarr_Pred[0], PredSpdarr_MLP[0],linestyle="None",marker="*",markersize=4,color="lime",zorder=5 )  
    ax1.plot(Posarr_Pred[0], PredSpdarr_MLP[0], color="lime",lw=0.3,label='MLP Prediction')
   
    #RNN Plot
    if RNN_Plot_Flag ==1:
        ax1.plot(Posarr_Pred[0], PredSpdarr_RNN[0],linestyle="None",marker="*",markersize=4,color="magenta",zorder=5)  
        ax1.plot(Posarr_Pred[0], PredSpdarr_RNN[0], color="magenta",lw=0.3,label='RNN Prediction')

   # ax1.legend()        

    
    ax1.set_title('Speed Prediction by Neural Network',fontsize = 11)
    ax1.set_ylabel('Speed [km/h]',fontsize = 10)
    ax1.set_xlabel('Roadway position [m]',fontsize = 10)

        
        
        
        
        
        
        
        
        
        
        
        
        
#%%
#%%
if __name__ == '__main__':
    

    
    
    
    UDP_Server = NN_UDP("192.168.0.1",10001,-1)
    
    RecvTimer = perpetualTimer(0.02,UDP_Server.recv)
    SendTimer = perpetualTimer(0.02,UDP_Server.send)
   # PrintOutTimer =  perpetualTimer(0.1,UDP_Server.PrintInput)
#    GetInputTimer =  perpetualTimer(0.01,UDP_Server.GetInput)
    Get_OutputTimer =  perpetualTimer(0.02,Get_Output_NN)


    RecvTimer.start()
    SendTimer.start()
#    PrintOutTimer.start()
    Get_OutputTimer.start()
    
    
    
    ani = animation.FuncAnimation(fig, animate, interval=20)
    plt.show()

        
    
    
#%%    
#    #%%
#if __name__ == '__main__':
#
#    UDP_Server = NN_UDP("192.168.0.1",10001,-1)
#    while True:
#        UDP_Server.recv()
##        UDP_Server.PrintInput()
#        A=UDP_Server.GetInput()
#        UDP_Server.send()
#        print(A)
#        
#    
#    
##    
##    
###    # interval 스케쥴러를 실행시키며, job_id는 "2" 입니다.
##
###    
##    count = 0
##    while True:
###        print ("Running main process...............")
##        time.sleep(1)
##        count += 1
##        if count == 10:
##            scheduler.kill_scheduler("1")
##            sys.exit()
##        
#        
