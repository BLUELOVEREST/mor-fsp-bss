import os
import pyroomacoustics as pra
import soundfile
import numpy as np
import argparse
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='iva盲源分离')
    parser.add_argument("inputwav",help="输入音频路径",type = str)
    parser.add_argument("outputwav",help="输出音频路径")
    parser.add_argument("Voiceprint_method",help="声纹识别方案: 可选 moran 或者 jd",type = str)
    args = parser.parse_args()
    INPUTWAV = args.inputwav
    OUTPUTWAV = args.outputwav


    inputwav=soundfile.read(INPUTWAV)
    L=2048
    X=np.array([pra.stft(inputwav[:, ch], L, L, transform=np.fft.rfft, zp_front=L//2, zp_back=L//2) for ch in range(inputwav.shape[1])])
    X = np.moveaxis(X, 0, 2)
    t_start = time.time()
    ite_num=50
    print("ite_num : ",ite_num)
    Y = pra.bss.auxiva(X, n_src=2,n_iter=ite_num, model = 'gauss', proj_back=True)
    t_stop = time.time() - t_start
    # print("Y.shape : ",Y.shape)
    y = np.array([pra.istft(Y[:,:,ch], L, L, transform=np.fft.irfft, zp_front=L//2, zp_back=L//2) for ch in range(Y.shape[2])])
    # print(y.shape)
    # print(y)
    # print(np.moveaxis(y,0,1))
    z = np.append(y[0],y[1])
    # # print("z.shape : ", z.shape)
    # soundfile.write(OUTPUTWAV,np.moveaxis(y,0,1),fs)
    soundfile.write(OUTPUTWAV,z,fs)


    print("iva_duration : ", t_stop)

