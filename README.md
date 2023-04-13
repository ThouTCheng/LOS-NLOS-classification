# LOS-NLOS-classification
depend CIR(channel impulse response) to identification  LOS or NLOS channel 

input: CIR,dim=2*256,where ‘2’ denotes the real part and imaginary part. ,‘256’ denotes the first 256 time-domain sampling points\n
label:one hot label;(e.p. [0,1] or [1,0])\n
output:1*2dim;\n
model sturcture:\n

![image](https://user-images.githubusercontent.com/41950342/231702503-10a57622-2a99-440a-8aeb-0d91383f303b.png)
