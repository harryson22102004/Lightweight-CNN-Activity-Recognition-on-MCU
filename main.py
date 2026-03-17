import numpy as np
 
# Simulate TFLite-compatible MobileNetV2-style model for HAR
ACTIVITIES=['walking','running','cycling','sitting','standing','lying']
N_FEATURES=3*50  # 3 axes, 50 time steps
 
def extract_features(window):
    feats=[]
    for axis in range(3):
        d=window[:,axis]
        feats+=[d.mean(),d.std(),d.max()-d.min(),
                np.sqrt(np.mean(d**2)),  # RMS
                np.sum(np.diff(np.sign(np.diff(d)))!=0),  # zero crossings
                *np.fft.rfft(d)[:5].real.tolist()]
    return np.array(feats)
 
class TinyConvNet:
    """Simulate a TFLite quantized conv net (INT8) for MCU deployment."""
    def __init__(self):
        self.w1=np.random.randn(32,3,5)*0.1
        self.w2=np.random.randn(64,32,3)*0.1
        self.fc=np.random.randn(len(ACTIVITIES),64*10)*0.1
    def conv1d(self,x,w):
        out=np.zeros((w.shape[0],x.shape[1]-w.shape[2]+1))
        for i in range(w.shape[0]):
            for t in range(out.shape[1]):
                out[i,t]=np.sum(w[i]*x[:,t:t+w.shape[2]])
        return np.maximum(0,out)
    def forward(self,x):
        h=self.conv1d(x,self.w1); h=self.conv1d(h,self.w2)
        h=h.flatten()[:self.fc.shape[1]]; return self.fc[:,:len(h)]@h[:self.fc.shape[1]]
 
model=TinyConvNet()
data=np.random.randn(50,3)
out=model.forward(data.T)
pred=ACTIVITIES[out.argmax()]
print(f"Predicted activity: {pred}")
print(f"Model params (approx): {32*3*5+64*32*3+len(ACTIVITIES)*640} weights")
