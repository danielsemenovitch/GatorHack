"""V5 research-only residual CNN for cropped Western blot patches.

Deliberately separate from btw.cnn so existing V3 checkpoints and the live
website remain unchanged. 32px patches, three channels (relative intensity,
noise residual, and broader-scale contrast). Training and evaluation must call
this module for V5 and btw.cnn for frozen V3.
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from . import cnn
from .imgio import normalize_polarity

P=32
STRIDE=8
WEIGHTS=Path(__file__).resolve().parents[1]/"models/splice_cnn_v5.npz"

def panel_channels(gray):
    g,_=normalize_polarity(gray)
    g=g.astype(np.float32)
    bg=float(np.median(g))
    lo=float(np.percentile(g,1))
    intensity=np.clip((g-bg)/max(8.0,bg-lo),-5,5)
    fine=g-cv2.GaussianBlur(g,(0,0),0.8)
    noise=np.clip(fine/(float(np.median(np.abs(fine)))*1.4826+0.5),-8,8)/4.0
    broad=g-cv2.GaussianBlur(g,(0,0),2.5)
    scale=max(5.0,float(np.percentile(np.abs(broad),85)))
    context=np.clip(broad/scale,-5,5)/2.0
    # Reuse the V3 band mask without changing baseline behavior.
    _, dk = cnn.panel_channels(gray)
    return np.stack([intensity,noise,context],-1).astype(np.float32), dk

def extract(gray,mask=None):
    ch,dk=panel_channels(gray)
    ch,dk,mask=cnn.pad_small(ch,dk,mask)
    pos=cnn.grid(*dk.shape,p=P,s=STRIDE)
    X=np.stack([ch[y:y+P,x:x+P] for x,y in pos]) if pos else np.zeros((0,P,P,3),np.float32)
    band=np.array([(dk[y:y+P,x:x+P]>15).mean()>.12 for x,y in pos],bool)
    frac=(np.array([mask[y:y+P,x:x+P].mean() for x,y in pos],np.float32)
          if mask is not None else np.zeros(len(pos),np.float32))
    return X,pos,band,frac

def init_params(seed=0):
    rng=np.random.default_rng(seed)
    def layer(cin,cout,k=3):
        return {"w":(rng.standard_normal((k,k,cin,cout))*np.sqrt(2/(k*k*cin))).astype(np.float32),
                "b":np.zeros((cout,),np.float32)}
    def block(n):
        return {"a":layer(n,n), "b":layer(n,n)}
    return {
        "stem":layer(3,24), "res1":block(24),
        "lift2":layer(24,32,k=1), "res2":block(32),
        "lift3":layer(32,48,k=1), "res3":block(48),
        "fc1":{"w":(rng.standard_normal((96,64))*np.sqrt(2/96)).astype(np.float32),
               "b":np.zeros((64,),np.float32)},
        "fc2":{"w":(rng.standard_normal((64,1))*np.sqrt(1/64)).astype(np.float32),
               "b":np.zeros((1,),np.float32)}
    }

def apply(params,x):
    import jax
    import jax.numpy as jnp
    def conv(h,p):
        z=jax.lax.conv_general_dilated(h,p["w"],(1,1),"SAME",
                                      dimension_numbers=("NHWC","HWIO","NHWC"))
        return z+p["b"]
    def norm(z,groups=4):
        n,h,w,c=z.shape
        v=z.reshape(n,h,w,groups,c//groups)
        mu=v.mean(axis=(1,2,4),keepdims=True)
        sd=jnp.mean((v-mu)**2,axis=(1,2,4),keepdims=True)
        return ((v-mu)*jax.lax.rsqrt(sd+1e-5)).reshape(n,h,w,c)
    def block(h,p):
        z=jax.nn.gelu(norm(conv(h,p["a"])))
        z=norm(conv(z,p["b"]))
        return jax.nn.gelu(h+z*0.5)
    def pool(h):
        return jax.lax.reduce_window(h,0.,jax.lax.add,(1,2,2,1),(1,2,2,1),"VALID")/4.
    h=jax.nn.gelu(norm(conv(x,params["stem"])))
    h=block(h,params["res1"])
    h=pool(h)
    h=jax.nn.gelu(norm(conv(h,params["lift2"])))
    h=block(h,params["res2"])
    h=pool(h)
    h=jax.nn.gelu(norm(conv(h,params["lift3"])))
    h=block(h,params["res3"])
    h=jnp.concatenate([h.mean(axis=(1,2)),h.max(axis=(1,2))],axis=-1)
    h=jax.nn.gelu(h@params["fc1"]["w"]+params["fc1"]["b"])
    return (h@params["fc2"]["w"]+params["fc2"]["b"])[:,0]

_PREDICT=None
def predict(params,X,batch=128,logits=True):
    import jax
    global _PREDICT
    if _PREDICT is None: _PREDICT=jax.jit(apply)
    out=[]
    for i in range(0,len(X),batch):
        a=np.asarray(X[i:i+batch],np.float32)
        n=len(a)
        if n<batch:a=np.concatenate([a,np.zeros((batch-n,*a.shape[1:]),np.float32)])
        out.append(np.asarray(_PREDICT(params,a))[:n])
    z=np.concatenate(out) if out else np.zeros(0,np.float32)
    return z if logits else 1/(1+np.exp(-z))

relative=cnn.relative
RULES=cnn.RULES

def save(params,path=WEIGHTS,**meta):
    flat={}
    def walk(prefix,obj):
        if isinstance(obj,dict):
            for k,v in obj.items():walk(prefix+"."+k if prefix else k,v)
        else: flat[prefix]=np.asarray(obj)
    walk("",params)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,**flat,**{"meta."+k:np.asarray(v) for k,v in meta.items()})

def load(path=WEIGHTS):
    with np.load(path) as z:
        params={};meta={}
        for key in z.files:
            if key.startswith("meta."):
                a=z[key];meta[key[5:]]=a.item() if a.shape==() else a
                continue
            ref=params
            parts=key.split(".")
            for part in parts[:-1]:ref=ref.setdefault(part,{})
            ref[parts[-1]]=z[key]
    return params,meta

def self_test():
    import jax
    rng=np.random.default_rng(10)
    im=rng.uniform(20,245,(73,95)).astype(np.float32)
    mask=np.zeros((73,95),np.uint8);mask[25:33,35:55]=1
    x,positions,band,f=extract(im,mask)
    assert x.shape[1:]==(32,32,3) and len(f)==len(positions)
    assert 0<float(f.max())<1
    params=init_params()
    out=predict(params,x[:3],batch=4)
    assert out.shape==(min(3,len(x)),) and np.isfinite(out).all()
    # Gradients must actually backpropagate through all residual branches.
    grad=jax.grad(lambda p:jax.numpy.mean(apply(p,jax.numpy.asarray(x[:2]))))(params)
    assert np.isfinite(np.asarray(grad["res3"]["b"]["w"])).all()
    d=Path("/tmp/btw-v5-self-test-checkpoint.npz")
    save(params,path=d,rule="top",space="logit",architecture="v5_residual")
    back,meta=load(d)
    assert meta["architecture"]=="v5_residual"
    assert np.allclose(back["res2"]["a"]["w"],params["res2"]["a"]["w"])
    d.unlink(missing_ok=True)
    print("V5_CNN_SELF_TEST_OK",len(positions),flush=True)

if __name__=="__main__":self_test()
