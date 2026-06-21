#!/usr/bin/env python3
"""Clone a flux2 gguf's structure (KV metadata verbatim) and replace each weight
with the OFFICIAL ModelOpt NVFP4 checkpoint's value, decoded faithfully (cos 0.995):
  E2M1 (first-elem HIGH nibble) * weight_scale(F8E4M3, SWIZZLE_32_4_4) * weight_scale_2(F32).
Output mode 'bf16' writes every weight as BF16 (decisive test: is official's quantization
the 4-step quality lever, independent of FP4 runtime?).

Usage: import_official.py <src_gguf> <official_safetensors> <bf16_base_st> <out_gguf>
"""
import sys,json,struct,numpy as np

SRC,OFF,BASE,OUT = sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
GT_BF16=30

def st_open(p):
    f=open(p,'rb');n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));return f,8+n,h
def st_raw(f,base,h,key):
    m=h[key];a,b=m['data_offsets'];f.seek(base+a);return np.frombuffer(f.read(b-a),dtype=np.uint8),m['dtype'],m['shape']
def e4m3_dec(u8):
    u=u8.astype(np.int32);s=np.where((u>>7)&1,-1.0,1.0);e=(u>>3)&0xF;m=u&0x7
    v=np.where(e==0,np.ldexp(m.astype(np.float64),-9),np.ldexp(1.0+m/8.0,e-7))
    return s*np.where((u&0x7F)==0,0.0,v)
def bf16_to_f32(u8):
    return (u8.view(np.uint16).astype(np.uint32)<<16).view(np.float32).astype(np.float32)
def f32_to_bf16(x):  # round-to-nearest-even truncation
    u=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32)
    r=((u>>16)&1)+0x7FFF; u=(u+r)>>16; return u.astype(np.uint16)
E2M1=np.array([0,0.5,1,1.5,2,3,4,6])
def e2m1_dec(n): return np.where((n>>3)&1,-1.0,1.0)*E2M1[n&0x7]
def swz_index(out,nb):
    R,RC,CC=128,32,4;rows=np.arange(out)[:,None];cols=np.arange(nb)[None,:]
    rb=rows//R;rem=rows%R;d4=rem//RC;d3=rem%RC;cbg=cols//CC;d5=cols%CC;cbg_cnt=(nb+CC-1)//CC
    return (((rb*cbg_cnt+cbg)*RC+d3)*16+d4*CC+d5)

fo,bo,ho=st_open(OFF)
fbf,bbf,hbf=st_open(BASE)

def official_bf16(name):
    """Return f32 array for tensor `name` from official (decoding U8/FP4 faithfully), or None."""
    if name not in ho: return None
    dt=ho[name]['dtype']
    if dt=='BF16':
        u,_,sh=st_raw(fo,bo,ho,name); return bf16_to_f32(u).reshape(sh)
    if dt=='F32':
        u,_,sh=st_raw(fo,bo,ho,name); return u.view(np.float32).reshape(sh).astype(np.float32)
    if dt=='U8':  # FP4 weight
        wu,_,wsh=st_raw(fo,bo,ho,name); su,_,_=st_raw(fo,bo,ho,name.replace('.weight','.weight_scale'))
        g=st_raw(fo,bo,ho,name.replace('.weight','.weight_scale_2'))[0].view(np.float32)[0]
        out,inn=wsh[0],wsh[1]*2; nb=inn//16
        W=wu.reshape(out,inn//2); e2=np.empty((out,inn),dtype=np.float64)
        e2[:,0::2]=e2m1_dec(W>>4); e2[:,1::2]=e2m1_dec(W&0xF)
        sc=e4m3_dec(su).ravel()[swz_index(out,nb)]
        rec=(e2.reshape(out,nb,16)*sc[:,:,None]).reshape(out,inn)*float(g)
        return rec.astype(np.float32)
    return None

# ---- parse SRC gguf: keep KV bytes verbatim, read tensor infos + data ----
f=open(SRC,'rb'); magic=f.read(4); ver=struct.unpack('<I',f.read(4))[0]
nt=struct.unpack('<Q',f.read(8))[0]; nkv=struct.unpack('<Q',f.read(8))[0]
kv_start=f.tell()
def rs():n=struct.unpack('<Q',f.read(8))[0];return f.read(n)
def sv(t):
    if t in(0,1,7):f.read(1)
    elif t in(2,3):f.read(2)
    elif t in(4,5,6):f.read(4)
    elif t in(10,11,12):f.read(8)
    elif t==8:rs()
    elif t==9:
        et=struct.unpack('<I',f.read(4))[0];nn=struct.unpack('<Q',f.read(8))[0]
        [(rs() if et==8 else sv(et)) for _ in range(nn)]
align=32
for _ in range(nkv):
    klen=struct.unpack('<Q',f.read(8))[0];k=f.read(klen);t=struct.unpack('<I',f.read(4))[0]
    if k==b'general.alignment': align=struct.unpack('<I',f.read(4))[0]
    else: sv(t)
kv_end=f.tell()
f.seek(kv_start); KV_BYTES=f.read(kv_end-kv_start)
# tensor infos
infos=[]
for _ in range(nt):
    klen=struct.unpack('<Q',f.read(8))[0];name=f.read(klen).decode()
    nd=struct.unpack('<I',f.read(4))[0];dims=[struct.unpack('<Q',f.read(8))[0] for _ in range(nd)]
    tt=struct.unpack('<I',f.read(4))[0];off=struct.unpack('<Q',f.read(8))[0]
    infos.append([name,dims,tt,off])
pos=f.tell(); data_start=(pos+align-1)//align*align
def read_src_data(off,tt,dims):
    ne=1
    for d in dims: ne*=d
    f.seek(data_start+off)
    if tt==GT_BF16: return f.read(ne*2)
    if tt==0: return f.read(ne*4)
    if tt==1: return f.read(ne*2)
    if tt==40:
        nb=dims[0]//64; ne1=1
        for d in dims[1:]: ne1*=d
        return f.read(ne1*nb*36)
    raise Exception(f"unhandled src type {tt} for {dims}")

# ---- decide each tensor's (type, byte length) WITHOUT holding data ----
plan=[]  # (name, dims, out_type, nbytes, is_official, src_off, src_tt)
def type_nbytes(tt,dims):
    ne=int(np.prod(dims))
    if tt==GT_BF16 or tt==1: return ne*2
    if tt==0: return ne*4
    if tt==40:
        nb=dims[0]//64; ne1=ne//dims[0]; return ne1*nb*36
    raise Exception(f"len? type {tt}")
nrepl=0
for name,dims,tt,off in infos:
    is_off = (name in ho) and (name.endswith('.weight') or '.scale' in name)
    if is_off:
        plan.append((name,dims,GT_BF16,type_nbytes(GT_BF16,dims),True,off,tt)); nrepl+=1
    else:
        plan.append((name,dims,tt,type_nbytes(tt,dims),False,off,tt))
print(f"replacing {nrepl}/{len(infos)} tensors with official (bf16)")

# ---- write OUT gguf (streamed) ----
o=open(OUT,'wb')
o.write(magic); o.write(struct.pack('<I',ver)); o.write(struct.pack('<Q',nt)); o.write(struct.pack('<Q',nkv))
o.write(KV_BYTES)
# info table with offsets
off=0
for name,dims,ot,nbytes,iso,soff,stt in plan:
    nb=name.encode(); o.write(struct.pack('<Q',len(nb))); o.write(nb)
    o.write(struct.pack('<I',len(dims)))
    for d in dims: o.write(struct.pack('<Q',d))
    o.write(struct.pack('<I',ot)); o.write(struct.pack('<Q',off))
    off += nbytes + ((-nbytes)%align)
# pad to alignment before data section
cur=o.tell(); o.write(b'\x00'*((-cur)%align))
# stream data
for i,(name,dims,ot,nbytes,iso,soff,stt) in enumerate(plan):
    if iso:
        arr=official_bf16(name); data=f32_to_bf16(arr.reshape(dims)).tobytes()
    else:
        data=read_src_data(soff,stt,dims)
    assert len(data)==nbytes, f"{name} {len(data)} vs {nbytes}"
    o.write(data); o.write(b'\x00'*((-len(data))%align))
o.close()
import os; print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes)")
