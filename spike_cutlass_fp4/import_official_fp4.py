#!/usr/bin/env python3
"""Import official ModelOpt NVFP4 -> a ggml block_nvfp4 gguf, preserving official's
EXACT e2m1 codes. Single-level fold: per-16 ggml d = e4m3(weight_scale*global).
Official BF16 layers -> bf16. Clones src gguf KV metadata verbatim.

Usage: import_official_fp4.py <src_gguf> <official_st> <out_gguf>
"""
import sys,json,struct,numpy as np,os
SRC,OFF,OUT=sys.argv[1],sys.argv[2],sys.argv[3]
GT_BF16=30; GT_NVFP4=40

def st_open(p):
    f=open(p,'rb');n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));return f,8+n,h
def st_raw(f,base,h,key):
    m=h[key];a,b=m['data_offsets'];f.seek(base+a);return np.frombuffer(f.read(b-a),dtype=np.uint8),m['dtype'],m['shape']
def e4m3_dec_byte(u8):
    u=u8.astype(np.int32);e=(u>>3)&0xF;m=u&0x7
    v=np.where(e==0,np.ldexp(m.astype(np.float64),-9),np.ldexp(1.0+m/8.0,e-7))
    return np.where((u&0x7F)==0,0.0,v)  # magnitude only (sign ignored)
# e4m3 encode via LUT over positive bytes 0..127 (RN nearest)
_pb=np.arange(128,dtype=np.uint8); _pv=e4m3_dec_byte(_pb)
_order=np.argsort(_pv); _sv=_pv[_order]; _sb=_pb[_order]
def e4m3_enc_pos(x):  # x>=0 -> byte (exp/mant only, sign=0)
    x=np.minimum(np.asarray(x,dtype=np.float64),448.0)
    idx=np.searchsorted(_sv,x)
    idx=np.clip(idx,0,len(_sv)-1)
    lo=np.clip(idx-1,0,len(_sv)-1)
    pick=np.where(np.abs(_sv[idx]-x)<np.abs(_sv[lo]-x),idx,lo)
    return _sb[pick].astype(np.uint8)
def bf16_to_f32(u8): return (u8.view(np.uint16).astype(np.uint32)<<16).view(np.float32).astype(np.float32)
def f32_to_bf16(x):
    u=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32); r=((u>>16)&1)+0x7FFF; return ((u+r)>>16).astype(np.uint16)
def swz_index(out,nb):
    R,RC,CC=128,32,4;rows=np.arange(out)[:,None];cols=np.arange(nb)[None,:]
    rb=rows//R;rem=rows%R;d4=rem//RC;d3=rem%RC;cbg=cols//CC;d5=cols%CC;cbg_cnt=(nb+CC-1)//CC
    return (((rb*cbg_cnt+cbg)*RC+d3)*16+d4*CC+d5)

fo,bo,ho=st_open(OFF)

def build_nvfp4(name,dims):
    """Return block_nvfp4 bytes [ne1, nblk, 36] for official weight `name`, matching ggml dims [ne0=in, ne1=out]."""
    wu,_,wsh=st_raw(fo,bo,ho,name); su,_,_=st_raw(fo,bo,ho,name.replace('.weight','.weight_scale'))
    g=float(st_raw(fo,bo,ho,name.replace('.weight','.weight_scale_2'))[0].view(np.float32)[0])
    out,inn=wsh[0],wsh[1]*2; nb16=inn//16
    # official codes (hi nibble = even elem), keep raw 4-bit code (sign+mag)
    W=wu.reshape(out,inn//2); code=np.empty((out,inn),dtype=np.uint8)
    code[:,0::2]=W>>4; code[:,1::2]=W&0xF
    # per-16 scale_full, de-swizzled -> e4m3 encode
    sc=e4m3_dec_byte(su).ravel()[swz_index(out,nb16)]*g   # (out,nb16)
    d=e4m3_enc_pos(sc)                                     # (out,nb16) uint8
    nblk=inn//64
    oc=code.reshape(out,nblk,64)
    qs=np.empty((out,nblk,32),dtype=np.uint8)
    for s in range(4):
        sub=oc[:,:,s*16:s*16+16]
        qs[:,:,s*8:s*8+8]=sub[:,:,0:8]|(sub[:,:,8:16]<<4)
    db=d.reshape(out,nblk,4).astype(np.uint8)
    blk=np.concatenate([db,qs],axis=2)   # (out,nblk,36)  ne1=out, nblk along in
    # ggml expects dims [ne0=in, ne1=out]; data laid [ne1][nblk][36]
    assert dims[0]==inn and dims[1]==out, f"{name} dims {dims} vs in{inn} out{out}"
    return blk.tobytes()

def official_data(name,dims):
    if name not in ho: return None
    dt=ho[name]['dtype']
    if dt=='U8':   return (GT_NVFP4, build_nvfp4(name,dims))
    if dt=='BF16':
        u,_,_=st_raw(fo,bo,ho,name); return (GT_BF16, f32_to_bf16(bf16_to_f32(u)).tobytes())
    if dt=='F32':
        u,_,_=st_raw(fo,bo,ho,name); return (GT_BF16, f32_to_bf16(u.view(np.float32)).tobytes())
    return None

# ---- parse SRC, keep KV verbatim ----
f=open(SRC,'rb');magic=f.read(4);ver=struct.unpack('<I',f.read(4))[0]
nt=struct.unpack('<Q',f.read(8))[0];nkv=struct.unpack('<Q',f.read(8))[0]
kv_start=f.tell()
def rs():n=struct.unpack('<Q',f.read(8))[0];return f.read(n)
def sv(t):
    if t in(0,1,7):f.read(1)
    elif t in(2,3):f.read(2)
    elif t in(4,5,6):f.read(4)
    elif t in(10,11,12):f.read(8)
    elif t==8:rs()
    elif t==9:
        et=struct.unpack('<I',f.read(4))[0];nn=struct.unpack('<Q',f.read(8))[0];[(rs() if et==8 else sv(et)) for _ in range(nn)]
align=32
for _ in range(nkv):
    kl=struct.unpack('<Q',f.read(8))[0];k=f.read(kl);t=struct.unpack('<I',f.read(4))[0]
    if k==b'general.alignment':align=struct.unpack('<I',f.read(4))[0]
    else: sv(t)
kv_end=f.tell();f.seek(kv_start);KV=f.read(kv_end-kv_start)
infos=[]
for _ in range(nt):
    kl=struct.unpack('<Q',f.read(8))[0];name=f.read(kl).decode();nd=struct.unpack('<I',f.read(4))[0]
    dims=[struct.unpack('<Q',f.read(8))[0] for _ in range(nd)];tt=struct.unpack('<I',f.read(4))[0];off=struct.unpack('<Q',f.read(8))[0]
    infos.append([name,dims,tt,off])
pos=f.tell();data_start=(pos+align-1)//align*align
def src_data(off,tt,dims):
    ne=int(np.prod(dims));f.seek(data_start+off)
    if tt in(GT_BF16,1):return f.read(ne*2)
    if tt==0:return f.read(ne*4)
    if tt==GT_NVFP4:nb=dims[0]//64;ne1=ne//dims[0];return f.read(ne1*nb*36)
    raise Exception(f"src type {tt}")
def nbytes_of(tt,dims):
    ne=int(np.prod(dims))
    if tt in(GT_BF16,1):return ne*2
    if tt==0:return ne*4
    if tt==GT_NVFP4:nb=dims[0]//64;ne1=ne//dims[0];return ne1*nb*36
    raise Exception(tt)

# plan (decide type without holding data)
plan=[];nrepl=0
for name,dims,tt,off in infos:
    use=(name in ho) and (name.endswith('.weight') or '.scale' in name)
    if use:
        ot=GT_NVFP4 if ho[name]['dtype']=='U8' else GT_BF16
        plan.append((name,dims,ot,nbytes_of(ot,dims),True,off,tt));nrepl+=1
    else:
        plan.append((name,dims,tt,nbytes_of(tt,dims),False,off,tt))
print(f"replacing {nrepl}/{len(infos)} with official")

o=open(OUT,'wb')
o.write(magic);o.write(struct.pack('<I',ver));o.write(struct.pack('<Q',nt));o.write(struct.pack('<Q',nkv));o.write(KV)
off=0
for name,dims,ot,nb,iso,soff,stt in plan:
    nbk=name.encode();o.write(struct.pack('<Q',len(nbk)));o.write(nbk);o.write(struct.pack('<I',len(dims)))
    for d in dims:o.write(struct.pack('<Q',d))
    o.write(struct.pack('<I',ot));o.write(struct.pack('<Q',off));off+=nb+((-nb)%align)
cur=o.tell();o.write(b'\x00'*((-cur)%align))
for name,dims,ot,nb,iso,soff,stt in plan:
    if iso:
        _,data=official_data(name,dims)
    else:
        data=src_data(soff,stt,dims)
    assert len(data)==nb,f"{name} {len(data)} vs {nb}"
    o.write(data);o.write(b'\x00'*((-len(data))%align))
o.close()
print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes)")
