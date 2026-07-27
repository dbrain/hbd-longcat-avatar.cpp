#!/usr/bin/env python3
"""Import an official-format ModelOpt NVFP4 LTX-2.3 DiT safetensors -> a ggml block_nvfp4 gguf
that longcat-avatar.cpp's LTXAV loader reads. Preserves official's EXACT e2m1 codes AND its
well-conditioned per-16 e4m3 block scales (UNFOLDED: per-16 ggml d = e4m3(weight_scale), the
per-tensor weight global weight_scale_2 is carried separately as a "<weight>.wglobal" sibling
tensor and folded into the cuBLASLt GEMM alpha — folding it into the block scale underflows
~85% of blocks into e4m3 subnormals = the patchy-colour bug).
Non-fp4 tensors (modulation/norm/bias/embed)
are written F32 (what the model allocates). Clones the SRC gguf's tensor list/dims/KV verbatim;
maps ggml bare names <- official 'model.diffusion_model.<name>'. All 4444 SRC tensors map (verified).

Usage: import_ltx_nvfp4.py <src_ggml_gguf> <official_nvfp4_st> <out_gguf>
Adapted from the Flux2 FP4 importer.
"""
import sys,json,struct,numpy as np,os
SRC,OFF,OUT=sys.argv[1],sys.argv[2],sys.argv[3]
PFX='model.diffusion_model.'
GT_F32=0; GT_BF16=30; GT_NVFP4=40
# Non-fp4 tensors: BF16 keeps the gguf small enough to fit the 5060 Ti resident+offload budget
# (F32 bloated it to 20GB -> OOM). The loader converts BF16->F32 for the F32-allocated params.
NONFP4_TYPE=GT_BF16  # trace mode: BF16 non-fp4 (keepf32 list keeps modulation/norm F32) -> triggers concat assert to pin offenders
                    # ggml_concat asserts uniform type, so BF16 breaks it). Bigger gguf, fit via offload.

def st_open(p):
    f=open(p,'rb');n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));return f,8+n,h
def st_raw(f,base,h,key):
    m=h[key];a,b=m['data_offsets'];f.seek(base+a);return np.frombuffer(f.read(b-a),dtype=np.uint8),m['dtype'],m['shape']
def e4m3_dec_byte(u8):
    u=u8.astype(np.int32);e=(u>>3)&0xF;m=u&0x7
    v=np.where(e==0,np.ldexp(m.astype(np.float64),-9),np.ldexp(1.0+m/8.0,e-7))
    return np.where((u&0x7F)==0,0.0,v)
_pb=np.arange(128,dtype=np.uint8); _pv=e4m3_dec_byte(_pb)
_order=np.argsort(_pv); _sv=_pv[_order]; _sb=_pb[_order]
def e4m3_enc_pos(x):
    x=np.minimum(np.asarray(x,dtype=np.float64),448.0)
    idx=np.searchsorted(_sv,x);idx=np.clip(idx,0,len(_sv)-1);lo=np.clip(idx-1,0,len(_sv)-1)
    pick=np.where(np.abs(_sv[idx]-x)<np.abs(_sv[lo]-x),idx,lo);return _sb[pick].astype(np.uint8)
def bf16_to_f32(u8): return (u8.view(np.uint16).astype(np.uint32)<<16).view(np.float32).astype(np.float32)
def f32_to_bf16(x):
    u=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32); r=((u>>16)&1)+0x7FFF; return ((u+r)>>16).astype(np.uint16)
def swz_index(out,nb):
    R,RC,CC=128,32,4;rows=np.arange(out)[:,None];cols=np.arange(nb)[None,:]
    rb=rows//R;rem=rows%R;d4=rem//RC;d3=rem%RC;cbg=cols//CC;d5=cols%CC;cbg_cnt=(nb+CC-1)//CC
    return (((rb*cbg_cnt+cbg)*RC+d3)*16+d4*CC+d5)

fo,bo,ho=st_open(OFF)

def read_wglobal(oname):
    # ModelOpt per-tensor weight global (weight_scale_2). UNFOLDED import keeps this OUT of
    # the per-block ue4m3 scale (folding it underflows ~85% of blocks into e4m3 subnormals ->
    # ~11.5% per-block error = the patchy colour). Carried as a sibling .wglobal tensor; the
    # FP4 cuBLASLt GEMM folds it into alpha (alpha = A_global * W_global).
    return float(st_raw(fo,bo,ho,oname.replace('.weight','.weight_scale_2'))[0].view(np.float32)[0])

def build_nvfp4(oname,dims):
    wu,_,wsh=st_raw(fo,bo,ho,oname); su,_,_=st_raw(fo,bo,ho,oname.replace('.weight','.weight_scale'))
    out,inn=wsh[0],wsh[1]*2; nb16=inn//16
    W=wu.reshape(out,inn//2); code=np.empty((out,inn),dtype=np.uint8)
    code[:,0::2]=W>>4; code[:,1::2]=W&0xF
    # UNFOLDED: store official's well-conditioned per-block e4m3 scale verbatim (no *g).
    sc=e4m3_dec_byte(su).ravel()[swz_index(out,nb16)]
    d=e4m3_enc_pos(sc); nblk=inn//64
    oc=code.reshape(out,nblk,64); qs=np.empty((out,nblk,32),dtype=np.uint8)
    for s in range(4):
        sub=oc[:,:,s*16:s*16+16]; qs[:,:,s*8:s*8+8]=sub[:,:,0:8]|(sub[:,:,8:16]<<4)
    db=d.reshape(out,nblk,4).astype(np.uint8)
    blk=np.concatenate([db,qs],axis=2)
    assert dims[0]==inn and dims[1]==out, f"{oname} dims {dims} vs in{inn} out{out}"
    return blk.tobytes()

def official_nonfp4(oname,ot):
    u,dt,_=st_raw(fo,bo,ho,oname)
    if dt=='BF16': x=bf16_to_f32(u)
    elif dt=='F32': x=u.view(np.float32)
    else: raise Exception(f"{oname} unexpected non-fp4 dtype {dt}")
    return f32_to_bf16(x).tobytes() if ot==GT_BF16 else np.ascontiguousarray(x,dtype=np.float32).tobytes()

# ---- parse SRC gguf (KV verbatim, KV=0 expected) ----
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
    infos.append([name,dims])

def nbytes_of(tt,dims):
    ne=int(np.prod(dims))
    if tt==GT_F32:return ne*4
    if tt==GT_BF16:return ne*2
    if tt==GT_NVFP4:nb=dims[0]//64;ne1=ne//dims[0];return ne1*nb*36
    raise Exception(tt)

plan=[];nfp4=0;nf32=0
wglobals={}   # bare weight name -> g (carried separately, folded into the GEMM alpha)
for name,dims in infos:
    oname=PFX+name
    assert oname in ho, f"missing official {oname}"
    if ho[oname]['dtype']=='U8':
        plan.append((name,oname,dims,GT_NVFP4));nfp4+=1
        wglobals[name]=read_wglobal(oname)
    else:
        # Small / graph-concatenated params (modulation, norm, embedders, biases, scales) MUST
        # stay F32 — the model allocates them F32 and ggml_concat asserts uniform type. Big
        # non-fp4 Linears (the ones official left BF16) go BF16 to fit VRAM.
        keepf32 = any(k in name for k in ('scale_shift','adaln','norm','embedder','.bias','.scale'))
        plan.append((name,oname,dims, GT_F32 if keepf32 else NONFP4_TYPE)); nf32+=1
# Sibling per-tensor weight globals: one 1-element F32 "<weight>.wglobal" per nvfp4 tensor.
# The loader reads these (unknown tensors are tolerated) and registers them so the FP4 GEMM
# folds W's global into alpha. nt must grow to include them.
for wname in wglobals:
    plan.append((wname+'.wglobal', None, [1], GT_F32))
nt = nt + len(wglobals)
print(f"plan: {nfp4} nvfp4 + {nf32} nonfp4 + {len(wglobals)} wglobal = {len(plan)} tensors")

o=open(OUT,'wb')
o.write(magic);o.write(struct.pack('<I',ver));o.write(struct.pack('<Q',nt));o.write(struct.pack('<Q',nkv));o.write(KV)
off=0
for name,oname,dims,ot in plan:
    nbk=name.encode();o.write(struct.pack('<Q',len(nbk)));o.write(nbk);o.write(struct.pack('<I',len(dims)))
    for d in dims:o.write(struct.pack('<Q',d))
    nb=nbytes_of(ot,dims);o.write(struct.pack('<I',ot));o.write(struct.pack('<Q',off));off+=nb+((-nb)%align)
cur=o.tell();o.write(b'\x00'*((-cur)%align))
for name,oname,dims,ot in plan:
    if name.endswith('.wglobal'):
        data=struct.pack('<f', wglobals[name[:-len('.wglobal')]])
    elif ot==GT_NVFP4:
        data=build_nvfp4(oname,dims)
    else:
        data=official_nonfp4(oname,ot)
    nb=nbytes_of(ot,dims);assert len(data)==nb,f"{name} {len(data)} vs {nb}"
    o.write(data);o.write(b'\x00'*((-len(data))%align))
o.close()
print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes)")
