#!/usr/bin/env python3
"""env_CV speech/noise discriminator for a rendered NAVA clip.
CV = std/mean of per-800-sample RMS envelope. >0.5 => speech (syllable structure),
<0.5 => flat noise. Also reports speechband% (200-3400 Hz energy fraction)."""
import sys, subprocess, numpy as np, os, tempfile, wave

def load_wav_16k_mono(path):
    if path.endswith(".wav"):
        wav = path
    else:
        wav = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg","-y","-i",path,"-ar","16000","-ac","1",wav],
                       check=True, capture_output=True)
    with wave.open(wav,"rb") as w:
        n = w.getnframes(); raw = w.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64)/32768.0
    return x

for path in sys.argv[1:]:
    if not os.path.exists(path):
        print(f"{os.path.basename(path):40s} MISSING"); continue
    try:
        x = load_wav_16k_mono(path)
    except Exception as e:
        print(f"{os.path.basename(path):40s} ERR {e}"); continue
    if x.size < 800:
        print(f"{os.path.basename(path):40s} too short ({x.size})"); continue
    # per-800-sample RMS envelope
    nblk = x.size//800
    blk = x[:nblk*800].reshape(nblk,800)
    rms = np.sqrt((blk**2).mean(axis=1)) + 1e-12
    cv = rms.std()/rms.mean()
    # speechband fraction
    X = np.abs(np.fft.rfft(x)); freqs = np.fft.rfftfreq(x.size, 1/16000)
    band = ((freqs>=200)&(freqs<=3400))
    sb = (X[band]**2).sum()/((X**2).sum()+1e-30)
    verdict = "SPEECH" if cv>0.5 else "noise "
    print(f"{os.path.basename(path):40s} env_CV={cv:.3f} [{verdict}]  speechband%={sb*100:5.1f}  dur={x.size/16000:.2f}s")
