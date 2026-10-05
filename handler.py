"""Face-Swap-Worker (RunPod Serverless, FaceFusion 3.9.1, 05.10.2026).
Input:  video_url, face_urls (Liste fester Gesichtsfotos des Models, 1-8), upload_url (Studio nimmt das Ergebnis per POST),
        optional swapper_model (Standard hyperswap_1c_256), pixel_boost (256/512/768/1024, Standard 512),
        enhancer_blend (0-100, Standard 35; 0 = kein Nachschaerfen), selector (one|many, Standard one)
Output: {"ok": true, "duration": s, "size_kb": n, "processing_time": s, "uploaded": true}  oder {"error": "...", "stderr": "..."}
FaceFusion-Inhaltsfilter bleibt unveraendert aktiv (Lizenz + Hash-Pruefung im Programm)."""
import runpod, os, subprocess, requests, uuid, time, threading

FF = "/opt/facefusion"; CACHE = "/tmp/fs"; os.makedirs(CACHE, exist_ok=True)
SWAPPERS = {"hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256"}

def dl(url, ext):
    p = f"{CACHE}/{uuid.uuid4().hex[:8]}.{ext}"
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(p, "wb") as f:
            for c in r.iter_content(1 << 20): f.write(c)
    return p

def dur(p):
    try: return float(subprocess.check_output(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", p]).decode().strip())
    except Exception: return 0.0

def handler(event):
    inp = event.get("input", {}) or {}; t0 = time.time(); jid = event.get("id", "?")
    try:
        faces = [u for u in (inp.get("face_urls") or []) if u][:8]
        if not inp.get("video_url") or not faces: return {"error": "video_url und face_urls nötig"}
        model = inp.get("swapper_model") if inp.get("swapper_model") in SWAPPERS else "hyperswap_1c_256"
        boost = str(inp.get("pixel_boost") or 512)
        boost = boost if boost in ("256", "512", "768", "1024") else "512"
        blend = max(0, min(100, int(inp.get("enhancer_blend", 35))))
        selector = "many" if inp.get("selector") == "many" else "one"
        video = dl(inp["video_url"], "mp4")
        srcs = [dl(u, "jpg") for u in faces]
        out = f"{CACHE}/out_{uuid.uuid4().hex[:8]}.mp4"
        procs = ["face_swapper"] + (["face_enhancer"] if blend > 0 else [])
        cmd = ["python3", "facefusion.py", "headless-run", "-s", *srcs, "-t", video, "-o", out,
               "--processors", *procs, "--face-swapper-model", model, "--face-swapper-pixel-boost", f"{boost}x{boost}",
               "--face-selector-mode", selector, "--face-selector-order", "large-small",
               "--face-mask-types", "box", "occlusion", "--face-mask-blur", "0.3",
               "--execution-providers", "cuda", "--execution-thread-count", "16", "--video-memory-strategy", "tolerant",
               "--output-video-encoder", "libx264", "--output-video-quality", "92", "--output-video-preset", "medium", "--log-level", "info"]
        if blend > 0: cmd += ["--face-enhancer-model", "gfpgan_1.4", "--face-enhancer-blend", str(blend)]
        stop = threading.Event()
        def hb():
            t = 0
            while not stop.wait(30):
                t += 30; print(f"[{jid}] läuft {t}s", flush=True)
        threading.Thread(target=hb, daemon=True).start()
        try: r = subprocess.run(cmd, capture_output=True, text=True, timeout=3000, cwd=FF)
        finally: stop.set()
        log = ((r.stdout or "") + (r.stderr or ""))[-1500:]
        if r.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) < 1000:
            # 24.09.-Lehre: RunPod behaelt bei Fehlern nur den error-Text -> Ursache mit hinein
            return {"error": "FaceFusion fehlgeschlagen: " + log[-300:], "stderr": log}
        res = {"ok": True, "duration": dur(out), "size_kb": os.path.getsize(out) // 1024, "processing_time": round(time.time() - t0, 1), "uploaded": False, "log": log[-400:]}
        if inp.get("upload_url"):
            with open(out, "rb") as f:
                u = requests.post(inp["upload_url"], files={"file": ("output_swap.mp4", f, "video/mp4")}, timeout=900)
            res["uploaded"] = u.ok; res["upload_status"] = u.status_code
        return res
    except Exception as e:
        return {"error": str(e)[:500]}
    finally:
        for n in os.listdir(CACHE):
            try: os.remove(os.path.join(CACHE, n))
            except Exception: pass

runpod.serverless.start({"handler": handler})
