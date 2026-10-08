"""Face-Swap-Worker (RunPod Serverless, FaceFusion 3.9.1, 05.10.2026, erweitert 08.10.2026).
Input:  video_url, face_urls (Liste fester Gesichtsfotos des Models, 1-8), upload_url (Studio nimmt das Ergebnis per POST),
        optional swapper_model (Standard hyperswap_1c_256), pixel_boost (256/512/768/1024, Standard 512),
        enhancer_model (Standard gfpgan_1.4; codeformer, gpen_bfr_512/1024/2048, restoreformer_plus_plus),
        enhancer_blend (0-100, Standard 35; 0 = kein Nachschaerfen), enhancer_weight (0-1, nur codeformer),
        mask_types (Liste aus box/occlusion/region, Standard box+occlusion), mask_regions (Liste, nur mit region),
        mask_blur (0-1, Standard 0.3), mask_padding "oben rechts unten links" in % (Standard "0 0 0 0"),
        expression (0-100, Standard 0 = aus; Mimik des Originals per live_portrait zurueckholen),
        trim_start / trim_end (Sekunden, nur diesen Ausschnitt rechnen – fuer Tests),
        selector (one|many, Standard one)
Output: {"ok": true, "duration": s, "size_kb": n, "processing_time": s, "uploaded": true}  oder {"error": "...", "stderr": "..."}
FaceFusion-Inhaltsfilter bleibt unveraendert aktiv (Lizenz + Hash-Pruefung im Programm)."""
import runpod, os, subprocess, requests, uuid, time, threading

FF = "/opt/facefusion"; CACHE = "/tmp/fs"; os.makedirs(CACHE, exist_ok=True)
SWAPPERS = {"hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256", "inswapper_128", "inswapper_128_fp16",
            "simswap_256", "simswap_unofficial_512", "ghost_3_256", "uniface_256", "blendswap_256", "hififace_unofficial_256"}
ENHANCERS = {"codeformer", "gfpgan_1.2", "gfpgan_1.3", "gfpgan_1.4", "gpen_bfr_256", "gpen_bfr_512", "gpen_bfr_1024", "gpen_bfr_2048", "restoreformer_plus_plus"}
MASK_TYPES = {"box", "occlusion", "region"}
MASK_REGIONS = {"skin", "left-eyebrow", "right-eyebrow", "left-eye", "right-eye", "glasses", "nose", "mouth", "upper-lip", "lower-lip"}

def dl(url, ext):
    p = f"{CACHE}/{uuid.uuid4().hex[:8]}.{ext}"
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(p, "wb") as f:
            for c in r.iter_content(1 << 20): f.write(c)
    return p

def probe(p, entries):
    try: return subprocess.check_output(["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-show_entries", entries, "-of", "csv=p=0", p]).decode().strip()
    except Exception: return ""

def dur(p):
    try: return float(subprocess.check_output(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", p]).decode().strip())
    except Exception: return 0.0

def fps(p):
    r = probe(p, "stream=r_frame_rate")
    try: a, b = r.split("/"); return float(a) / float(b)
    except Exception: return 30.0

def pick(v, allowed, default):
    return v if v in allowed else default

def handler(event):
    inp = event.get("input", {}) or {}; t0 = time.time(); jid = event.get("id", "?")
    try:
        faces = [u for u in (inp.get("face_urls") or []) if u][:8]
        if not inp.get("video_url") or not faces: return {"error": "video_url und face_urls nötig"}
        model = pick(inp.get("swapper_model"), SWAPPERS, "hyperswap_1c_256")
        boost = str(inp.get("pixel_boost") or 512)
        boost = boost if boost in ("256", "512", "768", "1024") else "512"
        blend = max(0, min(100, int(inp.get("enhancer_blend", 35))))
        enhancer = pick(inp.get("enhancer_model"), ENHANCERS, "gfpgan_1.4")
        weight = max(0.0, min(1.0, float(inp.get("enhancer_weight", 0.5))))
        mtypes = [m for m in (inp.get("mask_types") or ["box", "occlusion"]) if m in MASK_TYPES] or ["box", "occlusion"]
        mregions = [m for m in (inp.get("mask_regions") or []) if m in MASK_REGIONS]
        mblur = max(0.0, min(1.0, float(inp.get("mask_blur", 0.3))))
        mpad = [str(max(0, min(100, int(x)))) for x in str(inp.get("mask_padding") or "0 0 0 0").split()[:4]]
        if len(mpad) != 4: mpad = ["0", "0", "0", "0"]
        expression = max(0, min(100, int(inp.get("expression", 0) or 0)))
        selector = "many" if inp.get("selector") == "many" else "one"
        video = dl(inp["video_url"], "mp4")
        srcs = [dl(u, "jpg") for u in faces]
        out = f"{CACHE}/out_{uuid.uuid4().hex[:8]}.mp4"
        procs = ["face_swapper"]
        if expression > 0: procs.append("expression_restorer")
        if blend > 0: procs.append("face_enhancer")
        cmd = ["python3", "facefusion.py", "headless-run", "-s", *srcs, "-t", video, "-o", out,
               "--processors", *procs, "--face-swapper-model", model, "--face-swapper-pixel-boost", f"{boost}x{boost}",
               "--face-selector-mode", selector, "--face-selector-order", "large-small",
               "--face-mask-types", *mtypes, "--face-mask-blur", f"{mblur:.2f}", "--face-mask-padding", *mpad,
               "--execution-providers", "cuda", "--execution-thread-count", "16", "--video-memory-strategy", "tolerant",
               "--output-video-encoder", "libx264", "--output-video-quality", "92", "--output-video-preset", "medium", "--log-level", "info"]
        if "region" in mtypes and mregions: cmd += ["--face-mask-regions", *mregions]
        if blend > 0:
            cmd += ["--face-enhancer-model", enhancer, "--face-enhancer-blend", str(blend)]
            if enhancer == "codeformer": cmd += ["--face-enhancer-weight", f"{weight:.2f}"]
        if expression > 0: cmd += ["--expression-restorer-model", "live_portrait", "--expression-restorer-factor", str(expression)]
        # Testausschnitt: nur Sekunden trim_start..trim_end rechnen (FaceFusion zaehlt in Frames)
        if inp.get("trim_start") is not None or inp.get("trim_end") is not None:
            f = fps(video)
            if inp.get("trim_start") is not None: cmd += ["--trim-frame-start", str(max(0, int(float(inp["trim_start"]) * f)))]
            if inp.get("trim_end") is not None: cmd += ["--trim-frame-end", str(max(1, int(float(inp["trim_end"]) * f)))]
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
            return {"error": "FaceFusion fehlgeschlagen: " + log[-300:], "stderr": log, "cmd": " ".join(cmd[3:])}
        res = {"ok": True, "duration": dur(out), "size_kb": os.path.getsize(out) // 1024, "processing_time": round(time.time() - t0, 1), "uploaded": False,
               "settings": {"swapper": model, "boost": boost, "enhancer": enhancer if blend > 0 else None, "blend": blend, "mask": mtypes, "regions": mregions, "expression": expression}, "log": log[-400:]}
        if inp.get("upload_url"):
            with open(out, "rb") as f:
                u = requests.post(inp["upload_url"], files={"file": (inp.get("upload_name") or "output_swap.mp4", f, "video/mp4")}, timeout=900)
            res["uploaded"] = u.ok; res["upload_status"] = u.status_code
        return res
    except Exception as e:
        return {"error": str(e)[:500]}
    finally:
        for n in os.listdir(CACHE):
            try: os.remove(os.path.join(CACHE, n))
            except Exception: pass

runpod.serverless.start({"handler": handler})
