#!/usr/bin/env python3
"""probe_camera.py — M0 camera capability freeze.

Enumerates UVC cameras WITHOUT opening a stream (no format setting, no
capture): reads sysfs + lsusb + v4l2-ctl --list-formats-ext, estimates
per-format bandwidth, and RECOMMENDS the bring-up format (MJPEG preferred
on USB2). Output is a JSON report; never touches driver streaming state.

Usage: python3 scripts/probe_camera.py --out reports/camera_caps.json
"""
import argparse, json, re, subprocess, datetime, os, sys


def run(cmd, timeout=10):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.stdout.strip()
    except Exception:
        return ""


def parse_lsusb():
    out = []
    for line in run("lsusb").splitlines():
        # Bus 001 Device 004: ID 046d:0825 Logitech, Inc. Webcam C270
        m = re.match(r"Bus\s+(\d+)\s+Device\s+(\d+):\s+ID\s+([0-9a-fA-F]{4}):([0-9a-fA-F]{4})\s*(.*)", line)
        if m:
            out.append({"bus": m.group(1), "dev": m.group(2),
                        "vid": m.group(3).lower(), "pid": m.group(4).lower(),
                        "desc": m.group(5).strip()})
    return out


USB2_BPS = 280_000_000  # effective bulk/isoc payload ceiling, conservative
BPP = {"MJPG": 0.35, "MJPEG": 0.35, "YUYV": 2.0, "YUY2": 2.0, "NV12": 1.5, "H264": 0.15}


def estimate_bps(w, h, fps, fourcc):
    bpp = BPP.get(fourcc.upper(), 1.0)
    if fourcc.upper() in ("MJPG", "MJPEG", "H264"):
        # compressed: bits-per-pixel heuristic * pixels * fps
        return int(w * h * fps * bpp)
    return int(w * h * fps * bpp)


def probe_v4l2_formats(dev):
    raw = run(f"v4l2-ctl -d {dev} --list-formats-ext 2>/dev/null")
    fmts = []
    cur_fourcc = None
    for line in raw.splitlines():
        m = re.search(r"\[\d+\]:\s+'(\w+)'", line)
        if m:
            cur_fourcc = m.group(1)
            continue
        m = re.match(r"\s*Size:\s*Discrete\s+(\d+)x(\d+)", line)
        if m and cur_fourcc:
            w, h = int(m.group(1)), int(m.group(2))
            fmts.append({"fourcc": cur_fourcc, "width": w, "height": h})
            continue
        m = re.match(r"\s*Interval:\s*Discrete\s+([\d.]+)s.*\(([\d.]+)\s*fps\)", line)
        if m and fmts:
            fmts[-1].setdefault("intervals_s", []).append(float(m.group(1)))
            fmts[-1].setdefault("fps", []).append(float(m.group(2)))
    # bandwidth per (fmt,res,maxfps)
    for f in fmts:
        fps = max(f.get("fps", [0])) if f.get("fps") else 0
        f["max_fps"] = fps
        f["est_bps"] = estimate_bps(f["width"], f["height"], fps, f["fourcc"])
        f["fits_usb2"] = f["est_bps"] <= USB2_BPS
    return raw, fmts


def recommend(fmts):
    # Prefer MJPEG 720p-class that fits USB2; else highest MJPEG; else anything.
    cands = [f for f in fmts if "MJPG" in f["fourcc"].upper() and f["fits_usb2"]]
    cands.sort(key=lambda f: (-(f["width"]*f["height"]), -(f["max_fps"] or 0)))
    if cands: return {"strategy": "MJPEG-first (USB2 safe)", "pick": cands[0]}
    mj = [f for f in fmts if "MJPG" in f["fourcc"].upper()]
    if mj:
        mj.sort(key=lambda f: (-(f["width"]*f["height"])))
        return {"strategy": "MJPEG-only (over USB2 budget, expect drops)", "pick": mj[0]}
    if fmts:
        return {"strategy": "no-MJPEG fallback (bandwidth risk)", "pick": fmts[0]}
    return {"strategy": "none", "pick": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/camera_caps.json")
    args = ap.parse_args()

    rep = {"collected_at": datetime.datetime.now().astimezone().isoformat(),
           "cameras": [], "recommendation": None, "hardware_pending": False}
    devices = sorted([d for d in os.listdir("/dev") if re.match(r"video\d+$", d)]) \
        if os.path.isdir("/dev") else []
    if not devices and not run("lsusb"):
        rep["hardware_pending"] = True
        rep["note"] = "No /dev/video* and no lsusb output: run ON the board with camera attached."
    usb = parse_lsusb()
    for d in devices:
        dev = f"/dev/{d}"
        raw, fmts = probe_v4l2_formats(dev)
        cap = run(f"v4l2-ctl -d {dev} --all 2>/dev/null | head -n 30")
        rep["cameras"].append({"device": dev, "formats": fmts,
                               "v4l2_all_head": cap,
                               "v4l2_raw_formats": raw[:4000]})
    if rep["cameras"]:
        allf = [f for c in rep["cameras"] for f in c["formats"]]
        rep["recommendation"] = recommend(allf)
    rep["lsusb"] = usb
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"[probe_camera] wrote {args.out} ({len(rep['cameras'])} video nodes)")


if __name__ == "__main__":
    main()
