"""曲のビートに合わせて切り替わる MV 風エンドロールを作る。

前半は全画面で拍ごとにカットを切り替え、最後の約1分だけ
「左に動画・右にクレジット」のエンドロールになる。

使い方:
    python3 make_mv.py            # footage/song.mp3, footage/clips.txt, footage/grid.json を使う

grid.json は {"bpm": 148, "t0": 0.388}（t0 = 拍0 の秒数）。拍の番号は t0 から数える。
"""
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import make_endroll as er
from make_endroll import BG, FPS, H, W, VH, VW, VX, VY

HERE = Path(__file__).parent
FOOT = HERE / "footage"
WORK = HERE / "build"
BG_HEX = "0x%02X%02X%02X" % BG

# (開始拍, 終了拍, 何拍ごとに切り替えるか, 画面, 光らせる間隔(拍) or None, ズーム)
# 拍の番号は grid.json の t0 から数える。4で割って1余る拍が小節の頭。
SECTIONS = [
    (1, 33, 8, "full", None, False),      # イントロ
    (33, 97, 4, "full", None, False),     # Aメロ
    (97, 121, 4, "full", None, False),    # Bメロ
    (121, 125, 2, "full", None, False),   # サビ前で加速
    (125, 129, 1, "full", None, False),
    (129, 197, 2, "full", 16, True),      # サビ1
    (197, 257, 4, "full", None, False),   # 2番
    (257, 321, 2, "full", 16, True),      # 盛り上がり
    (321, 357, 8, "panel", None, False),  # 静かなところからエンドロール
    (357, 441, 4, "panel", 84, False),    # ラストサビ
    (441, 445, 2, "panel", None, False),  # 最後のカットの前で加速
    (445, 453, 8, "panel", None, False),  # 最後のカットは長めに
]
END_CARD_BEAT = 453

# 素材をどの区間に割り当てるか（clips.txt のファイル名で区切る）
PANEL_QUIET_FROM = "IMG_5287.mov"   # 静かなパート用の最初の素材
PANEL_CHORUS_FROM = "IMG_5292.mov"  # ラストサビ（満員の客席から）


def beat_time(grid, k):
    return grid["t0"] + k * 60.0 / grid["bpm"]


def frame_at(t):
    return int(round(t * FPS))


def build_slots(grid):
    """各カットの (開始フレーム, フレーム数, 画面, 光らせるか, ズーム) を並べる。"""
    slots = []
    for b0, b1, step, mode, flash_every, zoom in SECTIONS:
        for k in range(b0, b1, step):
            f0 = frame_at(beat_time(grid, k))
            f1 = frame_at(beat_time(grid, min(k + step, b1)))
            flash = flash_every is not None and (k - b0) % flash_every == 0
            slots.append({"f0": f0, "n": f1 - f0, "mode": mode, "flash": flash, "zoom": zoom})
    return slots


def assign(slots, clips):
    """素材をカットに割り当てる。

    写真は1回だけ、動画は長さに応じて何回か（毎回ちがう部分を）使う。
    撮った順番はほぼ保ったまま、同じ素材が続けて出ないように並べる。
    """
    lens = [(e if e is not None else er.probe(f)[2]) - s for f, s, e in clips]
    photos = [er.is_image(f) for f, _, _ in clips]
    n_free = len(slots) - sum(photos)
    vids = [i for i, ph in enumerate(photos) if not ph]
    if n_free < len(vids):
        raise SystemExit(f"カット {len(slots)} 個に対して素材が多すぎます（{len(clips)} 個）")
    # 動画の使用回数：最低1回、残りは長さに比例（最大剰余法）
    counts = [1] * len(clips)
    extra = n_free - len(vids)
    total_len = sum(lens[i] for i in vids)
    raw = {i: extra * lens[i] / total_len for i in vids}
    for i in vids:
        counts[i] += int(raw[i])
    left = extra - sum(int(raw[i]) for i in vids)
    for i in sorted(vids, key=lambda i: raw[i] - int(raw[i]), reverse=True)[:left]:
        counts[i] += 1

    # 撮った順に並べてから、同じ素材が隣り合わないように近くのものと入れ替える
    pending = [(i, j) for i in range(len(clips)) for j in range(counts[i])]
    order, last = [], None
    while pending:
        pick = next((p for p in pending[:6] if p[0] != last), pending[0])
        pending.remove(pick)
        order.append(pick)
        last = pick[0]

    # 動画は使う回数ぶんに区切って、毎回ちがう部分を使う
    for sl, (i, j) in zip(slots, order):
        f, s, _ = clips[i]
        d = sl["n"] / FPS
        win = lens[i] / counts[i]
        start = s + j * win + max(0.0, (win - d) / 2)
        sl["src"], sl["ss"] = f, max(s, min(start, s + lens[i] - d))
    return slots


def video_filter(sl, w, h):
    f = sl["src"]
    n = sl["n"]
    if er.is_image(f):
        chain = (f"scale={w * 2}:{h * 2}:force_original_aspect_ratio=increase,crop={w * 2}:{h * 2},"
                 f"format=yuv420p,zoompan=z='1+0.06*on/{n}':x='iw/2-(iw/zoom/2)':"
                 f"y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={FPS}")
    else:
        vw, vh, _, hlg = er.probe(f)
        src = (er.TONEMAP if hlg else "") + f"fps={FPS},format=yuv420p,setsar=1"
        if vw * h < vh * w:  # 縦長は背景をぼかして埋める
            chain = (f"{src},split[a][b];[b]scale={w}:{h}:force_original_aspect_ratio=increase,"
                     f"crop={w}:{h},gblur=sigma=40,eq=brightness=-0.12[bb];"
                     f"[a]scale=-2:{h}[ff];[bb][ff]overlay=(W-w)/2:0")
        else:
            chain = f"{src},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
        if sl["zoom"]:  # 拍の頭でグッと寄って戻る
            chain += (f",zoompan=z='1+0.07*max(0,1-on/9)':x='iw/2-(iw/zoom/2)':"
                      f"y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={FPS}")
    if sl["flash"]:
        chain += ",fade=t=in:st=0:d=0.2:color=white"
    return chain + ",setsar=1,format=yuv420p"


def render_cut(i, sl):
    w, h = (W, H) if sl["mode"] == "full" else (VW, VH)
    out = WORK / f"cut_{i:03d}.mp4"
    if er.is_image(sl["src"]):
        inp = ["-loop", "1", "-framerate", str(FPS), "-t", f"{sl['n'] / FPS + 0.5:.3f}"]
    else:
        inp = ["-ss", f"{sl['ss']:.3f}"]
    cmd = (["ffmpeg", "-y", "-loglevel", "error"] + inp + ["-i", str(sl["src"]),
           "-filter_complex", f"[0:v]{video_filter(sl, w, h)}[v]", "-map", "[v]",
           "-frames:v", str(sl["n"]), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-r", str(FPS), "-pix_fmt", "yuv420p", str(out)])
    subprocess.run(cmd, check=True)
    got = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                              "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)],
                             capture_output=True, text=True).stdout.strip() or 0)
    if got < sl["n"]:  # 素材が途中で切れていたら最後のコマで埋める
        tmp = out.with_suffix(".pad.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-vf",
                        f"tpad=stop_mode=clone:stop={sl['n'] - got}", "-frames:v", str(sl["n"]),
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                        str(tmp)], check=True)
        tmp.replace(out)
    return out


def concat(paths, out):
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in paths))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-c", "copy", str(out)], check=True)


def title_png(path):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for layer, color in ((shadow, (0, 0, 0, 200)), (img, (255, 255, 255, 255))):
        d = ImageDraw.Draw(layer)
        d.text((W // 2, H // 2 - 10), "雅祭前夜祭", font=er.font("SemiBold", 120), fill=color, anchor="mb")
        d.text((W // 2, H // 2 + 30), "LIVE 2026", font=er.font("Regular", 56),
               fill=color if layer is shadow else (*er.ROLE_COLOR, 255), anchor="mt")
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    Image.alpha_composite(shadow, img).save(path)


def main():
    WORK.mkdir(exist_ok=True)
    grid = json.loads((FOOT / "grid.json").read_text())
    song = FOOT / "song.mp3"
    song_len = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                     "-of", "csv=p=0", str(song)], capture_output=True,
                                    text=True).stdout)
    clips = er.read_clips(FOOT / "clips.txt")
    names = [c[0].name for c in clips]
    q, c = names.index(PANEL_QUIET_FROM), names.index(PANEL_CHORUS_FROM)

    slots = build_slots(grid)
    full = [s for s in slots if s["mode"] == "full"]
    quiet = [s for s in slots if s["mode"] == "panel" and s["f0"] < frame_at(beat_time(grid, 357))]
    chorus = [s for s in slots if s["mode"] == "panel" and s not in quiet]
    plan = assign(full, clips[:q]) + assign(quiet, clips[q:c]) + assign(chorus, clips[c:])
    plan.sort(key=lambda s: s["f0"])
    print(f"{len(plan)} cuts, BPM {grid['bpm']:.0f}")

    with ThreadPoolExecutor(max_workers=4) as ex:
        outs = list(ex.map(lambda a: render_cut(*a), enumerate(plan)))

    # 前半（全画面）：曲の頭までは黒、その後カット
    lead = plan[0]["f0"]
    black = WORK / "lead.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"color=c={BG_HEX}:s={W}x{H}:r={FPS}", "-frames:v", str(lead),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                    str(black)], check=True)
    full_outs = [o for o, s in zip(outs, plan) if s["mode"] == "full"]
    panel_outs = [o for o, s in zip(outs, plan) if s["mode"] == "panel"]
    concat([black] + full_outs, WORK / "part_full.mp4")
    concat(panel_outs, WORK / "part_panel.mp4")

    # 後半（エンドロール）：右にクレジット、最後にエンドカード
    roll_start = frame_at(beat_time(grid, 321))
    card_start = frame_at(beat_time(grid, END_CARD_BEAT))
    total = frame_at(song_len)
    scroll_secs = (card_start - roll_start) / FPS
    card_secs = (total - card_start) / FPS
    base = er.render_base(placeholder=False)
    roll = er.render_roll(er.parse(HERE / "credits.txt"))
    card = er.render_end_card("雅祭実行委員会", "Thank you for coming!")
    title_png(WORK / "title.png")

    full_secs = roll_start / FPS
    filt = ";".join([
        f"[3:v]format=rgba,fade=t=in:st=0.3:d=0.8:alpha=1,fade=t=out:st=6.4:d=0.8:alpha=1[ttl]",
        f"[0:v]fade=t=out:st={full_secs - 0.25:.3f}:d=0.25:color={BG_HEX}[f0]",
        f"[f0][ttl]overlay=0:0:eof_action=pass[full]",
        f"[2:v]fade=t=in:st=0:d=0.5:color={BG_HEX},"
        f"fade=t=out:st={scroll_secs - 1.0:.3f}:d=1.0:color={BG_HEX}[pn]",
        f"[1:v][pn]overlay={VX}:{VY}:eof_action=pass[roll]",
        f"[full][roll]concat=n=2:v=1:a=0[v]",
        f"[4:a]afade=t=out:st={song_len - 1.5:.3f}:d=1.5[a]",
    ])
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(WORK / "part_full.mp4"),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", str(WORK / "part_panel.mp4"),
           "-loop", "1", "-framerate", str(FPS), "-t", "7.4", "-i", str(WORK / "title.png"),
           "-i", str(song),
           "-filter_complex", filt, "-map", "[v]", "-map", "[a]",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-t", f"{song_len:.3f}", "-movflags", "+faststart",
           str(HERE / "mv.mp4")]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in er.frames(base, roll, card, scroll_secs, card_secs):
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {HERE / 'mv.mp4'}")


if __name__ == "__main__":
    main()
