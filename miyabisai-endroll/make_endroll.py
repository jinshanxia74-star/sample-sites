"""credits.txt からエンドロール動画 (1920x1080, 30fps, MP4) を作る。

使い方:
    python3 make_endroll.py                      # 60秒
    python3 make_endroll.py --duration 90        # 曲の長さに合わせる
    python3 make_endroll.py --audio song.mp3     # 曲をのせる（最後はフェードアウト）
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
W, H, FPS = 1920, 1080, 30
FONT = "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf"

BG = (8, 8, 12)
NAME_COLOR = (245, 245, 245)
ROLE_COLOR = (214, 178, 106)  # 落ち着いた金色
TITLE_COLOR = (255, 255, 255)

TITLE_SIZE, ROLE_SIZE, NAME_SIZE = 96, 40, 56
NAME_GAP = 82       # 名前どうしの行間
SECTION_GAP = 120   # 役職ブロックどうしの間
CENTER_GAP = 50     # 中央線から役職名・名前までの距離


def parse(path):
    """credits.txt を (種類, 文字) のリストに変換する。"""
    items = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("# ") or line == "#":
            continue
        if not line:
            items.append(("gap", ""))
        elif line.startswith("!"):
            items.append(("title", line[1:].strip()))
        elif line.startswith("[") and line.endswith("]"):
            items.append(("role", line[1:-1]))
        else:
            items.append(("name", line))
    return items


def render_roll(items):
    """流れる部分を縦長の1枚絵にする。"""
    f_title = ImageFont.truetype(FONT, TITLE_SIZE)
    f_role = ImageFont.truetype(FONT, ROLE_SIZE)
    f_name = ImageFont.truetype(FONT, NAME_SIZE)

    # 先に各要素の y 座標を決めてから描く
    ops, y, pending_role = [], 0, None
    for kind, text in items:
        if kind == "gap":
            y += SECTION_GAP - NAME_GAP if pending_role is None else 0
        elif kind == "title":
            ops.append(("title", text, y))
            y += int(TITLE_SIZE * 1.4)
        elif kind == "role":
            pending_role = text
        elif kind == "name":
            if pending_role is not None:
                # 役職名は最初の名前と同じ行に、ベースラインをそろえて置く
                ops.append(("role", pending_role, y + (NAME_SIZE - ROLE_SIZE)))
                pending_role = None
            ops.append(("name", text, y))
            y += NAME_GAP

    img = Image.new("RGB", (W, y + NAME_SIZE), BG)
    d = ImageDraw.Draw(img)
    cx = W // 2
    for kind, text, ty in ops:
        if kind == "title":
            d.text((cx, ty), text, font=f_title, fill=TITLE_COLOR, anchor="mt")
        elif kind == "role":
            d.text((cx - CENTER_GAP, ty), text, font=f_role, fill=ROLE_COLOR, anchor="rt")
        else:
            d.text((cx + CENTER_GAP, ty), text, font=f_name, fill=NAME_COLOR, anchor="lt")
    return np.asarray(img)


def render_end_card(main, sub):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W // 2, H // 2 - 20), main, font=ImageFont.truetype(FONT, 88),
           fill=TITLE_COLOR, anchor="mb")
    d.text((W // 2, H // 2 + 40), sub, font=ImageFont.truetype(FONT, 44),
           fill=ROLE_COLOR, anchor="mt")
    return np.asarray(img).astype(np.float32)


def edge_mask():
    """画面の上下の端でふわっと消えるようにするマスク。"""
    fade = 140
    m = np.ones(H, np.float32)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    m[:fade], m[-fade:] = ramp, ramp[::-1]
    return m[:, None, None]


def frames(roll, card, duration, card_secs):
    bg = np.array(BG, np.float32)
    mask = edge_mask()
    scroll_n = int((duration - card_secs) * FPS)
    travel = roll.shape[0] + H  # 下から入って上へ抜けきるまで
    canvas = np.empty((H, W, 3), np.float32)

    for i in range(scroll_n):
        top = H - int(travel * i / (scroll_n - 1))  # 絵の上端の画面上の位置
        canvas[:] = bg
        src0, src1 = max(0, -top), min(roll.shape[0], H - top)
        if src1 > src0:
            canvas[top + src0:top + src1] = roll[src0:src1]
        yield (bg + (canvas - bg) * mask).astype(np.uint8)

    card_n = int(card_secs * FPS)
    fade_in, fade_out = FPS, int(FPS * 1.5)
    for i in range(card_n):
        a = min(1.0, i / fade_in, (card_n - 1 - i) / fade_out)
        yield (bg + (card - bg) * max(0.0, a)).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--credits", type=Path, default=HERE / "credits.txt")
    p.add_argument("--out", type=Path, default=HERE / "endroll.mp4")
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--card-secs", type=float, default=8)
    p.add_argument("--card-main", default="雅祭実行委員会")
    p.add_argument("--card-sub", default="Thank you for coming!")
    p.add_argument("--audio", type=Path)
    a = p.parse_args()

    roll = render_roll(parse(a.credits)).astype(np.float32)
    card = render_end_card(a.card_main, a.card_sub)

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if a.audio:
        cmd += ["-i", str(a.audio), "-af", f"afade=t=out:st={a.duration - 3}:d=3",
                "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.out)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames(roll, card, a.duration, a.card_secs):
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
