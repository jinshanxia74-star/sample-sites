"""credits.txt からエンドロール動画 (1920x1080, 30fps, MP4) を作る。

左に動画、右にクレジットが流れる2分割レイアウト。最後は全画面のエンドカード。

使い方:
    python3 make_endroll.py                               # 動画なし（左は仮の枠）
    python3 make_endroll.py --video a.mp4 b.mp4 c.mp4     # 左に流す動画（順につなぐ）
    python3 make_endroll.py --duration 90 --audio song.mp3
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
W, H, FPS = 1920, 1080, 30

SERIF = "/usr/share/fonts/opentype/noto/NotoSerifCJK-{}.ttc"  # index 0 = JP


def font(weight, size):
    return ImageFont.truetype(SERIF.format(weight), size, index=0)


BG = (8, 8, 12)
TITLE_COLOR = (255, 255, 255)
NAME_COLOR = (240, 238, 232)
ROLE_COLOR = (200, 168, 104)  # 落ち着いた金色
FRAME_COLOR = (90, 76, 50)

# 左：動画パネル（16:9）
VX, VY, VW, VH = 80, 225, 1120, 630
# 右：クレジットの列
CX = 1570          # 列の中心
COL_W = 620        # 列の幅（これを超える行は自動で縮める）

TITLE_SIZE, ROLE_SIZE, NAME_SIZE = 64, 30, 48
ROLE_TO_NAME = 52   # 役職名から最初の名前まで
NAME_GAP = 70       # 名前どうしの行間
SECTION_GAP = 110   # 役職ブロックどうしの間


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


def fitted(weight, size, text):
    """列の幅に収まるまで文字を小さくする。"""
    f = font(weight, size)
    while f.getlength(text) > COL_W and size > 20:
        size -= 2
        f = font(weight, size)
    return f


def render_roll(items):
    """右の列に流れる部分を縦長の1枚絵（列の幅だけ）にする。"""
    ops, y = [], 0
    for kind, text in items:
        if kind == "gap":
            y += SECTION_GAP - NAME_GAP
        elif kind == "title":
            ops.append((fitted("SemiBold", TITLE_SIZE, text), TITLE_COLOR, text, y))
            y += int(TITLE_SIZE * 1.45)
        elif kind == "role":
            ops.append((fitted("Regular", ROLE_SIZE, text), ROLE_COLOR, text, y))
            y += ROLE_TO_NAME
        else:
            ops.append((fitted("Medium", NAME_SIZE, text), NAME_COLOR, text, y))
            y += NAME_GAP

    img = Image.new("RGB", (COL_W, y + NAME_SIZE), BG)
    d = ImageDraw.Draw(img)
    for f, color, text, ty in ops:
        d.text((COL_W // 2, ty), text, font=f, fill=color, anchor="mt")
    return np.asarray(img).astype(np.float32)



def render_base(placeholder):
    """クレジット以外の固定部分（背景・動画の枠）。"""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    if placeholder:
        d.rectangle((VX, VY, VX + VW, VY + VH), fill=(22, 22, 28))
        d.text((VX + VW // 2, VY + VH // 2), "ここに動画", font=font("Regular", 44),
               fill=(90, 90, 100), anchor="mm")
    # 細い金の枠
    d.rectangle((VX - 12, VY - 12, VX + VW + 12, VY + VH + 12), outline=FRAME_COLOR, width=1)
    return np.asarray(img).astype(np.float32)


def render_end_card(main, sub):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W // 2, H // 2 - 24), main, font=font("SemiBold", 84), fill=TITLE_COLOR, anchor="mb")
    d.line((W // 2 - 60, H // 2, W // 2 + 60, H // 2), fill=ROLE_COLOR, width=1)
    d.text((W // 2, H // 2 + 28), sub, font=font("Regular", 38), fill=ROLE_COLOR, anchor="mt")
    return np.asarray(img).astype(np.float32)


def edge_mask():
    """クレジットが列の上下の端でふわっと消えるようにするマスク。"""
    fade = 160
    m = np.ones(H, np.float32)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    m[:fade], m[-fade:] = ramp, ramp[::-1]
    return m[:, None, None]


def frames(base, roll, card, scroll_secs, card_secs):
    bg = np.array(BG, np.float32)
    mask = edge_mask()
    x0 = CX - COL_W // 2
    scroll_n = int(scroll_secs * FPS)
    travel = roll.shape[0] + H  # 下から入って上へ抜けきるまで
    col = np.empty((H, COL_W, 3), np.float32)
    frame = base.copy()

    for i in range(scroll_n):
        top = H - int(travel * i / (scroll_n - 1))  # 絵の上端の画面上の位置
        col[:] = bg
        src0, src1 = max(0, -top), min(roll.shape[0], H - top)
        if src1 > src0:
            col[top + src0:top + src1] = roll[src0:src1]
        frame[:, x0:x0 + COL_W] = bg + (col - bg) * mask
        yield frame.astype(np.uint8)

    card_n = int(card_secs * FPS)
    fade_in, fade_out = FPS, int(FPS * 1.5)
    for i in range(card_n):
        a = max(0.0, min(1.0, i / fade_in, (card_n - 1 - i) / fade_out))
        yield (bg + (card - bg) * a).astype(np.uint8)


def ffmpeg_cmd(a, scroll_secs):
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    for v in a.video:
        cmd += ["-i", str(v)]
    audio_idx = 1 + len(a.video)
    if a.audio:
        cmd += ["-i", str(a.audio)]

    if a.video:
        # 各動画を枠に収めてつなぎ、足りなければ最後のコマで止める。
        # クレジットが流れ終わる1秒前から暗転させ、エンドカードの上には出さない。
        bg_hex = "0x%02X%02X%02X" % BG
        parts = [f"[{i + 1}:v]scale={VW}:{VH}:force_original_aspect_ratio=decrease,"
                 f"pad={VW}:{VH}:(ow-iw)/2:(oh-ih)/2:color={bg_hex},setsar=1,fps={FPS}[v{i}]"
                 for i in range(len(a.video))]
        joined = "".join(f"[v{i}]" for i in range(len(a.video)))
        parts.append(f"{joined}concat=n={len(a.video)}:v=1:a=0,"
                     f"tpad=stop_mode=clone:stop_duration={a.duration},"
                     f"trim=duration={scroll_secs},"
                     f"fade=t=in:st=0:d=1:color={bg_hex},"
                     f"fade=t=out:st={scroll_secs - 1.5}:d=1.5:color={bg_hex}[clip]")
        parts.append(f"[0:v][clip]overlay={VX}:{VY}:eof_action=pass[out]")
        cmd += ["-filter_complex", ";".join(parts), "-map", "[out]"]
    else:
        cmd += ["-map", "0:v"]

    if a.audio:
        cmd += ["-map", f"{audio_idx}:a", "-af", f"afade=t=out:st={a.duration - 3}:d=3",
                "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.out)]
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--credits", type=Path, default=HERE / "credits.txt")
    p.add_argument("--out", type=Path, default=HERE / "endroll.mp4")
    p.add_argument("--video", type=Path, nargs="*", default=[])
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--card-secs", type=float, default=8)
    p.add_argument("--card-main", default="雅祭実行委員会")
    p.add_argument("--card-sub", default="Thank you for coming!")
    p.add_argument("--audio", type=Path)
    a = p.parse_args()

    scroll_secs = a.duration - a.card_secs
    base = render_base(placeholder=not a.video)
    roll = render_roll(parse(a.credits))
    card = render_end_card(a.card_main, a.card_sub)

    proc = subprocess.Popen(ffmpeg_cmd(a, scroll_secs), stdin=subprocess.PIPE)
    for f in frames(base, roll, card, scroll_secs, a.card_secs):
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
