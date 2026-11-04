# sanglianwork.py — a monogram drawn out of letters
import numpy as np
from PIL import Image, ImageFont, ImageDraw

TAU = 2 * np.pi
DEPTH = 0.22                     # how thick the S is
CELL = (8, 13)                   # one letter = one pixel
RAMP = " .,:;=+*sw#%@"           # dark → bright

# — the shape: rasterise an S, then give it a front, a back and walls
def glyph_mask(ch="S", size=420):
    font = ImageFont.truetype("Poppins-Bold.ttf", size)
    img = Image.new("L", (size, size), 0)
    ImageDraw.Draw(img).text((size * 0.18, -size * 0.12), ch, 255, font=font)
    return np.asarray(img) > 127

def surface(mask):
    ys, xs = np.nonzero(mask)
    p = np.stack([xs, -ys], 1) / mask.shape[0] * 2.4
    p -= p.mean(0)
    front = np.c_[p, np.full(len(p), DEPTH)]
    back = np.c_[p, np.full(len(p), -DEPTH)]
    return front, back

def walls(mask, steps=14):
    edge = mask & ~erode(mask)
    gy, gx = np.gradient(blur(mask.astype(float)))
    ys, xs = np.nonzero(edge)
    n = -np.stack([gx[ys, xs], -gy[ys, xs], 0 * xs], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    z = np.linspace(-DEPTH, DEPTH, steps)
    return edge_points(xs, ys, z), np.repeat(n, steps, 0)

# — the stroke: where a pen would be at each point of the S
def stroke_param(p, r=0.5):
    low = p[:, 1] < 0
    a = np.arctan2(p[:, 1] + r, p[:, 0])
    b = (-np.pi / 2 - np.arctan2(p[:, 1] - r, p[:, 0])) % TAU
    s = np.where(low, (a + 2.6) / 4.2 * 0.5, 0.5 + b / 4.2 * 0.5)
    return np.clip(s, 0, 1)

# — light: a key light, a moving rim light, and a lime glint
def lights(t):
    key = np.array([-0.35, 0.55, 0.76])
    rim = np.array([np.cos(t * 0.8), 0.4, np.sin(t * 0.8)])
    return key / np.linalg.norm(key), rim / np.linalg.norm(rim)

def shade(n, t):
    key, rim = lights(t)
    d = np.clip(n @ key, 0, 1)
    r = np.clip(n @ rim, 0, 1) ** 3
    return 0.14 + 0.72 * d + 0.55 * r

def spin(t):
    return TAU * 0.085 * max(t - 5.2, 0) ** 1.08

def rot_y(p, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack([c * p[:, 0] + s * p[:, 2], p[:, 1],
                     -s * p[:, 0] + c * p[:, 2]], 1)

def project(p, cam=7.0, f=2240):
    w = cam - p[:, 2]
    return f * p[:, 0] / w, -f * p[:, 1] / w, w

# — orbits: two tilted rings, like the S is something being built
def orbit(tilt, n=420, r=1.55):
    a = np.linspace(0, TAU, n, endpoint=False)
    ring = np.c_[r * np.cos(a), 0 * a, r * np.sin(a)]
    return rot_z(rot_x(ring, 0.28), tilt)

ORBITS = [orbit(+0.72), orbit(-0.72)]

# — the draw: the pen sweeps the stroke, letters land behind it
DRAW = (0.4, 5.0)

def gather(s, t):
    """points appear where the pen has passed"""
    head = (t - DRAW[0]) / (DRAW[1] - DRAW[0]) * 1.08
    shown = s < head
    spark = shown & (head - s < 0.035)
    return shown, spark

def zbuffer(cols, rows, depth):
    order = np.argsort(-depth)
    grid = np.full((ROWS, COLS), -1)
    grid[rows[order], cols[order]] = order
    return grid

def letters(b):
    return np.take(list(RAMP), np.clip(b * len(RAMP), 0, len(RAMP) - 1).astype(int))

def frame(t):
    p = rot_y(SHAPE, spin(t))
    x, y, w = project(p)
    shown, spark = gather(STROKE, t)
    b = shade(rot_y(NORMALS, spin(t)), t)
    glint = (b > 1.0) & (RNG.random(len(b)) < 0.06)
    canvas = render(x[shown], y[shown], w[shown], b[shown])
    canvas.paint(spark, "*", LIME)
    canvas.paint(glint, "+", LIME)
    canvas.reflect(floor=0.92, fade=0.18)
    return canvas

def run(seconds=20, fps=30):
    for i in range(seconds * fps):
        yield frame(i / fps).to_image()

# sanglianwork.de — built with AI, shipped for real.
