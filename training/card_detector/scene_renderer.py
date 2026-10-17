"""Domain-modelled synthetic scenes for the OBB card detector (renderer v2).

Scenes model what the scanner actually sees: binder pages (dark and light, 2x2 to 4x5
pockets, single pages and spreads) with clear plastic pockets, loose and overlapping
cards, hand-held close-ups, sleeves/toploaders, playmats and cluttered desks, hands
over cards, and non-card rectangles (empty pockets, booster packs, phones, notes).
A simulated camera applies true perspective, then sensor/lens/compression effects.

Labels come from the renderer: every *complete* card (all four corners inside the
frame and at least `min_visible` of its area unoccluded) gets its true quad and a
derived minimum-area OBB. Cards cut by the frame or mostly hidden stay unlabeled and
are recorded as such. These are synthetic pixels; they are not camera evidence.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

CARD_ASPECT = 63 / 88  # width / height
TEX_H = 420
TEX_W = round(TEX_H * CARD_ASPECT)


@dataclass
class Placed:
    quad: np.ndarray  # world/image corners TL, TR, BR, BL of the artwork
    source: str
    kind: str  # 'card'
    context: str  # pocket, loose, held, case
    mask_id: int = 0


@dataclass
class SceneState:
    image: np.ndarray  # float32 RGB working image
    ids: np.ndarray  # int32 instance id of the topmost card per pixel (0 = none)
    cards: list[Placed] = field(default_factory=list)
    distractors: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- textures

def smooth_noise(rng: np.random.Generator, h: int, w: int, scale: float) -> np.ndarray:
    small = rng.normal(0, 1, (max(2, int(h / scale)), max(2, int(w / scale)))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def load_texture(path: str, rng: np.random.Generator) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert('RGB')
        arr = np.asarray(im.resize((TEX_W, TEX_H), Image.BILINEAR), dtype=np.float32)
    # Print/colour variation between card stock, photos and scans.
    arr = arr * rng.uniform(.85, 1.12) + rng.uniform(-12, 12, 3)
    if rng.random() < .25:  # holo/foil sheen
        yy, xx = np.mgrid[0:TEX_H, 0:TEX_W].astype(np.float32)
        phase = (xx * rng.uniform(-.04, .04) + yy * rng.uniform(-.04, .04)) + rng.uniform(0, 6.3)
        rainbow = np.stack([np.sin(phase), np.sin(phase + 2.1), np.sin(phase + 4.2)], -1)
        arr = arr + rainbow * rng.uniform(10, 45)
    return np.clip(arr, 0, 255)


def card_alpha(rng: np.random.Generator) -> np.ndarray:
    a = np.zeros((TEX_H, TEX_W), np.uint8)
    r = int(TEX_W * rng.uniform(.03, .05))
    cv2.rectangle(a, (r, 0), (TEX_W - 1 - r, TEX_H - 1), 255, -1)
    cv2.rectangle(a, (0, r), (TEX_W - 1, TEX_H - 1 - r), 255, -1)
    for cx, cy in [(r, r), (TEX_W - 1 - r, r), (r, TEX_H - 1 - r), (TEX_W - 1 - r, TEX_H - 1 - r)]:
        cv2.circle(a, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return a.astype(np.float32) / 255


def background(rng: np.random.Generator, h: int, w: int, textures: list[str]) -> tuple[np.ndarray, str]:
    kind = str(rng.choice(['wood', 'playmat', 'cloth', 'clutter', 'plain', 'dark'], p=[.22, .22, .16, .2, .1, .1]))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == 'wood':
        tone = rng.uniform([95, 60, 35], [200, 150, 105])
        f = rng.uniform(.004, .03)
        bands = np.sin(xx * f + np.sin(yy * f * .2) * 2) * rng.uniform(4, 14) + smooth_noise(rng, h, w, 12) * 6
        out = tone + bands[..., None] * [1, .8, .6]
    elif kind == 'playmat':
        art = load_texture(textures[int(rng.integers(len(textures)))], rng)
        # Large art crop: not card-shaped, so it is background, never a card.
        # Artwork interior only: no card frame, text box or card-shaped outline.
        y0 = int(rng.uniform(.12, .2) * TEX_H)
        crop = art[y0:y0 + int(TEX_H * .3), int(TEX_W * .1):int(TEX_W * .9)]
        out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(1, 6)) * rng.uniform(.5, 1)
    elif kind == 'cloth':
        tone = rng.uniform(20, 200, 3)
        weave = np.sin(xx * rng.uniform(.6, 1.4)) * np.sin(yy * rng.uniform(.6, 1.4)) * rng.uniform(2, 8)
        out = tone + (weave + smooth_noise(rng, h, w, 40) * 10)[..., None]
    elif kind == 'dark':
        out = rng.uniform(8, 40, 3) + smooth_noise(rng, h, w, 60)[..., None] * 6
    elif kind == 'plain':
        out = rng.uniform(150, 245, 3) + smooth_noise(rng, h, w, 90)[..., None] * 4
    else:  # clutter: blobs, strips and keyboard-like grids of mixed colours
        out = np.full((h, w, 3), rng.uniform(30, 160, 3), np.float32)
        for _ in range(int(rng.integers(15, 45))):
            c = rng.uniform(0, 255, 3)
            shape = rng.random()
            x, y = int(rng.uniform(0, w)), int(rng.uniform(0, h))
            s = int(rng.uniform(.03, .2) * min(w, h))
            if shape < .4:
                cv2.circle(out, (x, y), s, c.tolist(), -1, cv2.LINE_AA)
            elif shape < .7:
                cv2.ellipse(out, (x, y), (s, int(s * rng.uniform(.2, .6))), rng.uniform(0, 180), 0, 360, c.tolist(), -1, cv2.LINE_AA)
            else:
                pts = (np.array([x, y]) + rng.uniform(-s, s, (int(rng.integers(5, 9)), 2))).astype(np.int32)
                cv2.fillPoly(out, [cv2.convexHull(pts)], c.tolist(), cv2.LINE_AA)
        if rng.random() < .5:  # keyboard
            kx, ky = int(rng.uniform(0, w * .6)), int(rng.uniform(0, h * .6))
            k = int(rng.uniform(.015, .03) * w)
            for i in range(int(rng.integers(8, 16))):
                for j in range(int(rng.integers(3, 6))):
                    cv2.rectangle(out, (kx + i * k, ky + j * k), (kx + i * k + k - 3, ky + j * k + k - 3), rng.uniform(10, 70, 3).tolist(), -1)
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(.5, 3))
    shade = 1 - rng.uniform(0, .25) * (((xx / w - rng.uniform(.2, .8)) ** 2 + (yy / h - rng.uniform(.2, .8)) ** 2) * 2)
    return np.clip(out * shade[..., None], 0, 255).astype(np.float32), kind


# ----------------------------------------------------------------------------- compositing

def warp_onto(state: SceneState, tex: np.ndarray, alpha: np.ndarray, quad: np.ndarray, card_id: int = 0,
              shadow: float = 0.0) -> None:
    """Composite a texture onto the quad (TL, TR, BR, BL) of the scene; update the id map for cards."""
    h, w = state.ids.shape
    th, tw = alpha.shape
    src = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]])
    x0, y0 = np.floor(quad.min(0)).astype(int) - 12
    x1, y1 = np.ceil(quad.max(0)).astype(int) + 12
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    local = (quad - [x0, y0]).astype(np.float32)
    m = cv2.getPerspectiveTransform(src, local)
    size = (x1 - x0, y1 - y0)
    a = cv2.warpPerspective(alpha, m, size, flags=cv2.INTER_LINEAR)
    region = state.image[y0:y1, x0:x1]
    if shadow > 0:
        sa = cv2.GaussianBlur(np.roll(a, (3, 2), (0, 1)), (0, 0), 3)
        region *= (1 - shadow * sa)[..., None]
    pix = cv2.warpPerspective(tex, m, size, flags=cv2.INTER_LINEAR)
    region[:] = pix * a[..., None] + region * (1 - a[..., None])
    ids = state.ids[y0:y1, x0:x1]
    if card_id:
        ids[a > .5] = card_id
    else:
        ids[a > .5] = 0  # an occluder hides cards below


def quad_from(center, w, h, angle_deg, jitter=0.0, rng=None) -> np.ndarray:
    local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], np.float64)
    if jitter and rng is not None:
        local += rng.uniform(-1, 1, (4, 2)) * [w * jitter, h * jitter]
    r = math.radians(angle_deg)
    rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    return local @ rot.T + np.asarray(center, np.float64)


def place_card(state: SceneState, rng, textures, quad, context, shadow=.25) -> None:
    path = textures[int(rng.integers(len(textures)))]
    tex = load_texture(path, rng)
    idx = len(state.cards) + 1
    warp_onto(state, tex, card_alpha(rng), quad, idx, shadow)
    state.cards.append(Placed(quad=quad, source=path, kind='card', context=context, mask_id=idx))


def plastic(state: SceneState, rng, quad: np.ndarray, strength: float) -> None:
    """Clear sleeve/pocket film over a region: haze, specular streaks and edge ridges (not an occluder)."""
    th, tw = 64, 46
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    ang = rng.uniform(-1, 1)
    band = np.exp(-((xx * math.cos(ang) + yy * math.sin(ang) - rng.uniform(0, 60)) / rng.uniform(3, 14)) ** 2)
    film = np.full((th, tw, 3), 235, np.float32)
    haze = np.full((th, tw), rng.uniform(.03, .14) * strength, np.float32) + band * rng.uniform(0, .55) * strength
    haze[:2, :] += .25 * strength; haze[-2:, :] += .2 * strength; haze[:, :2] += .2 * strength; haze[:, -2:] += .2 * strength
    h, w = state.ids.shape
    x0, y0 = np.maximum(np.floor(quad.min(0)).astype(int) - 2, 0)
    x1, y1 = np.minimum(np.ceil(quad.max(0)).astype(int) + 2, [w, h])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    src = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]])
    m = cv2.getPerspectiveTransform(src, (quad - [x0, y0]).astype(np.float32))
    size = (x1 - x0, y1 - y0)
    a = cv2.warpPerspective(np.clip(haze, 0, .85), m, size, flags=cv2.INTER_LINEAR)
    f = cv2.warpPerspective(film, m, size, flags=cv2.INTER_LINEAR)
    region = state.image[y0:y1, x0:x1]
    region[:] = region * (1 - a[..., None]) + f * a[..., None]


def rect_texture(rng, w: int, h: int, textures: list[str], kind: str) -> np.ndarray:
    """Textures for non-card rectangles."""
    if kind == 'pack':  # crimped foil booster pack with stretched art
        art = load_texture(textures[int(rng.integers(len(textures)))], rng)
        t = cv2.resize(art[: int(TEX_H * .7)], (w, h)) * .8 + 40
        t[: max(2, h // 12)] = rng.uniform(120, 230, 3)
        t[-max(2, h // 12):] = rng.uniform(120, 230, 3)
        for y in range(0, max(1, h // 12), 3):
            t[y] *= .7
        return np.clip(t, 0, 255)
    if kind == 'phone':
        t = np.full((h, w, 3), rng.uniform(5, 40), np.float32)
        if rng.random() < .6:
            t[h // 12: -h // 12, w // 20: -w // 20] = rng.uniform(40, 240, 3)
        return t
    if kind == 'note':
        t = np.full((h, w, 3), rng.uniform(200, 250, 3), np.float32)
        for y in range(h // 8, h, max(4, h // 14)):
            t[y:y + 2, w // 10: int(w * rng.uniform(.4, .9))] = rng.uniform(20, 90)
        return t
    # deck box / sealed product
    return np.full((h, w, 3), rng.uniform(20, 230, 3), np.float32) + smooth_noise(rng, h, w, 8)[..., None] * 10


def distractor(state: SceneState, rng, textures, center, size) -> None:
    kind = str(rng.choice(['pack', 'phone', 'note', 'box']))
    aspect = {'pack': rng.uniform(.5, .6), 'phone': rng.uniform(.45, .5), 'note': rng.uniform(.7, 1.3), 'box': rng.uniform(.6, 1.6)}[kind]
    hgt = size * rng.uniform(.8, 1.6)
    wid = hgt * aspect
    q = quad_from(center, wid, hgt, rng.uniform(0, 360), .02, rng)
    tw, th = 64, max(8, int(64 / aspect))
    tex = rect_texture(rng, tw, th, textures, kind)
    alpha = np.ones((th, tw), np.float32)
    warp_onto(state, tex, alpha, q, 0, .3)
    state.distractors.append(kind)


# ----------------------------------------------------------------------------- layouts (world plane)

def binder_world(rng, textures, card_w: float, empty: bool = False, upright: bool = True):
    """World image with one page or a spread; returns state and page outline for camera framing."""
    rows, cols = [(3, 3), (3, 3), (3, 3), (3, 4), (4, 4), (4, 5), (2, 2), (2, 3), (2, 4)][int(rng.integers(9))]
    spread = rng.random() < .45
    gap = card_w * rng.uniform(.04, .12)
    pocket_w = card_w * rng.uniform(1.03, 1.1)
    pocket_h = pocket_w / CARD_ASPECT * rng.uniform(.98, 1.03)
    margin = card_w * rng.uniform(.15, .35)
    page_w = cols * pocket_w + (cols - 1) * gap + 2 * margin
    page_h = rows * pocket_h + (rows - 1) * gap + 2 * margin
    spine = card_w * rng.uniform(.15, .5) if spread else 0
    content_w = page_w * (2 if spread else 1) + spine
    pad = max(content_w, page_h) * rng.uniform(.25, .9)
    W, H = int(content_w + 2 * pad), int(page_h + 2 * pad)
    bg, bg_kind = background(rng, H, W, textures)
    state = SceneState(image=bg, ids=np.zeros((H, W), np.int32))
    dark = rng.random() < .7
    page_color = rng.uniform(8, 45, 3) if dark else rng.uniform(170, 240, 3)
    cover = rng.uniform(5, 60, 3) if rng.random() < .8 else rng.uniform(60, 200, 3)
    ox, oy = pad, pad
    # Binder cover rim around the page(s).
    rim = card_w * rng.uniform(.05, .25)
    cv2.rectangle(state.image, (int(ox - rim), int(oy - rim)), (int(ox + content_w + rim), int(oy + page_h + rim)), cover.tolist(), -1)
    full = rng.random() < .55
    fill = 0.0 if empty else 1.0 if full else rng.uniform(.2, .95)
    pockets = []
    for p in range(2 if spread else 1):
        px = ox + p * (page_w + spine)
        page = np.full((int(page_h), int(page_w), 3), page_color, np.float32) + smooth_noise(rng, int(page_h), int(page_w), 30)[..., None] * 3
        state.image[int(oy):int(oy) + page.shape[0], int(px):int(px) + page.shape[1]] = page
        for r in range(rows):
            for c in range(cols):
                x = px + margin + c * (pocket_w + gap)
                y = oy + margin + r * (pocket_h + gap)
                pockets.append(np.array([[x, y], [x + pocket_w, y], [x + pocket_w, y + pocket_h], [x, y + pocket_h]], np.float64))
    if spread:
        sx = ox + page_w
        cv2.rectangle(state.image, (int(sx), int(oy - rim)), (int(sx + spine), int(oy + page_h + rim)), (cover * .7).tolist(), -1)
        if rng.random() < .5:  # rings
            for k in range(3):
                cy = int(oy + page_h * (.2 + .3 * k))
                cv2.ellipse(state.image, (int(sx + spine / 2), cy), (int(spine * .45), int(spine * .2)), 0, 0, 360, (190, 190, 195), max(2, int(spine * .08)), cv2.LINE_AA)
    for q in pockets:
        if rng.random() < fill:
            center = q.mean(0) + rng.uniform(-.03, .03, 2) * card_w
            ch = card_w / CARD_ASPECT
            cq = quad_from(center, card_w, ch, rng.uniform(-2.5, 2.5) + (180 if not upright and rng.random() < .04 else 0), .004, rng)
            place_card(state, rng, textures, cq, 'pocket', shadow=.1)
        # Pocket film over cards and empty pockets alike (empty pockets are hard negatives).
        plastic(state, rng, q, rng.uniform(.5, 1.3))
    outline = np.array([[ox - rim, oy - rim], [ox + content_w + rim, oy - rim], [ox + content_w + rim, oy + page_h + rim], [ox - rim, oy + page_h + rim]])
    return state, outline, {'layout': 'binder', 'rows': rows, 'cols': cols, 'spread': spread, 'dark_page': dark, 'background': bg_kind}


def loose_world(rng, textures, card_w: float, count: int | None = None, upright: bool = True):
    n = count if count is not None else int(rng.choice([1, 2, 3, 4, 6, 8, 12]))
    side = card_w * (2.2 + math.sqrt(n) * 1.9)
    W = H = int(side * rng.uniform(1.4, 2.2))
    bg, bg_kind = background(rng, H, W, textures)
    state = SceneState(image=bg, ids=np.zeros((H, W), np.int32))
    c0 = np.array([W / 2, H / 2])
    for _ in range(int(rng.integers(0, 4))):
        distractor(state, rng, textures, c0 + rng.uniform(-.45, .45, 2) * side, card_w * rng.uniform(.8, 1.6))
    fan = rng.random() < .25
    for i in range(n):
        if fan:
            center = c0 + np.array([i * card_w * rng.uniform(.25, .5), i * card_w * .05])
            angle = -20 + i * rng.uniform(4, 10)
        else:
            center = c0 + rng.uniform(-.38, .38, 2) * side
            # Upright scope: within +/-60 degrees, spread evenly (sideways/upside-down are out of scope).
            angle = rng.uniform(-60, 60) if upright else rng.uniform(-180, 180) if rng.random() < .4 else rng.uniform(-25, 25)
        q = quad_from(center, card_w, card_w / CARD_ASPECT, angle, .003, rng)
        sleeve = rng.random()
        if sleeve < .2:  # toploader: thick clear shell drawn first
            sq = quad_from(center, card_w * 1.18, card_w / CARD_ASPECT * 1.12, angle)
            tex = np.full((64, 46, 3), 225, np.float32)
            a = np.full((64, 46), .25, np.float32)
            warp_onto(state, tex, a, sq, 0, .3)
        place_card(state, rng, textures, q, 'loose', shadow=.3)
        if sleeve < .45:
            plastic(state, rng, quad_from(center, card_w * 1.05, card_w / CARD_ASPECT * 1.04, angle), rng.uniform(.4, 1))
    outline = np.array([[c0[0] - side / 2, c0[1] - side / 2], [c0[0] + side / 2, c0[1] - side / 2],
                        [c0[0] + side / 2, c0[1] + side / 2], [c0[0] - side / 2, c0[1] + side / 2]])
    return state, outline, {'layout': 'fan' if fan else 'loose', 'cards_placed': n, 'background': bg_kind}


def negative_world(rng, textures, card_w: float):
    """No cards: empty binder pages, or packs, phones, notes and boxes on varied surfaces."""
    if rng.random() < .5:
        state, outline, meta = binder_world(rng, textures, card_w, empty=True)
        meta['layout'] = 'negative-empty-binder'
        return state, outline, meta
    W = H = int(card_w * rng.uniform(6, 10))
    bg, bg_kind = background(rng, H, W, textures)
    state = SceneState(image=bg, ids=np.zeros((H, W), np.int32))
    for _ in range(int(rng.integers(1, 6))):
        distractor(state, rng, textures, rng.uniform(.2, .8, 2) * [W, H], card_w * rng.uniform(.8, 1.8))
    outline = np.array([[W * .2, H * .2], [W * .8, H * .2], [W * .8, H * .8], [W * .2, H * .8]])
    return state, outline, {'layout': 'negative-objects', 'background': bg_kind}


# ----------------------------------------------------------------------------- camera

def camera_homography(rng, outline: np.ndarray, out_w: int, out_h: int, fill: float, upright: bool = True) -> np.ndarray:
    """World plane -> image via a tilted pinhole camera framing `outline` at `fill` of the frame."""
    c = outline.mean(0)
    extent = np.linalg.norm(outline[2] - outline[0]) / 2
    tilt = math.radians(rng.choice([rng.uniform(0, 12), rng.uniform(10, 32), rng.uniform(28, 48)], p=[.35, .45, .2]))
    axis = rng.uniform(0, 2 * math.pi)
    roll = math.radians(rng.uniform(-15, 15) + (90 * rng.choice([-1, 1]) if not upright and rng.random() < .15 else 0))
    pts = (outline - c) / extent  # unit-ish square
    pts3 = np.c_[pts, np.zeros(4)]
    ax = np.array([math.cos(axis), math.sin(axis), 0])
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + math.sin(tilt) * K + (1 - math.cos(tilt)) * K @ K
    Rz = np.array([[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
    p = (Rz @ R @ pts3.T).T
    dist = rng.uniform(2.2, 3.5)
    proj = p[:, :2] / (dist + p[:, 2:3]) * dist
    span = np.ptp(proj, 0).max()
    scale = fill * min(out_w, out_h) / max(span, 1e-6)
    shift = np.array([out_w / 2, out_h / 2]) + rng.uniform(-.18, .18, 2) * [out_w, out_h] - proj.mean(0) * scale
    dst = proj * scale + shift
    return cv2.getPerspectiveTransform(outline.astype(np.float32), dst.astype(np.float32))


def hands(state: SceneState, rng, count: int) -> None:
    """Procedural hands/fingers in image space; they occlude whatever is below."""
    h, w = state.ids.shape
    for _ in range(count):
        skin = np.array([rng.uniform(150, 245), rng.uniform(100, 190), rng.uniform(80, 160)], np.float32) * rng.uniform(.5, 1.05)
        mask = np.zeros((h, w), np.uint8)
        edge = int(rng.integers(4))
        base = np.array([[rng.uniform(0, w), h + 20], [rng.uniform(0, w), -20], [-20, rng.uniform(0, h)], [w + 20, rng.uniform(0, h)]][edge])
        palm_r = int(min(w, h) * rng.uniform(.08, .2))
        tip = base + (np.array([w / 2, h / 2]) - base) * rng.uniform(.25, .65) + rng.uniform(-.2, .2, 2) * [w, h]
        cv2.ellipse(mask, tuple(base.astype(int)), (int(palm_r * 1.6), palm_r), rng.uniform(0, 180), 0, 360, 255, -1, cv2.LINE_AA)
        direction = (tip - base) / (np.linalg.norm(tip - base) + 1e-6)
        normal = np.array([-direction[1], direction[0]])
        for f in range(int(rng.integers(1, 5))):
            start = base + normal * (f - 1.5) * palm_r * .5
            end = start + direction * np.linalg.norm(tip - base) * rng.uniform(.6, 1.1) + normal * rng.uniform(-.1, .1) * palm_r
            cv2.line(mask, tuple(start.astype(int)), tuple(end.astype(int)), 255, max(3, int(palm_r * rng.uniform(.28, .4))), cv2.LINE_AA)
        a = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 1.2)
        shade = (1 + smooth_noise(rng, h, w, 25) * .06)[..., None]
        state.image[:] = state.image * (1 - a[..., None]) + skin * shade * a[..., None]
        state.ids[a > .5] = 0


def camera_effects(img: np.ndarray, rng) -> tuple[np.ndarray, dict]:
    h, w = img.shape[:2]
    fx: dict = {}
    out = img * rng.uniform(.55, 1.25) * rng.uniform(.88, 1.12, 3)  # exposure, white balance
    gamma = rng.uniform(.8, 1.25)
    out = 255 * (np.clip(out, 0, 255) / 255) ** gamma
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if rng.random() < .55:  # lamp glare blobs
        for _ in range(int(rng.integers(1, 4))):
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            rx, ry = w * rng.uniform(.04, .25), h * rng.uniform(.02, .12)
            t = rng.uniform(0, math.pi)
            u = (xx - cx) * math.cos(t) + (yy - cy) * math.sin(t)
            v = -(xx - cx) * math.sin(t) + (yy - cy) * math.cos(t)
            m = np.exp(-((u / rx) ** 2 + (v / ry) ** 2) * 2) * rng.uniform(.3, .95)
            out = out + (255 - out) * m[..., None]
        fx['glare'] = True
    vig = rng.uniform(0, .45)
    out *= (1 - vig * (((xx / w - .5) ** 2 + (yy / h - .5) ** 2) * 2))[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    r = rng.random()
    if r < .25:  # motion blur
        k = int(rng.integers(3, 15))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2] = 1
        kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((k / 2, k / 2), rng.uniform(0, 180), 1), (k, k))
        out = cv2.filter2D(out, -1, kernel / max(kernel.sum(), 1e-6))
        fx['motion_blur'] = k
    elif r < .55:
        s = rng.uniform(.4, 2.2)
        out = cv2.GaussianBlur(out, (0, 0), s)
        fx['defocus'] = round(s, 2)
    if rng.random() < .3:  # video-style resolution loss
        f = rng.uniform(.35, .75)
        out = cv2.resize(cv2.resize(out, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
        fx['downscale'] = round(f, 2)
    noise = rng.uniform(1, 9)
    out = np.clip(out + rng.normal(0, noise, out.shape) * (1 + (255 - out) / 255), 0, 255).astype(np.uint8)
    if rng.random() < .25:  # chroma shift
        s = int(rng.integers(1, 3))
        out[..., 0] = np.roll(out[..., 0], s, 1)
        out[..., 2] = np.roll(out[..., 2], -s, 0)
    q = int(rng.integers(35, 95))
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format='JPEG', quality=q)
    out = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert('RGB'))
    fx.update(noise=round(noise, 1), jpeg=q)
    return out, fx


# ----------------------------------------------------------------------------- scene

def extend_world(state: SceneState, H: np.ndarray, out_w: int, out_h: int, rng, textures) -> tuple[SceneState, np.ndarray]:
    """Grow the world with fresh background so the camera never sees mirrored (unlabeled) content."""
    frame = np.float64([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    hh, ww = state.ids.shape
    try:
        world = cv2.perspectiveTransform(frame[None], np.linalg.inv(H))[0]
    except np.linalg.LinAlgError:
        return state, H
    # Rays above the horizon map behind the camera; clamp to a generous margin.
    limit = int(.5 * max(ww, hh))  # beyond this, edge replication of fresh background is enough
    world[:, 0] = np.clip(world[:, 0], -limit, ww + limit)
    world[:, 1] = np.clip(world[:, 1], -limit, hh + limit)
    x0, y0 = min(0, int(np.floor(world[:, 0].min())) - 4), min(0, int(np.floor(world[:, 1].min())) - 4)
    x1, y1 = max(ww, int(np.ceil(world[:, 0].max())) + 4), max(hh, int(np.ceil(world[:, 1].max())) + 4)
    if (x0, y0, x1, y1) == (0, 0, ww, hh):
        return state, H
    big_w, big_h = x1 - x0, y1 - y0
    # Cap the canvas; a coarser background is fine far from the cards.
    scale = min(1.0, 5000 / max(big_w, big_h))
    bg, _ = background(rng, max(2, int(big_h * scale)), max(2, int(big_w * scale)), textures)
    bg = cv2.resize(bg, (big_w, big_h), interpolation=cv2.INTER_LINEAR) if scale < 1 else bg
    bg[-y0:-y0 + hh, -x0:-x0 + ww] = state.image
    ids = np.zeros((big_h, big_w), np.int32)
    ids[-y0:-y0 + hh, -x0:-x0 + ww] = state.ids
    shift = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float64)
    out = SceneState(image=bg, ids=ids, cards=state.cards, distractors=state.distractors)
    for card in out.cards:
        card.quad = card.quad - [x0, y0]
    return out, H @ shift


def render(seed: int, textures: list[str], out_w: int, out_h: int, min_visible: float = .6, upright: bool = True) -> tuple[np.ndarray, dict]:
    """upright=True (renderer v3): cards within +/-60 degrees; upright=False reproduces renderer v2."""
    rng = np.random.default_rng(seed)
    kind = str(rng.choice(['binder', 'loose', 'held', 'negative'], p=[.5, .22, .16, .12]))
    card_w = float(rng.uniform(60, 120))  # world pixels per card width; near final image scale
    if kind == 'binder':
        state, outline, meta = binder_world(rng, textures, card_w, upright=upright)
        fill = rng.choice([rng.uniform(.55, .95), rng.uniform(.9, 1.6)], p=[.65, .35])
    elif kind == 'loose':
        state, outline, meta = loose_world(rng, textures, card_w, upright=upright)
        fill = rng.uniform(.5, 1.1)
    elif kind == 'held':
        state, outline, meta = loose_world(rng, textures, card_w, count=int(rng.choice([1, 1, 2, 3])), upright=upright)
        meta['layout'] = 'held'
        fill = rng.uniform(.9, 1.6)
    else:
        state, outline, meta = negative_world(rng, textures, card_w)
        fill = rng.uniform(.5, 1.3)
    H = camera_homography(rng, outline, out_w, out_h, fill, upright)
    state, H = extend_world(state, H, out_w, out_h, rng, textures)
    img = cv2.warpPerspective(state.image, H, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ids = cv2.warpPerspective(state.ids.astype(np.float32), H, (out_w, out_h), flags=cv2.INTER_NEAREST, borderValue=0).astype(np.int32)
    if kind == 'held':
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0, 1.5)) if rng.random() < .3 else img
    image_state = SceneState(image=img.astype(np.float32), ids=ids)
    n_hands = int(rng.choice([0, 1, 2], p=[.45, .4, .15])) if kind != 'held' else int(rng.choice([1, 2]))
    hands(image_state, rng, n_hands)
    pixels, fx = camera_effects(image_state.image, rng)
    instances, unlabeled = [], []
    for card in state.cards:
        quad = cv2.perspectiveTransform(card.quad[None].astype(np.float64), H)[0]
        inside = bool(np.all(quad[:, 0] >= 0) and np.all(quad[:, 0] <= out_w - 1) and np.all(quad[:, 1] >= 0) and np.all(quad[:, 1] <= out_h - 1))
        area = abs(cv2.contourArea(quad.astype(np.float32)))
        visible = float((image_state.ids == card.mask_id).sum() / max(area, 1))
        short = min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[3] - quad[0]))
        record = {'true_corners': quad.round(2).tolist(), 'context': card.context, 'visible_fraction': round(min(visible, 1.0), 3),
                  'source': card.source}
        if inside and visible >= min_visible and short >= 10:
            instances.append(record)
        else:
            record['reason'] = 'outside_frame' if not inside else 'occluded' if visible < min_visible else 'too_small'
            unlabeled.append(record)
    meta.update({'seed': seed, 'kind': kind, 'width': out_w, 'height': out_h, 'effects': fx, 'hands': n_hands,
                 'distractors': state.distractors, 'label_origin': 'synthetic_renderer_v3_upright' if upright else 'synthetic_renderer_v2', 'real_camera_validation': False})
    return pixels, {**meta, 'instances': instances, 'unlabeled_cards': unlabeled}
