"""
Text rendering engine with 10 TikTok-optimized display styles.
Uses PIL/Pillow for text rendering onto video frames.
"""

import os
import math
import urllib.request
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Font download URLs (Google Fonts GitHub)
FONT_URLS = {
    "roboto": "https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Bold.ttf",
    "montserrat": "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf",
    "bebas_neue": "https://github.com/dharmatype/Bebas-Neue/raw/master/fonts/BebasNeue-Regular.ttf",
}

SYSTEM_FONTS = {
    "arial": ["arial.ttf", "arialbd.ttf"],
    "impact": ["impact.ttf"],
    "times": ["times.ttf", "timesbd.ttf"],
    "courier": ["cour.ttf", "courbd.ttf"],
    "comic_sans": ["comicbd.ttf", "comic.ttf"],
}

DISPLAY_STYLES = [
    "karaoke", "subtitles", "word_pop", "typewriter", "word_wave",
    "glow_pulse", "slide_in", "bounce_drop", "fade_flow", "neon_flash",
]

LINE_MODES = ["single_line", "two_lines", "three_lines", "word_by_word", "full_verse"]

POSITIONS = ["top", "upper_third", "center", "lower_third", "bottom"]


def _parse_color(hex_str):
    hex_str = hex_str.lstrip("#")
    try:
        if len(hex_str) >= 6:
            return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16), 255)
    except ValueError:
        pass
    return (255, 255, 255, 255)


def _ease_out_bounce(t):
    t = max(0.0, min(1.0, t))
    if t < 1/2.75:
        return 7.5625 * t * t
    elif t < 2/2.75:
        t -= 1.5/2.75
        return 7.5625 * t * t + 0.75
    elif t < 2.5/2.75:
        t -= 2.25/2.75
        return 7.5625 * t * t + 0.9375
    else:
        t -= 2.625/2.75
        return 7.5625 * t * t + 0.984375


def _ease_out_elastic(t):
    if t <= 0: return 0.0
    if t >= 1: return 1.0
    p = 0.3
    return math.pow(2, -10*t) * math.sin((t - p/4) * (2*math.pi)/p) + 1


def _ease_in_out_cubic(t):
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4 * t * t * t
    return 1 - math.pow(-2 * t + 2, 3) / 2


class TextRenderer:
    def __init__(self, config):
        self.config = config
        self.font = self._load_font(config.get("font_size", 42), config.get("font_family", "arial"))
        self.text_color = _parse_color(config.get("text_color", "#FFFFFF"))
        self.highlight_color = _parse_color(config.get("highlight_color", "#FFD700"))
        self.outline_color = _parse_color(config.get("outline_color", "#000000"))
        self.outline_w = config.get("outline_thickness", 3)
        self.bg_style = config.get("background_style", "none")
        self.bg_opacity = config.get("background_opacity", 0.6)
        self.shadow = config.get("text_shadow", "enable") == "enable"
        self.position = config.get("text_position", "bottom")
        self.alignment = config.get("text_alignment", "center")
        self.line_mode = config.get("line_display", "single_line")
        self.style = config.get("display_style", "subtitles")
        self._display_cache = None

    # ── Font loading ─────────────────────────────────────────────────

    def _load_font(self, size, family):
        path = self._find_font(family)
        if path:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()

    def _find_font(self, family):
        family = family.lower()
        fonts_dir = os.path.join(os.path.dirname(__file__), "fonts")

        # Check cached downloads
        cached = os.path.join(fonts_dir, f"{family}.ttf")
        if os.path.exists(cached):
            return cached

        # Check system fonts (Windows)
        if family in SYSTEM_FONTS:
            win_fonts = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
            for fname in SYSTEM_FONTS[family]:
                p = os.path.join(win_fonts, fname)
                if os.path.exists(p):
                    return p

        # Download if available
        if family in FONT_URLS:
            return self._download_font(family, fonts_dir)

        return None

    def _download_font(self, family, fonts_dir):
        os.makedirs(fonts_dir, exist_ok=True)
        dest = os.path.join(fonts_dir, f"{family}.ttf")
        url = FONT_URLS[family]
        try:
            print(f"[TextRenderer] Downloading font '{family}'...")
            with urllib.request.urlopen(url, timeout=5) as response, open(dest, "wb") as out:
                while True:
                    chunk = response.read(1024 * 64)
                    if not chunk:
                        break
                    out.write(chunk)
            print(f"[TextRenderer] ✅ Font '{family}' downloaded")
            return dest
        except Exception as e:
            try:
                if os.path.exists(dest) and os.path.getsize(dest) == 0:
                    os.remove(dest)
            except Exception:
                pass
            print(f"[TextRenderer] ❌ Font download failed: {e}")
            return None

    # ── Position helpers ─────────────────────────────────────────────

    def _get_y_base(self, h, text_h, n_lines=1):
        line_spacing = int(text_h * 0.3)
        total_h = text_h * n_lines + line_spacing * max(n_lines - 1, 0)
        margin = int(h * 0.05)
        if self.position == "top":
            return margin
        elif self.position == "upper_third":
            y = int(h * 0.20)
            return max(margin, min(y, h - total_h - margin))
        elif self.position == "center":
            return max(margin, (h - total_h) // 2)
        elif self.position == "lower_third":
            # Text sits around one third up from the bottom, slightly lower.
            y = int(h * 0.68)
            return max(margin, min(y, h - total_h - margin))
        else:  # bottom
            return max(margin, h - total_h - margin)

    def _get_x_start(self, w, text_w):
        pad = int(w * 0.05)  # 5% margin on each side
        if self.alignment == "left":
            return pad
        elif self.alignment == "right":
            return w - text_w - pad
        return (w - text_w) // 2

    # ── Word layout with wrapping ────────────────────────────────────

    def _measure_words(self, words):
        sizes = []
        for w in words:
            bbox = self.font.getbbox(w)
            sizes.append((bbox[2] - bbox[0], bbox[3] - bbox[1]))
        return sizes

    def _get_space_width(self):
        bbox = self.font.getbbox(" ")
        return max(bbox[2] - bbox[0], 4)

    def _word_positions(self, words, sizes, frame_w, y):
        """Calculate word positions with automatic wrapping when text exceeds frame width."""
        space = self._get_space_width()
        margin = int(frame_w * 0.05)
        max_w = frame_w - margin * 2  # Usable width

        # Split words into rows that fit within max_w
        rows = []  # Each row: list of (word_index, word, size)
        current_row = []
        current_row_w = 0

        for i, (sw, sh) in enumerate(sizes):
            word_w = sw + (space if current_row else 0)
            if current_row and current_row_w + word_w > max_w:
                # Start new row
                rows.append(current_row)
                current_row = [(i, words[i], sizes[i])]
                current_row_w = sw
            else:
                current_row.append((i, words[i], sizes[i]))
                current_row_w += word_w

        if current_row:
            rows.append(current_row)

        # Calculate text height for line spacing
        text_h = max(s[1] for s in sizes) if sizes else 30
        line_spacing = int(text_h * 0.3)

        # Build positions for all words across wrapped rows
        positions = [None] * len(words)
        for row_idx, row in enumerate(rows):
            row_w = sum(s[0] for _, _, s in row) + space * max(len(row) - 1, 0)
            x = self._get_x_start(frame_w, row_w)
            row_y = y + row_idx * (text_h + line_spacing)
            for word_idx, word, (sw, sh) in row:
                positions[word_idx] = (x, row_y)
                x += sw + space

        return positions

    def _get_wrapped_row_count(self, words, sizes, frame_w):
        """Count how many rows text will wrap into."""
        space = self._get_space_width()
        margin = int(frame_w * 0.05)
        max_w = frame_w - margin * 2
        rows = 1
        current_w = 0
        for i, (sw, sh) in enumerate(sizes):
            word_w = sw + (space if current_w > 0 else 0)
            if current_w > 0 and current_w + word_w > max_w:
                rows += 1
                current_w = sw
            else:
                current_w += word_w
        return rows

    # ── Core drawing helpers ─────────────────────────────────────────

    def _draw_text(self, draw, text, x, y, color, alpha=255):
        c = (color[0], color[1], color[2], min(color[3], alpha))
        ow = self.outline_w
        if self.shadow:
            sc = (0, 0, 0, min(128, alpha))
            draw.text((x+2, y+2), text, font=self.font, fill=sc)
        if ow > 0:
            oc = (self.outline_color[0], self.outline_color[1], self.outline_color[2], min(self.outline_color[3], alpha))
            for dx in range(-ow, ow+1):
                for dy in range(-ow, ow+1):
                    if dx*dx + dy*dy <= ow*ow and (dx != 0 or dy != 0):
                        draw.text((x+dx, y+dy), text, font=self.font, fill=oc)
        draw.text((x, y), text, font=self.font, fill=c)

    def _draw_background(self, overlay, draw, x, y, w, h):
        if self.bg_style == "none":
            return
        pad = 12
        alpha = int(self.bg_opacity * 255)
        bx1, by1 = x - pad, y - pad // 2
        bx2, by2 = x + w + pad, y + h + pad // 2
        if self.bg_style == "solid_bar":
            draw.rectangle([0, by1, overlay.width, by2], fill=(0, 0, 0, alpha))
        elif self.bg_style == "gradient_bar":
            for gy in range(by1, by2):
                t = (gy - by1) / max(by2 - by1, 1)
                a = int(alpha * (1.0 - abs(t - 0.5) * 1.5))
                draw.rectangle([0, gy, overlay.width, gy+1], fill=(0, 0, 0, max(0, a)))
        elif self.bg_style == "rounded_box":
            draw.rounded_rectangle([bx1, by1, bx2, by2], radius=12, fill=(0, 0, 0, alpha))
        elif self.bg_style == "blur_box":
            draw.rounded_rectangle([bx1, by1, bx2, by2], radius=8, fill=(0, 0, 0, alpha))
        elif self.bg_style == "shadow_only":
            for i in range(3):
                a = int(alpha * (0.3 - i * 0.08))
                draw.rounded_rectangle([bx1-i*2, by1-i, bx2+i*2, by2+i], radius=10, fill=(0, 0, 0, max(0, a)))

    # ── Preprocess lyrics for display mode ──────────────────────────

    def _prepare_display_lyrics(self, aligned_lyrics):
        """Transform lyrics structure based on line_display mode.

        word_by_word: each word becomes its own 'line'
        single_line: original lines (no change)
        two_lines / three_lines / full_verse: original lines (handled in styles)
        """
        if self._display_cache is not None:
            return self._display_cache

        if self.line_mode == "word_by_word":
            # Flatten every word into its own single-word line
            display = []
            for line in aligned_lyrics:
                for w in line["words"]:
                    display.append({
                        "text": w["word"],
                        "line_start": w["start"],
                        "line_end": w["end"],
                        "words": [w],
                    })
            self._display_cache = display
            return display

        # All other modes use original lines
        self._display_cache = aligned_lyrics
        return aligned_lyrics

    # ── Find current lyrics position (binary-search based) ───────────

    def _find_position(self, timestamp, lyrics):
        """Find current line and word using word-level timestamps."""
        if not lyrics:
            return None, None

        # Binary search for the line whose time range contains timestamp
        lo, hi = 0, len(lyrics) - 1
        current_line = None

        while lo <= hi:
            mid = (lo + hi) // 2
            line = lyrics[mid]

            if timestamp < line["line_start"]:
                hi = mid - 1
            elif timestamp > line["line_end"]:
                lo = mid + 1
            else:
                current_line = mid
                break

        if current_line is None:
            return None, None

        # Find current word within line (linear scan — lines are short)
        line = lyrics[current_line]
        current_word = None
        for wi, w in enumerate(line["words"]):
            if w["start"] <= timestamp <= w["end"]:
                current_word = wi
                break
            if w["start"] <= timestamp:
                current_word = wi

        return current_line, current_word

    def _visible_word_items(self, line, timestamp):
        """Words that are allowed onscreen: never show words before their start."""
        if timestamp < line["line_start"] or timestamp > line["line_end"]:
            return []
        return [
            (idx, word_info)
            for idx, word_info in enumerate(line["words"])
            if timestamp >= word_info["start"]
        ]

    # ── Main render entry point ──────────────────────────────────────

    def render_frame(self, frame_tensor, timestamp, aligned_lyrics, bpm=120.0):
        h, w = frame_tensor.shape[:2]

        # Transform lyrics based on line_display mode
        display_lyrics = self._prepare_display_lyrics(aligned_lyrics)

        line_idx, word_idx = self._find_position(timestamp, display_lyrics)
        if line_idx is None:
            return frame_tensor
        if not self._visible_word_items(display_lyrics[line_idx], timestamp):
            return frame_tensor

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        ctx = {
            "line_idx": line_idx, "word_idx": word_idx or 0,
            "timestamp": timestamp, "bpm": bpm,
            "lyrics": display_lyrics, "w": w, "h": h,
        }

        style_fn = getattr(self, f"_style_{self.style}", self._style_subtitles)
        style_fn(overlay, draw, ctx)

        frame_u8 = (
            frame_tensor.detach()
            .clamp(0, 1)
            .mul(255)
            .to(dtype=torch.uint8, device="cpu")
            .contiguous()
        )
        frame_img = Image.fromarray(frame_u8.numpy()).convert("RGBA")
        composited = Image.alpha_composite(frame_img, overlay)
        rgb = composited.convert("RGB")
        result_u8 = torch.from_numpy(np.array(rgb, dtype=np.uint8, copy=True))
        result = result_u8.to(device=frame_tensor.device, dtype=frame_tensor.dtype)
        result.div_(255.0)
        return result

    # ── Style 1: Karaoke ─────────────────────────────────────────────

    def _style_karaoke(self, overlay, draw, ctx):
        line = ctx["lyrics"][ctx["line_idx"]]
        visible_items = self._visible_word_items(line, ctx["timestamp"])
        words = [w["word"] for _, w in visible_items]
        sizes = self._measure_words(words)
        if not sizes:
            return
        text_h = max(s[1] for s in sizes)
        n_rows = self._get_wrapped_row_count(words, sizes, ctx["w"])
        y = self._get_y_base(ctx["h"], text_h, n_rows)
        positions = self._word_positions(words, sizes, ctx["w"], y)

        for (original_idx, w_info), word, pos in zip(visible_items, words, positions):
            if pos is None:
                continue
            if ctx["timestamp"] >= w_info["end"]:
                # Already sung — highlight color, slightly dimmed
                color = self.highlight_color
                alpha = 200
            elif ctx["timestamp"] >= w_info["start"]:
                # Currently singing — highlight, full brightness
                t = (ctx["timestamp"] - w_info["start"]) / max(w_info["end"] - w_info["start"], 0.01)
                color = tuple(int(self.text_color[j] + (self.highlight_color[j] - self.text_color[j]) * t) for j in range(3)) + (255,)
                alpha = 255
            else:
                # Not yet sung — dimmed
                color = self.text_color
                alpha = 100
            self._draw_text(draw, word, pos[0], pos[1], color, alpha)

    # ── Style 2: Subtitles ───────────────────────────────────────────

    def _style_subtitles(self, overlay, draw, ctx):
        line = ctx["lyrics"][ctx["line_idx"]]
        visible_items = self._visible_word_items(line, ctx["timestamp"])
        words = [w["word"] for _, w in visible_items]
        sizes = self._measure_words(words)
        if not sizes:
            return
        text_h = max(s[1] for s in sizes)
        n_rows = self._get_wrapped_row_count(words, sizes, ctx["w"])
        y = self._get_y_base(ctx["h"], text_h, n_rows)
        positions = self._word_positions(words, sizes, ctx["w"], y)

        fade_dur = 0.15
        line_alpha = 255
        t = ctx["timestamp"]
        if t - line["line_start"] < fade_dur:
            line_alpha = int(255 * (t - line["line_start"]) / fade_dur)
        elif line["line_end"] - t < fade_dur:
            line_alpha = int(255 * (line["line_end"] - t) / fade_dur)
        line_alpha = max(0, min(255, line_alpha))

        for (original_idx, w_info), word, pos in zip(visible_items, words, positions):
            if pos is None:
                continue
            if t >= w_info["end"]:
                # Already sung
                color = self.highlight_color
                alpha = int(line_alpha * 0.8)
            elif t >= w_info["start"]:
                # Currently singing — bright highlight
                color = self.highlight_color
                alpha = line_alpha
            else:
                # Not yet sung — dimmed
                color = self.text_color
                alpha = int(line_alpha * 0.4)
            self._draw_text(draw, word, pos[0], pos[1], color, alpha)

    # ── Style 3: Word Pop ────────────────────────────────────────────

    def _style_word_pop(self, overlay, draw, ctx):
        line = ctx["lyrics"][ctx["line_idx"]]
        words = [w["word"] for w in line["words"]]
        sizes = self._measure_words(words)
        if not sizes:
            return
        text_h = max(s[1] for s in sizes)
        n_rows = self._get_wrapped_row_count(words, sizes, ctx["w"])
        y = self._get_y_base(ctx["h"], text_h, n_rows)
        positions = self._word_positions(words, sizes, ctx["w"], y)

        beat_dur = 60.0 / max(ctx["bpm"], 60)

        for i, (word, pos) in enumerate(zip(words, positions)):
            if pos is None:
                continue
            w_info = line["words"][i]
            if ctx["timestamp"] < w_info["start"]:
                continue
            elapsed = ctx["timestamp"] - w_info["start"]
            pop_t = min(elapsed / (beat_dur * 0.3), 1.0)
            scale = 1.0 + 0.4 * (1.0 - _ease_out_elastic(pop_t))
            alpha = int(255 * min(pop_t * 3, 1.0))

            sw, sh = sizes[i]
            cx, cy = pos[0] + sw // 2, pos[1] + sh // 2
            dx = int(cx - cx * scale + (scale - 1) * sw * 0.5)
            dy = int(cy - cy * scale + (scale - 1) * sh * 0.5)

            color = self.highlight_color if i == ctx["word_idx"] else self.text_color
            self._draw_text(draw, word, pos[0] - int((scale-1)*sw*0.5), pos[1] - int((scale-1)*sh*0.5), color, alpha)

    # ── Style 4: Typewriter ──────────────────────────────────────────

    def _style_typewriter(self, overlay, draw, ctx):
        line = ctx["lyrics"][ctx["line_idx"]]
        visible_items = self._visible_word_items(line, ctx["timestamp"])
        visible_words = [w["word"] for _, w in visible_items]
        visible_sizes = self._measure_words(visible_words)
        if not visible_sizes:
            return

        text_h = max(s[1] for s in visible_sizes)
        n_rows = self._get_wrapped_row_count(visible_words, visible_sizes, ctx["w"])
        y = self._get_y_base(ctx["h"], text_h, n_rows)
        positions = self._word_positions(visible_words, visible_sizes, ctx["w"], y)

        for i, (word, pos) in enumerate(zip(visible_words, positions)):
            if pos is None:
                continue
            self._draw_text(draw, word, pos[0], pos[1], self.highlight_color)

        # Blinking cursor after last visible word
        if positions and positions[-1] is not None and int(ctx["timestamp"] * 4) % 2 == 0:
            last_pos = positions[-1]
            lw = visible_sizes[-1][0]
            cursor_x = last_pos[0] + lw + 2
            cursor_h = visible_sizes[-1][1]
            draw.rectangle([cursor_x, last_pos[1], cursor_x + 3, last_pos[1] + cursor_h],
                          fill=self.highlight_color)

    # ── Style 5: Word Wave ───────────────────────────────────────────

    def _style_word_wave(self, overlay, draw, ctx):
        line = ctx["lyrics"][ctx["line_idx"]]
        visible_items = self._visible_word_items(line, ctx["timestamp"])
        words = [w["word"] for _, w in visible_items]
        sizes = self._measure_words(words)
        if not sizes:
            return
        text_h = max(s[1] for s in sizes)
        n_rows = self._get_wrapped_row_count(words, sizes, ctx["w"])
        y_base = self._get_y_base(ctx["h"], text_h, n_rows)
        positions = self._word_positions(words, sizes, ctx["w"], y_base)

        beat_dur = 60.0 / max(ctx["bpm"], 60)

        for (original_idx, w_info), word, pos in zip(visible_items, words, positions):
            if pos is None:
                continue
            wave_phase = (ctx["timestamp"] - w_info["start"]) / beat_dur * math.pi * 2
            if ctx["timestamp"] >= w_info["start"] and ctx["timestamp"] <= w_info["end"] + 0.3:
                offset_y = int(-15 * abs(math.sin(wave_phase)))
                color = self.highlight_color
                alpha = 255
            elif ctx["timestamp"] >= w_info["end"]:
                dist = abs(original_idx - (ctx["word_idx"] or 0))
                delayed_phase = wave_phase - dist * 0.5
                offset_y = int(-8 * max(0, math.sin(delayed_phase)))
                color = self.highlight_color
                alpha = 180
            self._draw_text(draw, word, pos[0], pos[1] + offset_y, color, alpha)

    # ── Style 6: Glow Pulse ──────────────────────────────────────────

    def _style_glow_pulse(self, overlay, draw, ctx):
        line = ctx["lyrics"][ctx["line_idx"]]
        visible_items = self._visible_word_items(line, ctx["timestamp"])
        words = [w["word"] for _, w in visible_items]
        sizes = self._measure_words(words)
        if not sizes:
            return
        text_h = max(s[1] for s in sizes)
        n_rows = self._get_wrapped_row_count(words, sizes, ctx["w"])
        y = self._get_y_base(ctx["h"], text_h, n_rows)
        positions = self._word_positions(words, sizes, ctx["w"], y)

        beat_dur = 60.0 / max(ctx["bpm"], 60)

        # Glow layer
        glow_overlay = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_overlay)

        for (original_idx, w_info), word, pos in zip(visible_items, words, positions):
            if pos is None:
                continue
            is_active = w_info["start"] <= ctx["timestamp"] <= w_info["end"] + 0.2
            if is_active:
                pulse = 0.5 + 0.5 * math.sin(ctx["timestamp"] / beat_dur * math.pi * 4)
                glow_alpha = int(180 * pulse)
                gc = (self.highlight_color[0], self.highlight_color[1], self.highlight_color[2], glow_alpha)
                glow_draw.text((pos[0], pos[1]), word, font=self.font, fill=gc)
                color = self.highlight_color
            elif ctx["timestamp"] > w_info["end"]:
                color = (*self.text_color[:3], 200)
            self._draw_text(draw, word, pos[0], pos[1], color)

        blurred = glow_overlay.filter(ImageFilter.GaussianBlur(radius=8))
        combined = Image.alpha_composite(overlay, blurred)
        overlay.paste(combined, (0, 0), combined)

    # ── Style 7: Slide In ────────────────────────────────────────────

    def _style_slide_in(self, overlay, draw, ctx):
        line = ctx["lyrics"][ctx["line_idx"]]
        visible_items = self._visible_word_items(line, ctx["timestamp"])
        words = [w["word"] for _, w in visible_items]
        sizes = self._measure_words(words)
        if not sizes:
            return
        text_h = max(s[1] for s in sizes)
        n_rows = self._get_wrapped_row_count(words, sizes, ctx["w"])
        y = self._get_y_base(ctx["h"], text_h, n_rows)
        positions = self._word_positions(words, sizes, ctx["w"], y)

        slide_dur = 0.3
        elapsed = ctx["timestamp"] - line["line_start"]
        t = min(1.0, elapsed / slide_dur)
        ease_t = _ease_in_out_cubic(t)

        from_left = ctx["line_idx"] % 2 == 0
        offset = int((1 - ease_t) * (ctx["w"] + 50)) * (-1 if from_left else 1)

        for i, (word, pos) in enumerate(zip(words, positions)):
            if pos is None:
                continue
            x = pos[0] - offset if from_left else pos[0] + offset
            self._draw_text(draw, word, x, pos[1], self.text_color)

    # ── Style 8: Bounce Drop ─────────────────────────────────────────

    def _style_bounce_drop(self, overlay, draw, ctx):
        line = ctx["lyrics"][ctx["line_idx"]]
        words = [w["word"] for w in line["words"]]
        sizes = self._measure_words(words)
        if not sizes:
            return
        text_h = max(s[1] for s in sizes)
        n_rows = self._get_wrapped_row_count(words, sizes, ctx["w"])
        y_target = self._get_y_base(ctx["h"], text_h, n_rows)
        positions = self._word_positions(words, sizes, ctx["w"], y_target)

        beat_dur = 60.0 / max(ctx["bpm"], 60)
        drop_h = ctx["h"] * 0.3

        for i, (word, pos) in enumerate(zip(words, positions)):
            if pos is None:
                continue
            w_info = line["words"][i]
            if ctx["timestamp"] < w_info["start"]:
                continue
            elapsed = ctx["timestamp"] - w_info["start"]
            drop_t = min(elapsed / (beat_dur * 0.4), 1.0)
            bounce = _ease_out_bounce(drop_t)
            y_off = int(-drop_h * (1.0 - bounce))
            color = self.highlight_color if i == ctx["word_idx"] else self.text_color
            self._draw_text(draw, word, pos[0], pos[1] + y_off, color)

    # ── Style 9: Fade Flow ───────────────────────────────────────────

    def _style_fade_flow(self, overlay, draw, ctx):
        lyrics = ctx["lyrics"]
        li = ctx["line_idx"]
        text_h = self.font.getbbox("Ay")[3] - self.font.getbbox("Ay")[1]
        line = lyrics[li]
        visible_items = self._visible_word_items(line, ctx["timestamp"])
        words = [w["word"] for _, w in visible_items]
        sizes = self._measure_words(words)
        if not sizes:
            return
        n_rows = self._get_wrapped_row_count(words, sizes, ctx["w"])
        y = self._get_y_base(ctx["h"], text_h, n_rows)
        positions = self._word_positions(words, sizes, ctx["w"], y)
        for word, pos in zip(words, positions):
            if pos is None:
                continue
            self._draw_text(draw, word, pos[0], pos[1], self.highlight_color, 255)

    # ── Style 10: Neon Flash ─────────────────────────────────────────

    def _style_neon_flash(self, overlay, draw, ctx):
        line = ctx["lyrics"][ctx["line_idx"]]
        visible_items = self._visible_word_items(line, ctx["timestamp"])
        words = [w["word"] for _, w in visible_items]
        sizes = self._measure_words(words)
        if not sizes:
            return
        text_h = max(s[1] for s in sizes)
        n_rows = self._get_wrapped_row_count(words, sizes, ctx["w"])
        y = self._get_y_base(ctx["h"], text_h, n_rows)
        positions = self._word_positions(words, sizes, ctx["w"], y)

        # Neon glow layers
        for blur_pass in [12, 6]:
            glow = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
            gd = ImageDraw.Draw(glow)
            for word, pos in zip(words, positions):
                if pos is None:
                    continue
                ga = 60 if blur_pass == 12 else 100
                gc = (*self.highlight_color[:3], ga)
                gd.text((pos[0], pos[1]), word, font=self.font, fill=gc)
            blurred = glow.filter(ImageFilter.GaussianBlur(radius=blur_pass))
            combined = Image.alpha_composite(overlay, blurred)
            overlay.paste(combined, (0, 0), combined)
        # Recreate draw since overlay pixels changed
        draw = ImageDraw.Draw(overlay)

        for (original_idx, w_info), word, pos, size in zip(visible_items, words, positions, sizes):
            if pos is None:
                continue
            is_new = w_info["start"] <= ctx["timestamp"] <= w_info["start"] + 0.1
            if is_new:
                # Flash burst
                sw, sh = size
                flash = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
                fd = ImageDraw.Draw(flash)
                fc = (255, 255, 255, 200)
                fd.text((pos[0], pos[1]), word, font=self.font, fill=fc)
                flash = flash.filter(ImageFilter.GaussianBlur(radius=15))
                combined = Image.alpha_composite(overlay, flash)
                overlay.paste(combined, (0, 0), combined)
                color = (255, 255, 255, 255)
            else:
                color = self.highlight_color
            self._draw_text(draw, word, pos[0], pos[1], color)
