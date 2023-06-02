import env

from PIL import Image, ImageDraw, ImageFont
from pilmoji import Pilmoji
from pilmoji.source import GoogleEmojiSource

from matplotlib import font_manager

font = ImageFont.truetype(
    font_manager.findfont(
        font_manager.FontProperties(family="sans-serif", weight="bold")
    ),
    45,
    encoding="unic",
)

def imgify_obs(obs: env.Observation):
    img = Image.new("RGB", (320, 340), "black")
    with Pilmoji(img, source=GoogleEmojiSource) as pilmoji:
        pilmoji.text((0, 0), env.stringify_obs(obs), (0, 0, 0), font)
    return img
