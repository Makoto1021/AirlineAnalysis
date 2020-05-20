import re
import emoji
from emoji import UNICODE_EMOJI

def demojify(text):
    new_text = re.sub(emoji.get_emoji_regexp(), r" ", text)
    return new_text
