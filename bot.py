import telegram
import onnxruntime
import numpy as np
import telegram.ext
from PIL import Image
from PIL import ImageOps
from secret import *

USAGE_TEXT = (
    "Send me some pictures of Simpsons. "
    "I'll try and guess their names :) "
    "You can use inline bots like @bing and @pic.")

def pad(im):
    w, h = im.size; m = np.max([w, h])
    hp, hpr = (m - w) // 2, (m - w) % 2
    vp, vpr = (m - h) // 2, (m - h) % 2
    return (hp + hpr, vp + vpr, hp, vp)

def norm(x):
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    return (x - mean) / std

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def load_labels(path):
    return np.array(open(path).read().splitlines())

def load_model(path):
    return onnxruntime.InferenceSession(path)

def predict(path, n):
    sz = (224,224)
    im = Image.open(path)
    im.thumbnail(sz, Image.ANTIALIAS)
    im = ImageOps.expand(im, pad(im))
    x = np.array(im) / 255.
    x = np.float32(norm(x))
    x = x.transpose(2,0,1)
    x = x.reshape((1,) + x.shape)
    y = model.run(None, {'x': x})
    probs = softmax(y[0][0])
    idx = np.argsort(-probs)[:n]
    return labels[idx], probs[idx]

def handle_text(update, context):
    update.message.reply_text(USAGE_TEXT)

def handle_photo(update, context):
    user = update.message.from_user
    chat_id = update.message.chat_id
    file_id = update.message.photo[-1]['file_id']
    context.bot.getFile(file_id).download('in.jpg')
    labels, probs = predict('in.jpg', 3)
    e = enumerate(zip(labels, probs))
    a = [f'{i+1}. {c}: {p:.8f}' for i, (c, p) in e]
    context.bot.send_message(chat_id, '\n'.join(a))

    if user['id'] != TG_BOT_OWNER_ID:
        with open('in.jpg', 'rb') as fd:
            msg = f"@{user['username']} {user['id']}"
            context.bot.send_photo(TG_BOT_OWNER_ID, fd, msg)

model = load_model('model.onnx')
labels = load_labels('labels.txt')

ft = telegram.ext.Filters.text
fp = telegram.ext.Filters.photo
h = telegram.ext.MessageHandler
u = telegram.ext.Updater(TG_BOT_TOKEN)
u.dispatcher.add_handler(h(ft, handle_text))
u.dispatcher.add_handler(h(fp, handle_photo))
u.start_polling(); u.idle()