from model import Model
import os, argparse, asyncio

from telegram import Update
from telegram.ext import filters, Application
from telegram.ext import CommandHandler, MessageHandler

parser = argparse.ArgumentParser()
parser.add_argument('id', type=int, help='bot owner id')
parser.add_argument('token', type=str, help='bot token')

args = parser.parse_args()
model = Model('model.onnx', 'labels.txt')

async def handle_text(update, context):

    usage_text = (
        "Send me some pictures of Simpsons. "
        "I'll try and guess their names :) "
        "You can use inline bots like @bing and @pic.")

    await update.message.reply_text(usage_text)

async def handle_photo(update, context):

    loop = asyncio.get_running_loop()

    user = update.message.from_user
    photo = update.message.photo[-1]
    chat_id = update.message.chat_id

    file = await context.bot.get_file(photo)
    path = photo['file_unique_id'] + '.jpg'
    await file.download_to_drive(path)

    labels, probs = await loop.run_in_executor(None, model.predict, path)
    msg = '\n'.join([f'{c}: {p:.8f}' for c, p in zip(labels, probs)])
    await context.bot.send_message(chat_id, msg)

    if user['id'] != args.id:
        with open(path, 'rb') as fd:
            msg = f"@{user['username']} {user['id']}"
            await context.bot.send_photo(args.id, fd, msg)

    os.remove(path)

app = Application.builder().token(args.token).build()
app.add_handler(MessageHandler(filters.TEXT, handle_text))
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
app.run_polling(allowed_updates=Update.ALL_TYPES)