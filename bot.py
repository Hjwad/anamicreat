# Import the required libraries
import telebot # Telegram bot library
import requests # HTTP library
import os # Operating system library
import torch # PyTorch library
import torchvision # PyTorch vision library
import numpy as np # Numpy library
import PIL # Python image library

# Define the bot token and create a bot object
TOKEN = "6174792297:AAGdM8JBr9qoZF0H_QOI3ATqKNDX_a3rlVc"
bot = telebot.TeleBot(TOKEN)

# Define the model URL and download the model file
MODEL_URL = "https://github.com/SerialLain3170/Anime-Semantic-Segmentation/releases/download/v1.0/diffusion.pt"
MODEL_FILE = "diffusion.pt"
if not os.path.exists(MODEL_FILE):
  r = requests.get(MODEL_URL, allow_redirects=True)
  open(MODEL_FILE, 'wb').write(r.content)

# Load the model and set it to evaluation mode
model = torch.jit.load(MODEL_FILE).cuda()
model.eval()

# Define a function to generate an image from a text query
def generate_image(query):
  # Encode the query into a tensor of shape (1, 128)
  query_tensor = model.encode_query([query]).cuda()
  # Generate an image from the query tensor using diffusion sampling
  image_tensor = model.sample(query_tensor)
  # Convert the image tensor to a PIL image object
  image = torchvision.transforms.ToPILImage()(image_tensor[0].cpu())
  # Return the image object
  return image

# Define a handler function for /start command
@bot.message_handler(commands=['start'])
def start(message):
  # Send a welcome message to the user
  bot.send_message(message.chat.id, "Hello, I am a diffusion anime telegram bot. I can generate anime images from text queries. To use me, just send me a text query and I will try to create an image that matches it.")

# Define a handler function for text messages
@bot.message_handler(func=lambda message: True)
def text(message):
  # Get the text query from the message
  query = message.text
  # Send a message to the user that the bot is working on the query
  bot.send_message(message.chat.id, "Generating an image for: " + query)
  # Try to generate an image from the query
  try:
    image = generate_image(query)
    # Save the image to a temporary file
    image.save("temp.jpg")
    # Send the image file to the user
    bot.send_photo(message.chat.id, open("temp.jpg", 'rb'))
    # Delete the temporary file
    os.remove("temp.jpg")
  except Exception as e:
    # If there is an error, send a message to the user with the error message
    bot.send_message(message.chat.id, "Sorry, something went wrong. Please try again later. Error: " + str(e))

# Start polling for updates from Telegram
bot.polling()
