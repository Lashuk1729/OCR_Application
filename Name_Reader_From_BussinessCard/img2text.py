# Import packages
import os
import cv2
from matplotlib import pyplot as plt
import argparse
import pytesseract
import nltk
from nltk.corpus import stopwords
try:
    import re2 as re
except ImportError:
    import re

#Construct and Parse The Argument
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required = True, help = "Path to the image")

args = vars(parser.parse_args())

# Load an color image in grayscale
img = cv2.imread(args["image"],0)

# Load the image using PIL (Python Imaging Library), Apply OCR, and then delete the temporary file
text = pytesseract.image_to_string(img)

# Cleaning the text
text = re.sub(r"\d", "", text)
text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)
text = re.sub('\W+',' ', text)

# Name Entity Recognition for Indian Names
# define an empty list
names = []

# open file and read the content in a list
with open('names.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        name = line[:-1]

        # add item to the list
        names.append(name)

# Tokenize and Part-of-Speech for the extracted text
tokens = nltk.tokenize.word_tokenize(text)
pos = nltk.pos_tag(tokens)

# consider only NNP proper noun
proper_noun = [x for (x,y) in pos if y in ('NNP')]

# consider the names on the cards
name_card = []
for elem in proper_noun:
    if elem.lower() in names:
        name_card.append(elem)

# name_card will have all the Indian Names present in the card
# We can save this list into a text file to get the name in the card
name_card = [name for name in name_card if len(name) > 3]

# To highlight the Names on the card
# get all data from the image
data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

# make a copy of this image to draw in
image_copy = img.copy()

# get all occurences of the that word
for target_word in name_card:
    word_occurences = [ i for i, word in enumerate(data["text"]) if word == target_word ]
    for occ in word_occurences:
        # extract the width, height, top and left position for that detected word
        w = data["width"][occ]
        h = data["height"][occ]
        l = data["left"][occ]
        t = data["top"][occ]
        # define all the surrounding box points
        p1 = (l, t)
        p2 = (l + w, t)
        p3 = (l + w, t + h)
        p4 = (l, t + h)
        # draw the 4 lines (rectangular)
        image_copy = cv2.line(image_copy, p1, p2, color=(255, 0, 0), thickness=2)
        image_copy = cv2.line(image_copy, p2, p3, color=(255, 0, 0), thickness=2)
        image_copy = cv2.line(image_copy, p3, p4, color=(255, 0, 0), thickness=2)
        image_copy = cv2.line(image_copy, p4, p1, color=(255, 0, 0), thickness=2)

# Plotting and Saving the resultant image
#image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
#plt.imsave("words.png", cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
plt.imsave("words.png",image_copy, cmap="gray")
plt.imshow(image_copy, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.show()
