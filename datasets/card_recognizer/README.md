Custom dataset format in order to train the card recognizer.
Mainly there are subfolders of each OP series with one image as .jpg and the corresponding .json with additional card information, like the original image url.

Actually, only the .jpg and the name of the .jpg is important for training as each card ID is a class to learn for the classifier.