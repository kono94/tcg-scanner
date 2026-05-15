Custom dataset format in order to train the card recognizer.
Mainly there are subfolders of each OP series with one image as .jpg and the corresponding .json with additional card information, like the original image url.

Actually, only the .jpg and the name of the .jpg is important for training as each card ID is a class to learn for the classifier.

The script-based pipeline now trains on all scraped images under `datasets/card_recognizer/cards/`.
Add real phone-camera validation examples under:

```text
datasets/card_recognizer/manual_eval/<card-id>/
```

For example:

```text
datasets/card_recognizer/manual_eval/OP01-001/IMG_1001.jpg
```

Validation uses this manual folder.

Multiple images for one class are supported. Keep the scraped flat layout:

```text
datasets/card_recognizer/cards/OP01/OP01-001.jpg
```

and add extra samples with a double-underscore suffix:

```text
datasets/card_recognizer/cards/OP01/OP01-001__phone-front.jpg
```

or use a class folder:

```text
datasets/card_recognizer/cards/OP01/OP01-001/phone-front.jpg
```

Only `__` means "same class, extra sample". A card like `OP01-001_p1` is treated as its own class.
