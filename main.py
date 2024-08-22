from inferencer import infere
from time import time

if __name__ == "__main__":
    start = time()
    card_id = infere("monkey.png")
    print(time() - start)
    print(card_id)