from app.inferencer import infere
from time import time
from app.util import PROJECT_ROOT

if __name__ == "__main__":
    start = time()
    card_id = infere(PROJECT_ROOT / 'resources' / 'test_images' / 'monkey.png')
    print(time() - start)
    print(card_id)