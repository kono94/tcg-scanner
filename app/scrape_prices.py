import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import Card, DATABASE_URL

def scrape_onepiece_prices():
    url = 'https://api.dotgg.gg/cgfw/getcards?game=onepiece&mode=indexed'
    r = requests.get(url)
    data = r.json()
    names = data["names"]
    idx_id = names.index("id")
    idx_name = names.index("name")
    idx_price = names.index("foilPrice")
    idx_slug = names.index("slug")
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine)
    db = Session()
    for row in data["data"]:
        card_id = row[idx_id]
        name = row[idx_name]
        price = float(row[idx_price]) if row[idx_price] else 0.0
        card = db.query(Card).filter(Card.card_id == card_id).first()
        image_url = f"https://static.dotgg.gg/onepiece/card/{card_id}.webp"
        print(image_url)
        if card:
            card.price = price
            card.name = name
            card.image_url = image_url
        else:
            db.add(Card(card_id=card_id, name=name, price=price, image_url=image_url))
    db.commit()
    db.close()
    print('Prices updated.')

if __name__ == '__main__':
    scrape_onepiece_prices()
