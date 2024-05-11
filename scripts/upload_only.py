from pymongo import MongoClient
from datetime import datetime

def upload_data(is_person_wearing_card: bool, image_path: str):
    # Connect to MongoDB
    client = MongoClient(
        'mongodb+srv://username:Singhabhii@cluster0.vgchusl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
    )
    db = client['IdAnalyzer']
    collection_name = 'IdAnalyzer'

    date_obj = datetime.now()
    isCard = is_person_wearing_card
    isOnlyPerson = not is_person_wearing_card
    owner_id = ""

    # Read image file into binary data
    with open(image_path, 'rb') as f:
        image_data = f.read()

    document = {
        "date": date_obj,
        "isCard": isCard,
        "isPersonWithoutId": isOnlyPerson,
        "owner_id": owner_id,
        "imageData": image_data
    }

    result = db[collection_name].insert_one(document)
    print("DOCUMENT INSERTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

wearing=False
path="D:\OTHER THINGS\Gmail and Youtube Icons\Ss (1).png"

upload_data(wearing,path)