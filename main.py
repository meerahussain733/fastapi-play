from fastapi import FastAPI
from pydantic import BaseModel

App = FastAPI()
class Item(BaseModel):
    text: str = None
    is_done: bool = False

items = []

@App.get("/")
def root():
    return {"Hello" : "Engineer"}
            
@App.post("/items")
def create_item(item: Item):
    items.append(item)
    return items

@App.get("/items/{item_id}")
def get_item(item_id: int) -> Item:
    item=items[item_id]
    return item

