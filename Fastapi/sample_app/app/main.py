from typing import Union, List
from fastapi import FastAPI, HTTPException
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pydantic import BaseModel


# uri = "mongodb+srv://radshamila:qXrJZjnAwaO0iwor@cluster0.w7se96h.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# uri = "mongodb://localhost:27017/"
uri = "mongodb://mongodb-container:27017/"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

movies = None

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    # database = client.get_database("sample_mflix")
    database = client.get_database("sample_database")
    movies = database.get_collection("sample_users")
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

app = FastAPI()

class User(BaseModel):
    name: str
    age: int
    address: str
    student: bool

class UpdateUser(BaseModel):
    name: Union[str, None] = None
    age: Union[int, None] = None
    address: Union[str, None] = None
    student: Union[bool, None] = None

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/users")
def insert_users(user: User):
    inserted = movies.insert_one(user.dict())
    return {
        "message": f"{user.name} added successfully.",
        "db_status": inserted.acknowledged
    }

@app.get("/users", response_model=List[dict])
def list_users():
    users = list(movies.find())
    for user in users:
        user['_id'] = str(user['_id'])  # Convert ObjectId to string for JSON serialization
    return users

@app.get("/users/{user_id}")
def get_user(user_id: str):
    try:
        user = movies.find_one({"_id": ObjectId(user_id)})
        if user:
            user['_id'] = str(user['_id'])
            return user
        raise HTTPException(status_code=404, detail="User not found")
    except:
        raise HTTPException(status_code=400, detail="Invalid user ID")

@app.put("/users/{user_id}")
def update_user(user_id: str, user_data: UpdateUser):
    try:
        update_dict = {k: v for k, v in user_data.dict().items() if v is not None}
        result = movies.update_one({"_id": ObjectId(user_id)}, {"$set": update_dict})
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="User not found")
        return {"message": "User updated successfully"}
    except:
        raise HTTPException(status_code=400, detail="Invalid user ID")

@app.delete("/users/{user_id}")
def delete_user(user_id: str):
    try:
        result = movies.delete_one({"_id": ObjectId(user_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="User not found")
        return {"message": "User deleted successfully"}
    except:
        raise HTTPException(status_code=400, detail="Invalid user ID")
