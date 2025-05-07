# Run with : 
# fastapi dev main.py

################################  First Steps  ################################

# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}



################################  Path Parameters  ################################

# http://127.0.0.1:8000/items/foo

# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/items/{item_id}")
# async def read_item(item_id):
#     return {"item_id": item_id}


# # this doesnt work
# # async def read_item(item_identity):
# #     return {"item_id": item_identity}

# # This function is not bound to the endpoint; only the top one.
# # The type of the path parameter item_id is string. 
# # But once passed it automatically gets converted to int
# async def read_item(item_id: int):
#     return {"item_id": item_id}



################################  Order matters  ################################

# http://127.0.0.1:8000/users/me

# from fastapi import FastAPI

# app = FastAPI()

# # if you reverse the order of the functions it'll print the string "me" as a user_id

# @app.get("/users/me")
# async def read_user_me():
#     return {"user_id": "the current user"}


# @app.get("/users/{user_id}")
# async def read_user(user_id: str):
#     return {"user_id": user_id}


# # The first one will always be used since the path matches first.
# # http://127.0.0.1:8000/users

# @app.get("/users")
# async def read_users():
#     return ["Rick", "Morty"]


# @app.get("/users")
# async def read_users2():
#     return ["Bean", "Elfo"]



################################  Predefined values  ################################

# http://127.0.0.1:8000/models/alexnet
# http://127.0.0.1:8000/models/abcde
# {
#   "detail": [
#     {
#       "type": "enum",
#       "loc": [
#         "path",
#         "model_name"
#       ],
#       "msg": "Input should be 'alexnet', 'resnet' or 'lenet'",
#       "input": "abcde",
#       "ctx": {
#         "expected": "'alexnet', 'resnet' or 'lenet'"
#       }
#     }
#   ]
# }

# from enum import Enum

# from fastapi import FastAPI


# class ModelName(str, Enum):
#     alexnet = "alexnet"
#     resnet = "resnet"
#     lenet = "lenet"


# app = FastAPI()


# @app.get("/models/{model_name}")
# async def get_model(model_name: ModelName):
#     if model_name is ModelName.alexnet:
#         return {"model_name": model_name, "message": "Deep Learning FTW!"}

#     if model_name.value == "lenet":
#         return {"model_name": model_name, "message": "LeCNN all the images"}

#     return {"model_name": model_name, "message": "Have some residuals"}



################################  Path parameters containing paths  ################################

# http://127.0.0.1:8000/files//home/johndoe/myfile.txt

# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/files/{file_path:path}")
# async def read_file(file_path: str):
#     return {"file_path": file_path}



################################  Query Parameters  ################################

# http://127.0.0.1:8000/items/?skip=0&limit=2

# from fastapi import FastAPI

# app = FastAPI()

# fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


# @app.get("/items/")
# async def read_item(skip: int = 0, limit: int = 10):
#     return fake_items_db[skip : skip + limit]



################################  Optional parameters  ################################

# http://127.0.0.1:8000/items/foo
# http://127.0.0.1:8000/items/foo?q=5

# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/items/{item_id}")
# async def read_item(item_id: str, q: str | None = None):
#     if q:
#         return {"item_id": item_id, "q": q}
#     return {"item_id": item_id}



################################  Query parameter type conversion  ################################

# http://127.0.0.1:8000/items/foo?short=on
# http://127.0.0.1:8000/items/foo?short=1
# http://127.0.0.1:8000/items/foo?short=true
# http://127.0.0.1:8000/items/foo?short=true
# http://127.0.0.1:8000/items/foo?http://127.0.0.1:8000/items/foo?short=trueshort=true&q=6
# http://127.0.0.1:8000/items/foo?http://127.0.0.1:8000/items/foo?short=trueshort=trueno&q=6

# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/items/{item_id}")
# async def read_item(item_id: str, q: str | None = None, short: bool = False):
#     item = {"item_id": item_id}
#     if q:
#         item.update({"q": q})
#     if not short:
#         item.update(
#             {"description": "This is an amazing item that has a long description"}
#         )
#     return item



################################  Multiple path and query parameters  ################################

# http://127john.0.0.1:8000/users/4/items/foo?short=true

# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/users/{user_id}/items/{item_id}")
# async def read_user_item(
#     user_id: int, item_id: str, q: str | None = None, short: bool = False
# ):
#     item = {"item_id": item_id, "owner_id": user_id}
#     if q:
#         item.update({"q": q})
#     if not short:
#         item.update(
#             {"description": "This is an amazing item that has a long description"}
#         )
#     return item




################################  Required query parameters  ################################

# http://127.0.0.1:8000/items/foo-item
# http://127.0.0.1:8000/items/foo-item?needy=65

from fastapi import FastAPI

app = FastAPI()


@app.get("/items/{item_id}")
async def read_user_item(item_id: str, needy: str):
    item = {"item_id": item_id, "needy": needy}
    return item