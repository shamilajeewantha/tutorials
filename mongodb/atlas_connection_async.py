import asyncio
from pymongo import AsyncMongoClient
from pymongo.server_api import ServerApi

async def main():
    # Replace the placeholder with your Atlas connection string
    uri = "mongodb+srv://radshamila:qXrJZjnAwaO0iwor@cluster0.w7se96h.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    # uri = "mongodb://localhost:27017/"

    # Create a MongoClient with a MongoClientOptions object to set the Stable API version
    client = AsyncMongoClient(uri, server_api=ServerApi(
        version='1', strict=True, deprecation_errors=True))

    try:
        # Send a ping to confirm a successful connection
        await client.admin.command({'ping': 1})
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        raise Exception("Unable to find the document due to the following error: ", e)


    try:
        database = client["sample_mflix"]
        movies = database["movies"]
        # Query for a movie that has the title 'Back to the Future'
        query = { "title": "Back to the Future" }
        movie = await movies.find_one(query)
        print(movie)

        inserted = await movies.insert_one({"name" : "Mongo's Burgers"})
        print(inserted)
        print(inserted.acknowledged)

    except Exception as e:
        raise Exception("Unable to find the document due to the following error: ", e)

# these two are wrong in the documentation!!
    try:
        database = client["sample_mflix"] 
        collection_list = await database.list_collections()
        async for c in collection_list:
            print(c)

        collection_list = await database.list_collection_names()
        for c in collection_list:
            print(c)
    except Exception as e:
        raise Exception("Unable to find the document due to the following error: ", e)


    finally:
        # Ensures that the client will close when you finish/error
        await client.close()

asyncio.run(main())

