from motor.motor_asyncio import AsyncIOMotorClient
from bson.objectid import ObjectId

MONGO_DETAILS = "mongodb://localhost:27017"  # 替换为你的 MongoDB 连接字符串

client = AsyncIOMotorClient(MONGO_DETAILS)

database = client.users_db

users_collection = database.get_collection("users_collection")


def user_helper(user) -> dict:
    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "email": user["email"],
        "full_name": user["full_name"],
        "hashed_password": user["hashed_password"],
        "disabled": user["disabled"],
    }
