from fastapi import FastAPI, Form, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, constr
from typing import Optional
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
from motor.motor_asyncio import AsyncIOMotorClient
from bson.objectid import ObjectId

app = FastAPI()

# 加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2PasswordBearer 是 OAuth2 的一种形式
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Jinja2模板
templates = Jinja2Templates(directory="templates")

# MongoDB连接
MONGO_DETAILS = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_DETAILS)
database = client.users_db
users_collection = database.get_collection("users_collection")


def get_password_hash(password):
	return pwd_context.hash(password)


async def verify_password(plain_password, hashed_password):
	return pwd_context.verify(plain_password, hashed_password)


async def get_user(username: str):
	user = await users_collection.find_one({"username": username})
	if user:
		return user


async def authenticate_user(username: str, password: str):
	user = await get_user(username)
	if not user:
		return False
	if not await verify_password(password, user["hashed_password"]):
		return False
	return user


class UserRegisterForm(BaseModel):
	username: constr(min_length=1)
	password: constr(min_length=8)
	email: EmailStr
	full_name: constr(min_length=1)


class UserLoginForm(BaseModel):
	username: constr(min_length=1)
	password: constr(min_length=8)


@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
	user = await authenticate_user(form_data.username, form_data.password)
	if not user:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Incorrect username or password",
			headers={"WWW-Authenticate": "Bearer"},
		)
	return {"access_token": user["username"], "token_type": "bearer"}


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
	return templates.TemplateResponse("login.html", {"request": request})


@app.get("/register", response_class=HTMLResponse)
async def register_form(request: Request):
	return templates.TemplateResponse("register.html", {"request": request})


@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
	form_data = UserLoginForm(username=username, password=password)
	user = await authenticate_user(form_data.username, form_data.password)
	if not user:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect username or password")
	return {"message": "Login successful"}


@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...), email: str = Form(...),
				   full_name: str = Form(...)):
	form_data = UserRegisterForm(username=username, password=password, email=email, full_name=full_name)
	existing_user = await get_user(form_data.username)
	if existing_user:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")

	hashed_password = get_password_hash(form_data.password)
	user = {
		"username": form_data.username,
		"full_name": form_data.full_name,
		"email": form_data.email,
		"hashed_password": hashed_password,
		"disabled": False,
	}
	await users_collection.insert_one(user)
	return {"message": "User registered successfully"}


if __name__ == "__main__":
	import uvicorn

	uvicorn.run(app, host="127.0.0.1", port=8000)
