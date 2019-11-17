import os
from flask import Flask, render_template
# from flask_pymongo import PyMongo
from pymongo import MongoClient

app = Flask(__name__)

# app.config["MONGO_URI"]
mongo = MongoClient("mongodb+srv://admin:admin@facterrcluster-v12sw.mongodb.net/test?retryWrites=true&w=majority")
# mongo = PyMongo(app)
db = mongo['news']
collection = db['newsData']

post = {"_id": 0, "title": "this is news title", "body": "body of news"}
# collection.insert_one(post)

result = collection.find({"title": "this is news title"})
for i in result:
	print (i)

@app.route("/")
@app.route("/dashboard")
def index():
	online_users = mongo.db.users.find({"online": True})
	return render_template("index.html", message="Dashboard", online_users=online_users);   

@app.route("/news")
def news():
	return render_template("news.html", message="News Feed");  

@app.route("/search")
def search():
	return render_template("search.html", message="Custom Search");   


if __name__ == "__main__":
	app.run(host='0.0.0.0', debug=True)