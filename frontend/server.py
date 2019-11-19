import os
from flask import Flask, render_template
from pymongo import MongoClient
from apscheduler.scheduler import Scheduler
import json
import requests

app = Flask(__name__)

mongo = MongoClient("mongodb+srv://admin:admin@facterrcluster-v12sw.mongodb.net/test?retryWrites=true&w=majority")
db = mongo['news']
collection = db['newsData']

schedule = Scheduler() # Scheduler object
schedule.start()


def fetch_real_news():
	url = ('https://newsapi.org/v2/top-headlines?'
			'country=ca&'
			'language=en&'
			'category=general&'
			'pageSize=1&'
			'apiKey=944fd308bb7a49798093550409b3c2b9')
	response = requests.get(url)
	response_dict =response.json()
	
	data_dict = {}
	data_dict['title'] = dict['articles'][0]['title']
	data_dict['description'] = dict['articles'][0]['content']
	data_dict['url'] = dict['articles'][0]['url']
	data_dict['publishedAt'] = dict['articles'][0]['publishedAt']
	collection.insert_one(data_dict) 

schedule.add_interval_job(fetch_real_news, minutes=1)

# post = {"_id": 0, "title": "this is news title", "body": "body of news"}
# collection.insert_one(post)

# result = collection.find({"title": "this is news title"})
# for i in result:
# 	print (i)

@app.route("/")
@app.route("/dashboard")
def index():
	# online_users = mongo.db.users.find({"online": True})
	return render_template("index.html", message="Dashboard");   

@app.route("/news")
def news():
	return render_template("news.html", message="News Feed");  

@app.route("/search")
def search():
	return render_template("search.html", message="Custom Search");   


if __name__ == "__main__":
	app.run(host='0.0.0.0', debug=True)