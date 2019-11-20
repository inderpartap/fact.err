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
			'page=1&'
			'apiKey=944fd308bb7a49798093550409b3c2b9')
	response = requests.get(url)
	response_dict =response.json()
	
	data_dict = {}
	article_dict = response_dict['articles'][0]
	data_dict['title'] = article_dict['title']
	data_dict['description'] = article_dict['description']
	data_dict['url'] = article_dict['url']
	data_dict['publishedAt'] = article_dict['publishedAt']
	print (data_dict)
	query = {'title':'data_dict["title"]'}
	collection.update_one(query, data_dict, upsert=True);
	# collection.insert_one(data_dict) 

schedule.add_interval_job(fetch_real_news, minutes=1)

@app.route("/")
@app.route("/dashboard")
def index():
	# online_users = mongo.db.users.find({"online": True})
	return render_template("index.html", message="Dashboard");   

@app.route("/news")
def news():
	results = collection.find({})
	return render_template("news.html", message="News Feed", news_list = results);  

@app.route("/search")
def search():
	return render_template("search.html", message="Custom Search");   


if __name__ == "__main__":
	app.run(host='0.0.0.0', debug=True)