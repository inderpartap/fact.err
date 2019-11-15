import os
from flask import Flask, render_template


app = Flask(__name__)

@app.route("/")
@app.route("/dashboard")
def index():
    return render_template("index.html", message="Dashboard");   

@app.route("/news")
def news():
    return render_template("news.html", message="News Feed");  

@app.route("/search")
def search():
    return render_template("search.html", message="Custom Search");   


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)