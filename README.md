<h2>fact.err</h2>

[![Requirements Status](https://requires.io/github/inderpartap/facterr-frontend/requirements.svg?branch=master)](https://requires.io/github/inderpartap/facterr-frontend/requirements/?branch=master)
<br>
[![Build Status](https://travis-ci.com/inderpartap/facterr-frontend.svg?branch=master)](https://travis-ci.com/inderpartap/facterr-frontend)

> Detecting fake news using **Neural Networks**

**:arrow_forward: You can try it out here at [https://facterr.herokuapp.com/](https://facterr.herokuapp.com/)**

***

## Table of contents

- [Contributing](#testing)
    - [Installing it locally](#installing-it-locally)
    - [Running it](#running-it)
    - [Contributers](#contributers)
- [Roadmap](#roadmap)

***


#### Installing it locally

```bash
$ virtualenv env              # Create virtual environment
$ source env/bin/activate     # Change default python to virtual one
(env)$ git clone https://github.com/inderpartap/fact.err.git
(env)$ cd fact.err
(env)$ git submodule update --init --recursive
(env)$ pip install -r requirements.txt
```

#### Running it

```sh
$ python app.py
```


#### Contributers

- [Inderpartap Cheema (inderpartap)](https://github.com/inderpartap) : **App Creation, Visualizations and DataBase Management**
- [Najeeb Qazi (najq)](https://github.com/najq) : **Model Creation with Deep Learning models and Distributed Processing, Visualizations**
- [Ishaan Sahay (VWJF)](https://github.com/VWJF): **AWS Management, ETL and Data Collection**
- [Sachin (sachwithgithub)](https://github.com/sachwithgithub): **AWS Management, EMR Cluster Configuration and Data Collection**

***

## Roadmap
[:arrow_up: Back to top](#table-of-contents)

- [x] Creating data warehouse
- [x] Cleaning the data
- [x] Training the data using Neural Networks
- [x] Deploying to heroku
- [ ] Creating a REST API
- [ ] Improving the UI
- [ ] Writing tests
- [ ] Simple API authentication
