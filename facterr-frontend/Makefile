# -*- coding: utf-8 -*-
# @Author: Inderpartap Cheema
# @Date:   2019-11-19
# @Last Modified by:   Inderpartap Cheema
# @GPLv3 License


clean:
	-find . -name '*.pyc' -delete
	-find . -name '__pycache__' -delete

run-background: clean
	# when running in CI server
	nohup gunicorn --pythonpath app:app &

run: clean
	# to be used when testing locally
	gunicorn --pythonpath app:app

test-api: clean run-background
	nosetests

deploy: clean
	# deploys app to heroku as well pushes the latest commits to the github
	# remote
	git push -u origin master
	git push -u heroku master

force-deploy: clean
	## DONT DO THIS! EVEN IF THE WORLD COMES TO AN END!
	git push -u origin master --force
	git push -u heroku master --force

.PHONY: help
help:
	@echo "\nPlease call with one of these targets:\n"
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F:\
        '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}'\
        | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$' | xargs | tr ' ' '\n' | awk\
        '{print "    - "$$0}'
	@echo "\n"
