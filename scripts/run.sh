#!/bin/bash

exec python3 download.py
exec python3 prepare.py
exec python3 train.py
exec python3 test.py 
