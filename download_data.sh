#!/bin/bash

set -e
	
mkdir -p data/raw

wget "https://storage.googleapis.com/kaggle-competitions-data/kaggle/9120/all.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1570044927&Signature=pMLu%2BjDPZZJh5ZZ%2Fw8SfzLpwSptw6tTKZMOAlDhlWvj3LhYUEljdyeyOOOWAaM5%2BrGW1mfT2%2FBUUG7T4HibIEyOHPISMf9ANLD%2Fe8wrdu%2Fwctl0jDxxXVvnXZzFUGXhP8oy5giAKpC5LwZj1Fi5PyFptOA2v9p3FskwNYDxN5P5GwvBgDvXyCSV2hBDeaPnWDkA14i7itkggT1ORh0rP%2Bn%2BGbfF1tdTjoy8lQK8alo2BqmNRrjp9KKvFTW84xdSCnr7YOjkRvmNeIOPhJPx6gLTLcJJj%2FIzosg5aYVUCaJmcd975%2F7i0BdKp2XkzM06CibIXYsjKIw5LznGmyB%2FRYA%3D%3D&response-content-disposition=attachment%3B+filename%3Dhome-credit-default-risk.zip" -O data/raw/home-credit-default-risk.zip

unzip data/raw/home-credit-default-risk.zip -d data/raw

rm data/raw/home-credit-default-risk.zip
