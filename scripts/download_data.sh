#!/bin/bash

download_dir='./data'

cd $download_dir
wget https://www.dropbox.com/s/72ezcewo8ewvp9y/abc_2.5k.tar.gz      #approx 400MB in size
tar -xf abc_2.5k.tar.gz && rm -rf abc_2.5k.tar.gz 
echo "download of abc data complete"
