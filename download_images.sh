#!/bin/bash

if [ $# -ne 1 ]; then
    echo Usage: $0 file_with_url-class
    exit 1
fi
file=$1

# Yes images
dir='OpenI/Yes/'
for url in `awk '$2==1 {print $1}' $file`; do
    name=`echo $url|awk -F'/' '{print $NF}'`
    wget $url -O ${dir}$name 2> /dev/null
done

# No images
dir='OpenI/No/'
for url in `awk '$2==0 {print $1}' $file`; do
    name=`echo $url|awk -F'/' '{print $NF}'`
    wget $url -O ${dir}$name 2> /dev/null
done
