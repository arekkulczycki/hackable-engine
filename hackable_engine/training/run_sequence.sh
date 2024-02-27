#!/bin/bash

for value in {0..7}
do
    version=$(($value + $1));
    echo $version;
done