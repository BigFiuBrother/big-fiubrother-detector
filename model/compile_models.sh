#!/usr/bin/env bash

mvNCCompile -s 12 -o ssd.graph -w ssd.caffemodel ssd.prototxt
mvNCCompile -s 12 -o ssd_longrange.graph -w ssd_longrange.caffemodel ssd_longrange.prototxt

rm *.npy