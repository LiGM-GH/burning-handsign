#!/usr/bin/sh

convert $1 \
    -virtual-pixel White \
    -define shepards:power=2.0 \
    -distort Shepards '300,300 300,250 300,300 250,300' \
    -distort Shepards '300,300 350,300 300,300 300,350' \
    $2

    # -define shepards:power=0.8 \
    # -distort Shepards '450,0 0,420 0,500 580,0' \
