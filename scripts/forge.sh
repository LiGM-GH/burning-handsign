#!/usr/bin/sh

convert $1 \
    -virtual-pixel White \
    -distort Barrel "0.0 0.0 -0.595 1.1" \
    $2

    # -wave 10x200 \
    # -swirl 50 \
    # -rotate -10 \
    # -implode -0.5 \

    # -define shepards:power=0.8 \
    # -distort Shepards '450,0 0,420 0,500 580,0' \
    # -distort Shepards '300,300 300,250 300,300 250,300' \
    # -distort Shepards '300,300 350,300 300,300 300,350' \
