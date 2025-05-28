#!/usr/bin/sh

convert $1 \
    -virtual-pixel White \
    -distort Barrel "0.0 0.0 -0.125 1.1" \
    -distort Arc 10         \
    $2

