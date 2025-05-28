#!/usr/bin/sh

convert $1                  \
    -virtual-pixel White    \
    -distort Arc 10          \
    $2
