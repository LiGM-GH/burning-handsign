#!/usr/bin/sh

convert $1                     \
    -virtual-pixel White       \
    -swirl 20                  \
    $2
