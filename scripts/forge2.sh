#!/usr/bin/sh

convert $1                     \
    -virtual-pixel White       \
    -wave 20x600 \
    $2
