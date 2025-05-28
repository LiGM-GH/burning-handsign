#!/usr/bin/sh

convert $1                     \
    -virtual-pixel White       \
    -region 300x300            \
    -implode .5                \
    $2
