#!/usr/bin/sh

sizes="$(magick identify -ping -format "%w %h" $1)"
sizex="$(echo $sizes | cut -d ' ' -f 1)"
sizey="$(echo $sizes | cut -d ' ' -f 2)"
echo $sizex > "sizex.txt"
echo $sizey > "sizey.txt"

if [ $sizex -lt $sizey ]
then
  magick $1 -rotate 90 rotated.png
  echo "true" > "rotated.txt"
  yolo predict model=../../models/yolo.pt source="rotated.png" save_txt=true
else
  yolo predict model=../../models/yolo.pt source=$1 save_txt=true
fi

sed runs/detect/predict/labels/image.txt \
  -e 's/\([[:digit:]]\+\)[[:space:]]\+\([[:digit:].]\+\)/\2 \1/' \
  | sort \
  > sorted.txt

cat sorted.txt \
  | sed -e 's/[[:digit:].]\+[[:space:]]\+\([[:digit:]]\+\).*/\1/' \
  | tr -d '\r\n' \
  > result.txt
