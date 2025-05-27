#!/usr/bin/sh

# convert $1 -threshold 40% -define connected-components:verbose=true -define connected-components:area-threshold=5 -define connected-components:mean-color=true -connected-components 8 $2
mogrify -colorspace gray $1 $2
mogrify -resize '600!x600!' $2 $2
convert -alpha deactivate $2 $2
