convert $1 -threshold 80% -define connected-components:verbose=true -define connected-components:area-threshold=5 -define connected-components:mean-color=true -connected-components 8 $2
mogrify -colorspace gray $2 $2
convert $2 -negate $2
mogrify -resize '220!x155!' $2 $2
convert -alpha deactivate $2 $2
