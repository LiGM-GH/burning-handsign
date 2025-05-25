convert $1 \
    -virtual-pixel White \
    -define shepards:power=3.0 \
    -distort Shepards '150,200 120,150 470,300 450,220' \
    -distort Shepards '155,265 240,245 280,170 310,250' \
    -distort Shepards '280,330 260,350 350,300 370,320' \
    $2

