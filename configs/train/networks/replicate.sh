#!/bin/bash
sed -e "s/EfficientNetB0/$1/g" -e "s/224/$2/g" EfficientNetB0.json > $1.json