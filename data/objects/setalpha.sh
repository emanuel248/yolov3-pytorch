#!/bin/bash
for f in */*.png;do convert $f -alpha on $f;done
