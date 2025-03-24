#!/bin/bash
# from https://stackoverflow.com/questions/151677/tool-for-adding-license-headers-to-source-files

for f in $(find . -name '*.py');
do
  if ! grep -q Copyright $f
  then
    cat copyright_header.txt $f >$f.new && mv $f.new $f
  fi
done
