#!/usr/bin/env sh
# This scripts downloads the chair data and unzips it.

echo "Downloading..."

wget --no-check-certificate http://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar 

echo "Unzipping..."

tar -xvf rendered_chairs.tar
# Creation is split out because leveldb sometimes causes segfault
# and needs to be re-created.

echo "Done."
