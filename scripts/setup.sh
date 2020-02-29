cd .. || exit
cd ./arc23 || exit
git submodule update --init --recursive
git checkout v0.2.0 # change this to update dependency version referenced
