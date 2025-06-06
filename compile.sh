set -ex

xargs apt-get -y install < apt.txt

pip install -r requirements.txt
pip install -e .


cd lmapf_lib/MAPFCompetition2023
# git submodule init
# git submodule update
./compile.sh

# cd ../..
# ls

# cd lmapf_lib/Guided-PIBT
# git submodule init
# git submodule update
# ./compile.sh