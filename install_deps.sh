#!/bin/bash

# Set-up dependencies
mkdir ${HOME}/Softwares
cd ${HOME}/Softwares

# Get ttor
echo '>>> Cloning TTor repo'
git clone https://github.com/Abeynaya/tasktorrent.git

# Get eigen 
echo '>>> Downlading Eigen-3.3.9'
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
tar -xvf eigen-3.3.9.tar.gz
rm eigen-3.3.9.tar.gz
mv eigen-3.3.9/ eigen

# Get metis
echo '>>> Downlading metis-5.1.0'
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
tar -xvf metis-5.1.0.tar.gz
rm metis-5.1.0.tar.gz
mv metis-5.1.0/ metis
cd metis
make config
make -j 2