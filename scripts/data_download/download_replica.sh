echo Creating the dataset path...

cd src/RIM/data

echo Downloading Replica
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip

echo Extracting dataset...
unzip Replica.zip

rm Replica.zip

cd ../../..

