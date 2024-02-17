echo Creating the dataset path...

cd src/RIM/data

echo Downloading MaiCity dataset...
wget https://www.ipb.uni-bonn.de/html/projects/mai_city/mai_city.tar.gz

echo Extracting dataset...
tar -xvf mai_city.tar.gz

echo Downloading MaiCity ground truth point cloud generated from sequence 02 and the ground truth model ...
cd mai_city
wget -O gt_map_pc_mai.ply -c https://uni-bonn.sciebo.de/s/DAMWVCC1Kxkfkyz/download
wget https://connecthkuhk-my.sharepoint.com/:u:/g/personal/jianheng_connect_hku_hk/EYq53BQ2HGdKv18RFA3L3PYBA4HGGnQZcpmH0_jYyAhZlA\?e\=dzeQ4G\&download=1 -O gt.ply -c
cd ..

rm mai_city.tar.gz

cd ../../..