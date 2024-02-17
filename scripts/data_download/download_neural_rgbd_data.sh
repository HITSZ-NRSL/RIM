echo Creating the dataset path...

cd src/RIM/data

echo Downloading Neural RGBD dataset...
wget http://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip

echo Extracting dataset...
unzip neural_rgbd_data.zip -d neural_rgbd_data

rm neural_rgbd_data.zip

# From https://github.com/dazinovic/neural-rgbd-surface-reconstruction?tab=readme-ov-file#dataset
cd neural_rgbd_data
wget https://kaldir.vc.in.tum.de/neural_rgbd/meshes.zip
unzip meshes.zip
rm meshes.zip

cd ../../../..