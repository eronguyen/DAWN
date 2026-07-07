SAVE_DIR=/mnt/localssd/calvin-dawn
hf download nero1342/CALVIN-DAWN --repo-type dataset --local-dir ${SAVE_DIR}

cd ${SAVE_DIR}
echo "Extracting calvin_opt.tar"
tar -xf calvin_opt.tar

# DROID
# uv pip install tensorflow tensorflow-datasets pillow
# cd /mnt/localssd/
# aws s3 cp s3://adobe-ero-s3-general/data/droid.zip 
# unzip -q droid.zip 
