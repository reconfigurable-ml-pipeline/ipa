gsutil cp -rn gs://ipa-results-1/results.zip ~/ipa/data
unzip ~/ipa/data/results.zip
mv results ~/ipa/data
rm ~/ipa/data/results.zip
gsutil cp -rn gs://ipa-models/myshareddir/* /mnt/myshareddir