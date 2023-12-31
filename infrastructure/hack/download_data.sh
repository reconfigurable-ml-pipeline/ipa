function download_data(){
    # download experiment logs
    gsutil cp -rn gs://ipa-results-1/results.zip ~/ipa/data
    unzip ~/ipa/data/results.zip
    mv results ~/ipa/data
    rm ~/ipa/data/results.zip

    # download ml models
    gsutil cp -rn 'gs://ipa-models/myshareddir/torchhub' /mnt/myshareddir
    gsutil cp -rn 'gs://ipa-models/myshareddir/huggingface' /mnt/myshareddir

    # download lstm trained model
    gsutil cp -r gs://ipa-models/lstm ~/ipa/data
}

download_data
