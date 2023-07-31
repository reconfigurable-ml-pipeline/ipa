run_pipeline() {
    BASE_DIR=~/infernece-pipeline-joint-optimization

    python $BASE_DIR/generate_dirs.py
    python $BASE_DIR/prediction-modules/lstm-module/train.py
    # python $BASE_DIR/models-to-minio/model-saver-resnet.py
    # python $BASE_DIR/models-to-minio/model-saver-yolo.py
    # python $BASE_DIR/models-to-minio/model-saver-transformers.py
    bash $BASE_DIR/download_models.sh
    bash $BASE_DIR/data/download.sh
    # gsutil cp -r gs://ipa-results/results $BASE_DIR/data

    for pipelineName in "ppn1 ppn2"; do
        for modelName in "mn1 mn2"; do
            echo $pipelineName $modelName
        done
    done
}

# Call the function
run_pipeline
