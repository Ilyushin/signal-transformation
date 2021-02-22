# Signal transformation


Helps to transform signal data like speech into different representations using TensorFlow API.
```bash
pip3 install signal-transformation
```

## Voice
1) Transform wav files to PCM and than save them to tf_records
    ```python
    import tensorflow as tf
    from signal_transformation.voice import tf_transformation
    
    
    source_path = '/path/to/wav/files'
    result_path = '/output/directory'
    
    
    with tf.compat.v1.Session() as sess:
        tf_transformation.wav_to_tf_records(
            sess=sess,
            audio_path=source_path,
            out_path=result_path
        )
    
    ```

2) Create overlapping signals dataset
    ```python
    from signal_transformation import helpers
    
    
    source_path = '/path/to/wav/files'
    result_path = '/output/directory'
    
    result = helpers.prepare_data(
        input_folder,
        output_folder,
        size=100,
        speakers_number=2,
        overlapping=True
    )
    
    ```
   
## Images
1) Imagenet dataset to TensorFlow records
```commandline
imagenet_to_tf_records --mode="grayscale" --train_data_dirs "/datasets/images/imagenet/train_1/raw /datasets/images/imagenet/train_2/raw" --validation_data_dir /datasets/images/imagenet/validation --validation_labels_file /datasets/images/imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt --output_dir /datasets/images/imagenet/tf_records 
```