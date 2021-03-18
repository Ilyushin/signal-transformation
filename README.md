# Signal transformation

[![PyPI version](https://badge.fury.io/py/signal-transformation.svg)](https://badge.fury.io/py/signal-transformation)

Helps to transform signal data like speech into different representations using TensorFlow API.

```bash
pip3 install signal-transformation
```

Mantains following types of output formats of the voice:

- PCM
- STFT
- MAGNITUDE
- MFCC
- LOG MFCC

## Voice

1) Transform wav files to needed format and then save them to tf_records
    ```python
    import pandas as pd
    import tensorflow as tf
    from signal_transformation.voice.tf_transformation import wav_to_tf_records, SpecFormat
    
    
    metadata_path = '/path/to/parquet/file'
    output_dir = '/output/directory'
    
    metadata = pd.read_parquet(metadata_path)
    
    wav_to_tf_records(
        metadata=metadata,
        output_dir=output_dir,
        spec_format=SpecFormat.MAGNITUDE
    )
    
    ```
   
   Or use as script:
   ```commandline
   python3 wav_to_tf_records --metadata_path="./vox1_metadata.gzip" --spec_format=3 --sample_rate=16000 --num_mfcc=13 --spect_shape=[300, 200, 1] --num_shards=512 --output_dir="/dataset/tf_records"
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