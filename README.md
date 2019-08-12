# Signal transformation


Helps to transform signal data like speech into different representations using TensorFlow API.
```bash
pip3 install signal-transformation
```

1) Transform wav files to PCM and than save them to tf_records
```python
import tensorflow as tf
from signal_transformation import tf_transformation


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