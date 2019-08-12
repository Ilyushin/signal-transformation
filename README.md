# Signal transformation


Helps to transform signal data like speech into different representations using TensorFlow API.
```
pip3 install signal-transformation
```

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