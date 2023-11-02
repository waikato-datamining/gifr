# gifr
gradio interfaces for Deep Learning Docker images that use Redis for receiving
data to make predictions on.

https://www.data-mining.co.nz/docker-images/


## Installation

```bash
pip install git+https://github.com/waikato-datamining/gifr.git
```

## Interfaces

### Image classification

```
usage: gifr-imgcls [-h] [--redis_host HOST] [--redis_port PORT]
                   [--redis_db DB] [--model_channel_in CHANNEL]
                   [--model_channel_out CHANNEL] [--timeout SECONDS]
                   [--launch_browser] [--share_interface]
                   [--logging_level {DEBUG,INFO,WARN,ERROR,CRITICAL}]

Image classification interface. Allows the user to select an image and display
the probabilities per label that the model generated.

optional arguments:
  -h, --help            show this help message and exit
  --redis_host HOST     The host with the redis server. (default: localhost)
  --redis_port PORT     The port of the redis server. (default: 6379)
  --redis_db DB         The redis database to use. (default: 0)
  --model_channel_in CHANNEL
                        The channel to send the data to for making
                        predictions. (default: images)
  --model_channel_out CHANNEL
                        The channel to receive the predictions on. (default:
                        predictions)
  --timeout SECONDS     The number of seconds to wait for a prediction.
                        (default: 1.0)
  --launch_browser      Whether to automatically launch the interface in a new
                        tab of the default browser. (default: False)
  --share_interface     Whether to publicly share the interface at
                        https://XYZ.gradio.live/. (default: False)
  --logging_level {DEBUG,INFO,WARN,ERROR,CRITICAL}
                        The logging level to use (default: WARN)
```
