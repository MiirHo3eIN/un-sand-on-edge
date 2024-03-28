# un-sand-on-edge
This repository contains different applications of UN-Supervised ANomaly Detection for AI-driven methods deployed ON resource-constrained devices at EDGE for the SHM application. 

The current vision of the project is shown below:

![Pipelines](./docs/pipeline.png)

# Data 

The dataset of this work consists of normal case and 9 levels of anomalies, configured by adding masses over the structure. 
The following table shows how these anomalies are applied to the top floor of the structure where sensors are installed: 

| Anomaly Level | Mass (kg) |
|---------------|-----------|
| Level 1       | 1.5        |
| Level 2       | 2.8        |
| Level 3       | 3.6        |
| Level 4       | 4.6        |
| Level 5       | 5.6        |
| Level 6       | 6.6        |
| Level 7       | 7.6        |
| Level 8       | 8.6        |
| Level 9       | 9.6        |

The data is accessible at [TBD]. 

The data folder MUST be kept locally on your machine; the person pushing a folder titled "data" would be responsible for the mess with the repository. 
We will hunt you for this push. 

We highly recommend saving the data somewhere in your local machine and creating a symlink to the top directory of this repository. 
This is usually done with the following command:  

``` bash
ln -s /path/to/the/dataset data
```

# Python Virtual Environment

## Python Requirement 
In order to run all the scripts, you should have downloaded and installed python 3.11 on your machine. You can do it by following the following tutorial: 

**Ubuntu**: 
[Python 3.11 Installation](https://www.linuxcapable.com/how-to-install-python-3-11-on-ubuntu-linux/)

**Windows**: 
[Python 3.11 Installation](https://www.python.org/downloads/release/python-3110/)

## Installation 
To set up the Python virtual environment, use the following commands in the folder of this repository:
```bash 
python3.11 -m venv myenv 
cd myenv/bin/
source activate
``` 

*Note:* You can chnge ```myenv``` with your desired environment name. 


