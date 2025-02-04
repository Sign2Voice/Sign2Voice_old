# TO DOs

- [x] check whether is running on G2T python==3.9.21 (if used in final pipe) (optional)
- [x] baseline model - run with example glosses ONGOING
  - [x] add files from _my_ssd_mobnet_ to _sample_model_ folder
- [ ] **PHOENIX weather model**
  - [ ] create labeled images of new glosses
  - [x] delete my_ssd_mobnet folder 
  - [ ] run updated code - label list, train test split, label map, rf record files, config (change no. of classes)
- [ ]**upload sl_rtod folder & update used_requirements.txt**
- [ ] **streamlit.app - create code to run sl_rtod with webacm**
  - setup _streamlit_sl_rtod.py_ file
- [ ] _transform data to increase training data size and re-train the model (optional)_

# Credits

The repo is largely based on Nicholas Renotte"s _Real Time Sign Language Detection with Tensorflow Object Detection and Python | Deep Learning SSD_ 

Youtube tutorial: https://www.youtube.com/watch?v=pDXdlXlaCco&ab_channel=NicholasRenotte
Github Repo: https://github.com/nicknochnack/RealTimeObjectDetection

# 1. create_training_data.ipynb


This script creates the labeled images for sign language glosses the model shall be able to recognize later on. 

Major steps include:
- LOAD load virtual environment in python3.10 venv
- CREATE images 
- TRANSFORM images (optional, to increase data set size and model accuracy)_
- SPLIT data set in test vs train

The current code trains a model on the labeled images for the sign language glosses _"hello", "yes", "no", " i love you", "thanks"_ used in the YouTube tutorial as well
 as the glosses _"montag", "auch", "mehr", "wolke", "als", "sonne", "ueberwiegend", "regen", "gewitter"_ taken from the _PHOENIX 2014 weather data_.

**In case you change or add closes ensure to modify the following code:**
- create the labels - add labels in dictionary 
- add all gloss training data in collectedimages and perform train test split

_Note that the signs are taken statically and do not take into consideration spatial & temporal context of a gloss._

# 2. train_cnn_model.ipynb

Subsequently, the pre trained model needs to be retrained on the labeled images from step 1.

**In case you change or add closes ensure to modify the following code:**
- create label maps - add new glosses incl. numbers in sequential order
- create tf_record files
- config - update no. of classes (set to 5 in sample model)
- LOAD the model - add no. of latest creaed .ckpt file in _my_SSD_mobnet_

**Sample .ckpt files of the model trained with sample glosses to be found in _models/sample_model_**.

