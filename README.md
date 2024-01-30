# automatic-tsr-quantification-for-pdac


##To download weights: 
  1. pip install gdown
  2. gdown --folder https://drive.google.com/drive/folders/1LzGj7nmuYVQwjcxFvUW7Zny3V1M-f6mM?usp=drive_link


###Steps
  - clone the repo and build the docker.
  - Run the docker.
  - Download weights of the models
  - For inference: modify the .sh in code/schedule_tumor_characterization.sh to make the input images visible and run it.
  - Apply inference.
  - Doublecheck that the .sh run_all_automatic_tsr.sh has correct paths to the input images. and run it.
    
