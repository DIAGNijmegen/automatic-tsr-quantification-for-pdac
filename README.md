# automatic-tsr-quantification-for-pdac
###Steps
  - clone the repo and build the docker.
  - Run the docker.
  - Download weights of the models
  - For inference: modify the .sh in code/schedule_tumor_characterization.sh to make the input images visible and run it.
  - Apply inference.
  - Doublecheck that the .sh run_all_automatic_tsr.sh has correct paths to the input images. and run it.
    

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Installation
1. Clone the repository
   ```bash
   git clone https://github.com/DIAGNijmegen/automatic-tsr-quantification-for-pdac.git

 2. Build the docker.
 3. Run the docker
 4. Download weights
  ```markdown
  ## Usage
  ```bash
  pip install gdown
  gdown --folder https://drive.google.com/drive/folders/1LzGj7nmuYVQwjcxFvUW7Zny3V1M-f6mM?usp=drive_link
