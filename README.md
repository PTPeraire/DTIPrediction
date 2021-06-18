# DTIPrediction

Supplementary data for _Using bioactivity signatures to predict drug-target interactions_.

Here we find, the raw ChEMBL data for the time-split mentioned in Torren-Peraire et al., for both the compound-based and target-based method. Also included is the relevant results for each methods. For compound-based we show the rank of the true interactions tested, for the target-based method the validation output of the novel targets. 

Futhermore, we include a guided notebook detailing the implementation of the compound-based method (**CB\_method**). A similar notebook will be implemented for the target-based method in due course.

**CB_data:** Raw data for 495 novel compounds from ChEMBL v28, used in time-split to validate compound-based method

**CB_validation:** Rank of interactions involving novel compounds (using CB_data) applying the compound-based method with GSig

**TB_data:** Raw data for novel targets (from ChEMBL v28) used to validate the target-based method

**TB_validation:** Output for 10-fold cross-validation using the target-based method, for each novel target
