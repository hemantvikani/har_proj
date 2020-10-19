# har_proj
Human activity recognition has broad applications in medical research and
human survey system. In this project, we designed a smartphone-based
recognition system that recognizes five human activities: (WALKING,
WALKING UPSTAIRS, WALKING DOWNSTAIRS, SITTING, STANDING,
LAYING). The system collected time series signals using a built-in
accelerometer, generated 561 features in both time and frequency domain,
and then reduced the feature dimensionality to improve the performance. The
activity data were trained and tested using 4 learning methods: Decision tree
classifier, k-nearest neighbor algorithm, random forest classifier, and logistic
regression classifier. 
The best classification rate in our experiment was 86.08%, which is achieved
by random forest with gini index and features selected by PCA. Classification
performance is robust to the orientation and the position of smartphones. 
