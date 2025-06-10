
## Steps

1 - Put all the files in data/raw folder
2 - ```docker compose up -d```
3 - ```python ./dags/split_data.py``` : To split the data into training and simulation and store into split
4 - ```python ./dags/initial_ingest.py```: To enter the training data into the raw bucket.
5 - 
    ```docker exec spark-client /bin/bash``` : To execute commands in the spark container
    ```cd dags```: Move to the dags folder
    ```spark-submit --master local[*] --driver-memory 4g --executor-memory 4g --packages org.apache.hadoop:hadoop-aws:3.4.1 transform_data.py```: Run transformation on the raw training data.
