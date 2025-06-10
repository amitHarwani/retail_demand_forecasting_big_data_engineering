FROM apache/airflow:2.10.5-python3.12

USER root
# Install OpenJDK 11 (a common and stable Java version for Spark)
# apt-get update to refresh package lists
# apt-get install -y openjdk-11-jdk to install Java Development Kit
# rm -rf /var/lib/apt/lists/* to clean up apt cache and keep image size down
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk \
    && apt-get install -y --no-install-recommends curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin


# Install Spark (and PySpark)
ENV SPARK_VERSION=4.0.0
ENV HADOOP_VERSION=3
RUN curl -sSL https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    | tar -xz -C /opt/ \
    && mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark

ENV SPARK_HOME=/opt/spark
ENV PATH="$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH"

USER airflow

WORKDIR /opt/airflow
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt