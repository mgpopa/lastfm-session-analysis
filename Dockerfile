FROM python:3.11-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless && \
    ln -s /usr/lib/jvm/java-17-openjdk-* /usr/lib/jvm/java-17-openjdk && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade prophet==1.3.0 numpy==1.26.4 pandas==2.0.3

#COPY exercise2.py .
#COPY exercise3.py .
#COPY exercise3_validation.py .

COPY *.py ./
RUN mkdir -p /app/output

CMD ["python", "exercise3.py", "exercise3_validation.py"]
