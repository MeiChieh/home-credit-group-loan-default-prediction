# Start from a base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Copy the helper module from the local filesystem into the container
COPY hepler /app/hepler

# Install Git and libgomp1 (required for lgbm)
   
RUN apt-get update && \
    apt-get install -y git libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install the required packages
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY ["vote_model.pkl", "app.py", "./"] .

# Expose the app port
EXPOSE 80

# Run command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]