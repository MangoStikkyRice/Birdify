# Use the official Python 3.11.9 slim image
FROM python:3.11.9-slim

# Set environment variable to disable output buffering (helpful for logging)
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker caching
COPY requirements.txt /app/

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . /app

# (Optional) Expose any ports your application needs (for example, 8000 for a web app)
# EXPOSE 8000

# Set the default command to run your application
# This assumes your entry point is main.py in the project root.
CMD ["python", "main.py"]
