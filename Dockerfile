# Step 1: Use an official Python runtime as a parent image
FROM python:3.11-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container at /app
COPY . /app

# Step 4: Install libgomp for LightGBM
RUN apt-get update && apt-get install -y libgomp1

# Step 5: Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Step 6: Collect static files (if your app uses them)
RUN python fraud_detection/manage.py collectstatic --noinput

# Step 7: Expose the port that the app will run on
EXPOSE 8000

# Step 8: Install Gunicorn
RUN pip install gunicorn

# Step 9: Run the app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "fraud_detection.wsgi:application"]
