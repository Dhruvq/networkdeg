# Use a lightweight Python version
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the app
CMD ["gunicorn", "-w", "3", "-b", "0.0.0.0:5000", "--timeout", "300", "app:app"]