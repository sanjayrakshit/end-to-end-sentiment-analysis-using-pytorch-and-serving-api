# lightweight python
FROM python:3.8-slim

RUN apt-get update

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN ls -la $APP_HOME/

# Install dependencies
RUN pip install -r requirements.txt

# Run the streamlit on container startup
# CMD [ "streamlit", "run","--server.enableCORS","false","imgwebapp.py" ]
CMD ["uvicorn", "predictor_endpoint:app", "--host", "0.0.0.0", "--port", "8001"]