# loan_default_prediction
This project focuses on building an automated loan default prediction system using machine learning techniques to assess the probability of default for each applicant, enabling proactive measures for risk mitigation.


# Docker
docker build -t ml-predict-api:latest -f predict_api/Dockerfile .

(MacOS)

docker run --rm -p 9000:9000 \
  -e MODEL_PATH=/app/exported_model \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -v "${PWD}/exported_model:/app/exported_model:ro" \
  --name ml-predict-api ml-predict-api:latest
