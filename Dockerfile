FROM tensorflow/serving:latest

COPY mymodel models/mymodel
COPY mymodel_more models/mymodel_more
COPY models.config models/
