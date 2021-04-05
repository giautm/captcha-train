FROM alpine as downloader
RUN apk upgrade && apk add curl unzip

ENV MODEL_VERION "0.1.0-alpha"
RUN curl -sfLo model.zip "https://github.com/giautm/captcha-train/releases/download/v${MODEL_VERION}/model-with-label-v${MODEL_VERION}.zip" && \
    unzip -q -d /model model.zip && \
    rm model.zip

FROM tensorflow/serving:2.4.0

ENV MODEL_NAME "captcha"
COPY --from=downloader /model /models/${MODEL_NAME}/1
