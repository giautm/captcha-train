# Captcha Train

Trainning source code for [giautm/captcha](https://github.com/giautm/captcha)

To run model in local:
```sh
docker build --file=Dockerfile --tag=captcha:dev .
docker run -p 8501:8501/tcp captcha:dev
```