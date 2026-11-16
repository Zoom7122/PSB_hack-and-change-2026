FROM nginx:1.27-alpine
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY API/static/index.html /usr/share/nginx/html/index.html
EXPOSE 8080
