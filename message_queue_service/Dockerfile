# Sử dụng image Node.js chính thức
FROM node:22-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép package.json và package-lock.json vào container
COPY package*.json ./

# Cài đặt dependencies
RUN npm install --production

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Lệnh chạy ứng dụng khi container khởi động
CMD ["node", "server.js"]
