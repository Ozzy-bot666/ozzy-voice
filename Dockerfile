FROM node:22-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source
COPY . .

# Create memory directory (will be overwritten by mount in production)
RUN mkdir -p /home/node/clawd

EXPOSE 3000

CMD ["npm", "start"]
