const express = require('express');
const app = express();
const http = require('http');
const cors = require('cors');
const { Server } = require('socket.io');
app.use(cors());

const server = http.createServer(app);

const io = new Server(server, {
  cors: {
    origin: '*',
  }
});

io.on('connection', (socket) => {
  console.log('Client connected');

  socket.on('offer', (message) => {
    console.log('Offer Received:', message);
    io.emit('offer', message); // Broadcast the received message to all connected clients
  });

  socket.on('answer', (message) => {
    console.log('Answer Received:', message);
    io.emit('answer', message); // Broadcast the received message to all connected clients
  });

  socket.on('client_candidate', (message) => {
    console.log('client_candidate Received:', message);
    io.emit('client_candidate', message); // Broadcast the received message to all connected clients
  });

  socket.on('server_candidate', (message) => {
    console.log('server_candidate Received:', message);
    io.emit('server_candidate', message); // Broadcast the received message to all connected clients
  });

  socket.on('ICE_over', (message) => {
    console.log('ICE_over Received:', message);
    io.emit('ICE_over', message); // Broadcast the received message to all connected clients
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected');
  });
});

const PORT = process.env.PORT || 8765;
server.listen(PORT, () => {
  console.log(`Socket.IO server running on http://localhost:${PORT}`);
});
