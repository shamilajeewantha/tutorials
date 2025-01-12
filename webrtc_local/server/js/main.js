'use strict';

const remoteVideo = document.getElementById('remoteVideo');
let remoteStream;
let remotePeerConnection;




// Logs an action (text) and the time when it happened on the console.
function trace(text) {
  text = text.trim();
  const now = (window.performance.now() / 1000).toFixed(3);

  console.log(now, text);
}

////////////////////////////////// socket.io ////////////////////////////////////////////////////

const socket = io("http://localhost:8765");

function sendMessage(type, message) {
  console.log('Client sending message: ', message);
  socket.emit(type, message);
}

socket.on('offer', function(message) {
  console.log('Client received offer:', message);
  handleOffer(message);
});


socket.on('ICE_over', function(message) {
  console.log('Client received ICE_over:', message);
  trace('remotePeerConnection createAnswer start.');
  remotePeerConnection.createAnswer()
    .then(createdAnswer)
    .catch(createAnswerError);
});


socket.on('client_candidate', function(message) {
  console.log('Client received client_candidate:', message);
  remotePeerConnection.addIceCandidate(new RTCIceCandidate(message))      
    .then(() => {
      handleConnectionSuccess();
    }).catch((error) => {
      handleConnectionFailure(error);
    });
});

socket.on('bye', function(message) {
  console.log('Client received bye:', message);
  handleRemoteHangup();
});

function handleRemoteHangup() {
  console.log('Session terminated.');
  remotePeerConnection.close();
  remotePeerConnection = null;
}

///////////////////////////////////////// Answer /////////////////////////////////////////////////

// Gets the name of a certain peer connection.
function getPeerName(peerConnection) {
  return (peerConnection === remotePeerConnection) ?
      'remotePeerConnection' : 'localPeerConnection';
}

// Logs success when setting session description.
function setDescriptionSuccess(peerConnection, functionName) {
  const peerName = getPeerName(peerConnection);
  trace(`${peerName} ${functionName} complete.`);
}

// Logs success when localDescription is set.
function setLocalDescriptionSuccess(peerConnection) {
  setDescriptionSuccess(peerConnection, 'setLocalDescription');
}

// Logs success when remoteDescription is set.
function setRemoteDescriptionSuccess(peerConnection) {
  setDescriptionSuccess(peerConnection, 'setRemoteDescription');
}

// Logs error when setting session description fails.
function setSessionDescriptionError(error) {
  trace(`Failed to create session description: ${error.toString()}.`);
}

// Logs error when setting session description fails.
function createAnswerError(error) {
  trace(`Failed to create answer: ${error.toString()}.`);
}

function handleOffer(description) {
  trace('remotePeerConnection setRemoteDescription start.');
  remotePeerConnection.setRemoteDescription(new RTCSessionDescription(description))
    .then(() => {
      setRemoteDescriptionSuccess(remotePeerConnection);
    }).catch(setSessionDescriptionError);
  

}

// Logs answer to offer creation and sets peer connection session descriptions.
function createdAnswer(description) {
  trace(`Answer from remotePeerConnection:\n${description.sdp}.`);

  trace('remotePeerConnection setLocalDescription start.');
  remotePeerConnection.setLocalDescription(description)
    .then(() => {
      setLocalDescriptionSuccess(remotePeerConnection);
    }).catch(setSessionDescriptionError);

  sendMessage('answer', description);
}

////////////////////////////////// Remote Media ////////////////////////////////////////////////////

// Handles remote MediaStream success by adding it as the remoteVideo src.
function gotRemoteMediaStream(event) {
  const mediaStream = event.stream;
  remoteVideo.srcObject = mediaStream;
  remoteStream = mediaStream;
  trace('Remote peer connection received remote stream.');
}

// Logs a message with the id and size of a video element.
function logVideoLoaded(event) {
  const video = event.target;
  trace(`${video.id} videoWidth: ${video.videoWidth}px, ` +
        `videoHeight: ${video.videoHeight}px.`);
}

remoteVideo.onloadedmetadata = logVideoLoaded;




////////////////////////////////// Peer Connection ////////////////////////////////////////////////////

const servers = null;  // Allows for RTC server configuration.

remotePeerConnection = new RTCPeerConnection(servers);
trace('Created remote peer connection object remotePeerConnection.');

remotePeerConnection.onicecandidate = handleIceCandidate;
remotePeerConnection.onaddstream = gotRemoteMediaStream;
remotePeerConnection.onremovestream = handleRemoteStreamRemoved;

function handleRemoteStreamRemoved(event) {
  console.log('Remote stream removed. Event: ', event);
}


///////////////////////////////////////// ICE Candidate /////////////////////////////////////////////////


function handleIceCandidate(event) {
  if (event.candidate) {
    console.log('Sending client ICE candidate to remote peer: ');
    sendMessage('server_candidate', event.candidate);
  } else {
    console.log('End of candidates.');
  }
}

// Logs changes to the connection state.
function handleConnectionChange(event) {
  const peerConnection = event.target;
  console.log('ICE state change event: ', event);
  trace(`${getPeerName(peerConnection)} ICE state: ` +
        `${peerConnection.iceConnectionState}.`);
}

// Logs that the connection succeeded.
function handleConnectionSuccess() {
  trace(`addIceCandidate success.`);
};

// Logs that the connection failed.
function handleConnectionFailure(error) {
  trace(`Failed to add ICE Candidate:\n`+
        `${error.toString()}.`);
}

window.onbeforeunload = function() {
  remotePeerConnection.close();
  remotePeerConnection = null;
};