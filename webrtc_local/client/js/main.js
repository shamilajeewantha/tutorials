'use strict';

// Audio will not be streamed because it is set to "audio: false" by default.
const mediaStreamConstraints = {
  video: true,
};


const localVideo = document.getElementById('localVideo');
let localStream;
let localPeerConnection;

// Define action buttons.
const startButton = document.getElementById('startButton');
const callButton = document.getElementById('callButton');
const hangupButton = document.getElementById('hangupButton');

// Add click event handlers for buttons.
startButton.onclick = startAction;
callButton.onclick = callAction;
hangupButton.onclick = hangupAction;

// Set up initial action buttons status: disable call and hangup.
callButton.disabled = true;
hangupButton.disabled = true;

///////////////////////////////////////// Fetch Local Media /////////////////////////////////////////////////

// Logs an action (text) and the time when it happened on the console.
function trace(text) {
  text = text.trim();
  const now = (window.performance.now() / 1000).toFixed(3);
  console.log(now, text);
}

// Sets the MediaStream as the video element src.
function gotLocalMediaStream(mediaStream) {
  localVideo.srcObject = mediaStream;
  localStream = mediaStream;
  trace('Received local stream.');
  callButton.disabled = false;  // Enable call button.
}

// Handles error by logging a message to the console.
function handleLocalMediaStreamError(error) {
  trace(`navigator.getUserMedia error: ${error.toString()}.`);
}

// Logs a message with the id and size of a video element.
function logVideoLoaded(event) {
  const video = event.target;
  console.log(event);/////////////////////////////////////////////////////////////
  trace(`${video.id} videoWidth: ${video.videoWidth}px, ` +
        `videoHeight: ${video.videoHeight}px.`);
}

localVideo.onloadedmetadata = logVideoLoaded;


////////////////////////////////// socket.io ////////////////////////////////////////////////////

const socket = io("http://localhost:8765");

function sendMessage(type, message) {
  console.log('Client sending message: ', message);
  socket.emit(type, message);
}

socket.on('server_candidate', function(message) {
  console.log('Client received server_candidate:', message);
  localPeerConnection.addIceCandidate(new RTCIceCandidate(message))      
    .then(() => {
      handleConnectionSuccess();
    }).catch((error) => {
      handleConnectionFailure(error);
    });
});

socket.on('answer', function(message) {
  console.log('Client received answer:', message);
  localPeerConnection.setRemoteDescription(new RTCSessionDescription(message))
    .then(() => {
      setRemoteDescriptionSuccess(localPeerConnection);
    }).catch(setSessionDescriptionError);
});

/////////////////////////////////// Offer Creation //////////////////////////////////////////////

// Gets the name of a certain peer connection.
function getPeerName(peerConnection) {
  return (peerConnection === localPeerConnection) ?
      'localPeerConnection' : 'remotePeerConnection';
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

// Logs offer creation and sets peer connection session descriptions.
function createdOffer(description) {
  trace(`Offer from localPeerConnection:\n${description.sdp}`);

  trace('localPeerConnection setLocalDescription start.');
  localPeerConnection.setLocalDescription(description)
    .then(() => {
      setLocalDescriptionSuccess(localPeerConnection);
    }).catch(setSessionDescriptionError);

  // Send the offer to the remote peer.
  console.log('sending offer message');
  sendMessage('offer',description);
}

function handleCreateOfferError(event) {
  console.log('createOffer() error: ', event);
}

///////////////////////////////////////// ICE Candidate /////////////////////////////////////////////////

function handleIceCandidate(event) {
  if (event.candidate) {
    console.log('Sending client ICE candidate to remote peer');
    sendMessage('client_candidate', event.candidate);
  } else {
    console.log('End of candidates.');
    sendMessage('ICE_over', 'gathering complete');
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

////////////////////////////////////// Button Handling ////////////////////////////////////////////


// Handles start button action: creates local MediaStream.
function startAction() {
  startButton.disabled = true;
  navigator.mediaDevices.getUserMedia(mediaStreamConstraints)
    .then(gotLocalMediaStream).catch(handleLocalMediaStreamError);
  trace('Requesting local stream.');
}

// Handles call button action: creates peer connection.
function callAction() {
  callButton.disabled = true;
  hangupButton.disabled = false;

  trace('Starting call.');

  // Get local media stream tracks.
  const videoTracks = localStream.getVideoTracks();
  if (videoTracks.length > 0) {
    trace(`Using video device: ${videoTracks[0].label}.`);
  }

  const servers = null;  // Allows for RTC server configuration.

  // Create peer connections and add behavior.
  localPeerConnection = new RTCPeerConnection(servers);
  trace('Created local peer connection object localPeerConnection.');

  localPeerConnection.onicecandidate = handleIceCandidate;

  // Add local stream to connection and create offer to connect.
  localPeerConnection.addStream(localStream);
  trace('Added local stream to localPeerConnection.');

  trace('localPeerConnection createOffer start.');
  localPeerConnection.createOffer()
    .then(createdOffer).catch(handleCreateOfferError);
}

// Handles hangup action: ends up call, closes connections and resets peers.
function hangupAction() {
  localPeerConnection.close();
  localPeerConnection = null;
  sendMessage('bye', 'Client has disconnected');
  hangupButton.disabled = true;
  callButton.disabled = false;
  trace('Ending call.');
}

window.onbeforeunload = function() {
  localPeerConnection.close();
  localPeerConnection = null;
};





