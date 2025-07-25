async function selectMediaSource() {
  if (!navigator.mediaDevices?.enumerateDevices) {
    console.log("enumerateDevices() not supported.");
    return;
  }

  try {
    const audioSelect = document.getElementById("audioSource");
    const videoSelect = document.getElementById("videoSource");
    audioSelect.innerHTML = "";
    videoSelect.innerHTML = "";

    const devices = await navigator.mediaDevices.enumerateDevices();
    console.log("Available media devices:", devices);
    devices.forEach((device) => {
      console.log(`${device.kind}: ${device.label} id = ${device.deviceId}`);
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.text = device.label || `${device.kind} ${audioSelect.length + 1}`;
      if (device.kind === "audioinput") {
        audioSelect.appendChild(option);
      } else if (device.kind === "videoinput") {
        videoSelect.appendChild(option);
      }
    });
  } catch (error) {
    console.log("Couldn't enumerate devices!", error);
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    alert("navigator.mediaDevices.getUserMedia() is not supported");
    return;
  }

  const audioSource = document.getElementById("audioSource").value;
  const videoSource = document.getElementById("videoSource").value;

  const constraints = {
    audio: audioSource ? { deviceId: { exact: audioSource } } : true,
    video: videoSource
      ? { deviceId: { exact: videoSource }, width: { exact: 1280 }, height: { exact: 720 } }
      : true,
  };

  try {
    const localMediaStream = await navigator.mediaDevices.getUserMedia(constraints);
    const video = document.querySelector("video");
    video.srcObject = localMediaStream;
  } catch (error) {
    console.log("Rejected!", error);
  }
}

document.getElementById("startBtn").addEventListener("click", startCamera);

// Populate device lists on page load
selectMediaSource();