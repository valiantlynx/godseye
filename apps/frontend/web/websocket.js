let ws;
let isSending = false;
let webSocketData = null; 

function changeDartTitle(input) {
    if (window.updateDashboardFromJS) {
      window.updateDashboardFromJS(JSON.stringify(input));
    } else {
      console.error("Dart function updateDashboardFromJS is not available.");
    }
  }

function startWebSocket(wsUrl) {
  ws = new WebSocket(wsUrl);

  ws.onopen = function () {
    console.log("WebSocket connected");
  };

  ws.onclose = function () {
    console.log("WebSocket closed");
    isSending = false; // Reset the sending flag if the WebSocket closes
  };

  ws.onmessage = function (event) {
    // console.log("Message received from server:", event.data);
  
    try {
      const receivedData = JSON.parse(event.data);
      console.log("Parsed data:", receivedData);
      webSocketData = receivedData;
  
    // returns processing until 20 frames is collected
      if (receivedData.hasOwnProperty("job")) {
        // console.log("Job status:", receivedData["job"]);
  
    // if we actually get a result
      } else if (receivedData.hasOwnProperty("result")) {
        changeDartTitle(receivedData);
      } else {
        console.log("Unexpected result:", receivedData);
      }
  
    } catch (e) {
      console.error("Error parsing received data:", e);
    }
  };  
}





function captureAndSendFrame(videoId) {
    const video = document.getElementById(videoId);
    if (!video) {
      console.log("Video element not found");
      return;
    }
  
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    const delay = 100; // Set delay to 100ms (10 frames per second)
  
    video.addEventListener("canplay", function () {
      console.log("Video is ready, starting frame capture");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
  
      function sendFrame() {
        if (ws && ws.readyState === WebSocket.OPEN && !isSending) {
          isSending = true; // Set the flag to indicate that a frame is being processed
  
          try {
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(
              (blob) => {
                if (blob) {
                  blob.arrayBuffer().then(
                    (buffer) => {
                      try {
                        ws.send(buffer);
                      } catch (sendError) {
                        console.error("Error sending frame:", sendError);
                      }
                    },
                    (bufferError) => {
                      console.error("Error converting blob to ArrayBuffer:", bufferError);
                    }
                  );
                } else {
                  console.error("Blob creation failed");
                }
                isSending = false; // Reset the flag after processing
              },
              "image/jpeg"
            );
          } catch (error) {
            console.error("Error capturing frame:", error);
            isSending = false; // Reset the flag on error
          }
        } else if (ws && ws.readyState !== WebSocket.OPEN) {
          console.log("WebSocket not ready. Retrying frame capture...");
        }
  
        // Set a single timeout at the end, ensuring we attempt the next frame after a delay
        setTimeout(sendFrame, delay);
      }
  
      // Start the frame capture loop
      setTimeout(sendFrame, delay);
    });
  }
  
  

function getWebsocketData(){
  return webSocketData;
}