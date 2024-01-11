import React, { useState, useRef } from "react";
import { Tensor, InferenceSession } from "onnxruntime-web";
import Loader from "./components/loader";
import { detectImage } from "./utils/detect";
import "./style/App.css";

const App = () => {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState("Loading OpenCV.js...");
  const [image, setImage] = useState(null);
  const [fps, setFps] = useState(0);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  let lastTime = 0;

  // configs
  const modelName = "yolov7-tiny.onnx";
  const modelInputShape = [1, 3, 640, 640];
  const classThreshold = 0.2;

  const extractFrame = (video) => {
    if (lastTime !== 0) {
      const fps = 1000 / (Date.now() - lastTime);
      setFps(fps);
    }
    lastTime = Date.now();
    const { videoWidth, videoHeight } = video;
    const canvas = document.createElement("canvas");
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);
    const dataUri = canvas.toDataURL("image/jpeg");
    imageRef.current.src = dataUri; // set image source
    setImage(dataUri);
    requestAnimationFrame(() => {
      extractFrame(video);
    });
  };
  cv["onRuntimeInitialized"] = async () => {
    // create session
    const modelUri = `${process.env.PUBLIC_URL}/model/${modelName}`;
    setLoading(`Loading YOLOv7 model... ${modelUri}`);
    const yolov7 = await InferenceSession.create(modelUri);
    // warmup model
    setLoading("Warming up model...");
    const tensor = new Tensor(
      "float32",
      new Float32Array(modelInputShape.reduce((a, b) => a * b)),
      modelInputShape
    );
    await yolov7.run({ images: tensor });

    setSession(yolov7);
    setLoading(false);

    const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true});
    videoRef.current.srcObject = mediaStream;
  };

  return (
    <div className="App">
      {loading && <Loader>{loading}</Loader>}
      {fps && <div className="fps">{fps.toFixed(2)} FPS</div>}
      <div className="content">
        <img
          ref={imageRef}
          src="#"
          alt=""
          style={{ display: image ? "block" : "none" }}
          onLoad={() => {
            detectImage(
              imageRef.current,
              canvasRef.current,
              session,
              classThreshold,
              modelInputShape
            );
          }}
        />
        <canvas
          id="canvas"
          width={modelInputShape[2]}
          height={modelInputShape[3]}
          ref={canvasRef}
        />
      </div>

      <video
        ref={videoRef}
        style={{ display: "none"}}
        autoPlay
        onLoadedMetadata={() => {
          extractFrame(videoRef.current);
        }
        }
      />
    </div>
  );
};

export default App;
