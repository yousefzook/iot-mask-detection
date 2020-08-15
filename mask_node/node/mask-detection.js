// called when the runtime loads the node on startup
module.exports = function (RED) {
  // node configurations
  var node;

  // tensorflow js module
  let tf = null;
  // tensorflow js data moule
  let tfd = null;
  // blazeface module for face detection
  let blaze = null;

  // model to detect faces in a single image
  var faceDetectionModel;
  // model to detect if a face is wearing a mask
  var maskDetectionModel;

  // offset useed in bordering faces
  var offset = null;

  // face detection parameters
  const returnTensors = false;
  const flipHorizontal = false;
  const annotateBoxes = false;

  /*
    Method used to initiate the needed modules: 
    tf: tensorflow js module
    tfd: tensorflow js data module
    blazeface: module for face detection
  */
  const initModules = function (node) {
    const globalContext = node.context().global;
    if (!tf) {
      tf = globalContext.get("tfjs");
    }
    if (!tf) {
      globalContext.set("tfjs", tf);
      node.log(`Loaded TensorFlow.js v${blaze}`);
    }
    tf = require("@tensorflow/tfjs-node");
    tfd = require("@tensorflow/tfjs-data");
    blaze = require("@tensorflow-models/blazeface");
    offset = tf.scalar(127.5);
  };

  /*
    Method used to register the new mask detection node. 
    It works as the main method.
  */
  const TFJSMaskDetection = function (config) {
    initModules(this);
    
    loadFaceDetectionModel();
    loadMaskDetectionModel();

    // file system module
    const fs = require("fs");
    node = this;
    RED.nodes.createNode(this, config);

    this.modelUrl = config.modelUrl;

    // register a listener to the 'input' event
    // which gets called whenever a message arrives at the node
    node.on("input", function (msg) {
      try {
        node.status({
          fill: "yellow",
          shape: "dot",
          text: "running inference...",
        });

        // read input data
        const inputData =
          typeof msg.payload === "string"
            ? fs.readFileSync(msg.payload)
            : msg.payload;

        runPrediction(tf.node.decodeImage(inputData), msg);
        node.status({});
      } catch (error) {
        node.status({
          fill: "red",
          shape: "dot",
          text: "Error Happened",
        });
        node.error(error, msg);
      }
    });

    node.on("close", function () {});
  };

  // run inference against the input image and return the prediction
  const runPrediction = async function (img, msg) {
    // tells whether there's a person not wearing a mask or not
    var dangerDetected = false;

    // ensure the image has only three channels
    img = tf.slice(img, [0, 0, 0], [-1, -1, 3]);

    console.log(
      "Input Image Shape",
      img.shape,
      "\n-------------------------------\n"
    );

    // list of detected faces
    let detectedFaces = [];
    try {
      detectedFaces = await faceDetectionModel.estimateFaces(
        img,
        returnTensors,
        flipHorizontal,
        annotateBoxes
      );

      // loop through the detected faces
      for (let i = 0; i < detectedFaces.length; i++) {
        // [mask probability, no mask probability]
        let probabilities = [];
        try {
          probabilities = await detectMask(img, detectedFaces[i]);

          // if one person isn't wearing a mask, there's danger
          if (probabilities[0] < 0.5) dangerDetected = true;
        } catch (e) {
          console.error("maskDetection:", e);
          return;
        }
        console.log(
          "Mask probability for face " + i + ": ",
          probabilities[0],
          "\n\n"
        );
      }

      // Danger exists --> Fire "Danger Is Around"
      // Safe --> Fire "Safety Ensured"
      msg.payload = dangerDetected ? "Danger Is Around" : "Safety Ensured";

      node.send(msg);
    } catch (e) {
      console.error("estimateFaces:", e);
      return;
    }
  };

  /*
  Method that detects whether a single face wears a mask or not
  img: whole image
  faceTensor: position of single face
  */
  async function detectMask(img, faceTensor) {
    const topLeft = [faceTensor.topLeft[1], faceTensor.topLeft[0]];
    const bottomRight = [faceTensor.bottomRight[1], faceTensor.bottomRight[0]];

    console.log("topLeft", topLeft);
    console.log("bottomRight", bottomRight);

    // [height, width]
    const size2D = [
      bottomRight[0] - topLeft[0] + 1,
      bottomRight[1] - topLeft[1] + 1,
    ];

    var start = [topLeft[0], topLeft[1], 0];
    var size3D = [size2D[0], size2D[1], 3];
    console.log("starting point: ", start);
    console.log("size3D", size3D);

    // crop the image
    var subImg = tf.slice(img, start, size3D);

    // form an image of size 224*224*3
    let face = tf.tidy(() =>
      subImg
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .sub(offset)
        .div(offset)
        .expandDims(0)
    );

    try {
      // use the mask detection model
      return await maskDetectionModel.predict(face).data();
    } catch (e) {
      console.error("maskDetection:", e);
      return;
    }
  }

  /*
  Loads the face detection model from disk, then make the node have a black color.
  */
  async function loadFaceDetectionModel() {
    console.log("loading face detection model");
    await blaze.load().then((m) => {
      faceDetectionModel = m;
      console.log("face detection model loaded");
      node.status({
        fill: "black",
        shape: "dot",
        text: "Face model is ready",
      });
    });
  }

  /*
  Loads the mask detection model from disk, then make the node have a black color
  */
  async function loadMaskDetectionModel() {
    console.log("loading mask detection model");
    await tf.loadLayersModel("file://mask_tfjs/model.json").then((m) => {
      maskDetectionModel = m;
      console.log("mask detection model loaded");
      node.log("Mask Model Loaded.");
      node.status({
        fill: "green",
        shape: "dot",
        text: "Mask model is ready",
      });
    });
  }

  // register the TensorFlow.js Mask Detection node
  RED.nodes.registerType("mask-detection", TFJSMaskDetection);
};
