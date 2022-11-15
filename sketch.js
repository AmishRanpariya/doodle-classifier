// https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%257B_22k_22_3A_22_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22bas_5C_22_22%257D%255D%22))&prefix=bas&forceOnObjectsSortingFiltering=false

// https://learn.ml5js.org/#/reference/neural-network

const IMAGE_WIDTH_FOR_MODEL = 36;
const IMAGE_CHANNELS_FOR_MODEL = 4;

const IMAGE_WIDTH_FOR_PREVIEW = 36 * 8;

const RATIO = IMAGE_WIDTH_FOR_PREVIEW / IMAGE_WIDTH_FOR_MODEL;

let mixedData = [];
function preload() {
	loadJSON("/data/6mixed4000each.json", (data) => {
		mixedData = data.data;
		mixedData = shuffle(mixedData);
	});
}

let pg;
let model;
let canvas;
let state = "training";
let res;

function setup() {
	canvas = createCanvas(IMAGE_WIDTH_FOR_PREVIEW * 2, IMAGE_WIDTH_FOR_PREVIEW);
	pixelDensity(1);
	const layers = [
		{
			type: "conv2d",
			filters: 16,
			kernelSize: 3,
			strides: 1,
			activation: "relu",
			kernelInitializer: "GlorotUniform",
			dilationRate: 1,
		},
		{
			type: "conv2d",
			filters: 16,
			kernelSize: 3,
			strides: 1,
			activation: "relu",
			kernelInitializer: "GlorotUniform",
		},
		{
			type: "maxPooling2d",
			poolSize: [2, 2],
			padding: "valid",
			strides: [2, 2],
		},
		{
			type: "conv2d",
			filters: 32,
			kernelSize: 3,
			strides: 1,
			activation: "relu",
			kernelInitializer: "GlorotUniform",
		},
		{
			type: "conv2d",
			filters: 32,
			kernelSize: 3,
			strides: 1,
			activation: "relu",
			kernelInitializer: "GlorotUniform",
		},
		{
			type: "maxPooling2d",
			poolSize: [2, 2],
			padding: "valid",
			strides: [2, 2],
		},
		{
			type: "conv2d",
			filters: 64,
			kernelSize: 3,
			strides: 1,
			activation: "relu",
			kernelInitializer: "GlorotUniform",
		},
		{
			type: "conv2d",
			filters: 64,
			kernelSize: 3,
			strides: 1,
			activation: "relu",
			kernelInitializer: "GlorotUniform",
		},
		{
			type: "maxPooling2d",
			poolSize: [2, 2],
			padding: "valid",
			strides: [2, 2],
		},
		{
			type: "dropout",
			rate: 0.1,
			seed: null,
		},
		{
			type: "flatten",
		},
		{
			type: "dense",
			kernelInitializer: "GlorotUniform",
			activation: "tanh",
			units: 512,
		},
		{
			type: "dense",
			kernelInitializer: "GlorotUniform",
			activation: "softmax",
		},
	];

	const options = {
		task: "imageClassification",
		inputs: [
			IMAGE_WIDTH_FOR_MODEL,
			IMAGE_WIDTH_FOR_MODEL,
			IMAGE_CHANNELS_FOR_MODEL,
		],
		layers,
		debug: true,
	};

	model = ml5.neuralNetwork(options);
	// model = ml5.imageClassifier("DoodleNet", modelReady);
	const modelDetails = {
		weights: "models/400model2cat/model.weights.bin",
		model: "models/400model2cat/model.json",
		metadata: "models/400model2cat/model_meta.json",
	};
	// model.load(modelDetails, modelLoaded);

	pg = createGraphics(IMAGE_WIDTH_FOR_MODEL, IMAGE_WIDTH_FOR_MODEL);
	pg.strokeWeight(0.5);
	pg.stroke(0);
	pg.noFill();
	background(255);

	res = createDiv("Result:");
}

let doodle = [];
function modelLoaded() {
	console.log("Model Loaded");
}
function draw() {
	if (state == "training") {
	} else if (state == "testing") {
		if (mouseIsPressed) {
			pg.strokeWeight(0.2);
			pg.line(pmouseX / RATIO, pmouseY / RATIO, mouseX / RATIO, mouseY / RATIO);
			strokeWeight((1 * RATIO) / 2);
			line(pmouseX, pmouseY, mouseX, mouseY);
		}
		if (frameCount % 20 == 0) {
			// 60/20 times in a second
			classifyCanvas();
		}
	}
	image(
		pg,
		IMAGE_WIDTH_FOR_PREVIEW,
		0,
		IMAGE_WIDTH_FOR_PREVIEW,
		IMAGE_WIDTH_FOR_PREVIEW
	);
}

function classifyCanvas() {
	model.classify({ image: pg }, gotResult);
}

function gotResult(error, results) {
	if (error) {
		console.error(error);
	}
	if (results?.[0]?.label) {
		res.html(
			"Result: " +
				results[0].label +
				"<br/>Confidence: " +
				results[0].confidence
		);
	} else {
		res.html("Result: Awaiting");
	}
}

function showDoodle(shape) {
	pg.background(255);
	pg.strokeWeight(random(1, 5));
	pg.stroke(0);
	pg.noFill();
	pg.push();
	pg.translate(pg.width / 2, pg.height / 2);
	const scale = random(0.5, 1);
	const offsetX = random(IMAGE_WIDTH_FOR_MODEL / 20, IMAGE_WIDTH_FOR_MODEL / 5);
	for (let i = 0; i < shape.length; i++) {
		pg.beginShape();
		for (let j = 0; j < shape[i][0].length; j++) {
			pg.vertex(
				mapper(shape[i][0][j] - 128, scale) + offsetX,
				mapper(shape[i][1][j] - 128, scale)
			);
		}
		pg.endShape();
	}
	pg.pop();
}

function mapper(x, scale) {
	return map(x, 0, 256, 0, IMAGE_WIDTH_FOR_MODEL * scale);
}

function whileTraining(epoch, loss) {
	console.log(`epoch: ${epoch}, loss:${loss}`);
}

function doneTraining() {
	console.log("done!");
	console.log("PREDICT MODE");
	state = "testing";
}

function keyPressed() {
	if (key == "t") {
		state = "training";
		noLoop();
		for (let i = 0; i < mixedData.length; i++) {
			if (!mixedData[i].recognized) continue;
			if (i > 8000) break;
			showDoodle(mixedData[i].drawing);
			console.log(i);
			model.addData({ image: pg }, { label: mixedData[i].word });
		}
		model.normalizeData();
		let options = {
			epochs: 100,
		};
		model.train(options, whileTraining, doneTraining);
		loop();
		return false;
	} else if (key == "s") {
		console.log("SAVING");
		model.save("model");
		console.log("SAVED");
	} else if (key == "c") {
		console.log("Classifying");
		classifyCanvas();
	} else if (key == "p") {
		console.log("PREDICT MODE");
		state = "testing";
	} else if (key == "x") {
		console.log("CLEARED");
		pg.background(255);
		background(255);
	}
}

function modelReady() {
	console.log("Model Ready");
	console.log(model);
}
