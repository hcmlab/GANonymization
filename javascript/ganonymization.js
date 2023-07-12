// Preprocessing function
function preprocess(imageData, imageSize) {
    // Reshape Array
    const [inputWidth, inputHeight, inputChannels] = [imageSize, imageSize, 4];
    const [outputBatchSize, outputChannels, outputHeight, outputWidth] = [1, 3, imageSize, imageSize];

    const inputNdArray = ndarray(new Float32Array(imageData.data), [inputWidth, inputHeight, inputChannels]);
    const outputArray = new Float32Array(outputChannels * outputHeight * outputWidth);
    const outputNdArray = ndarray(outputArray, [outputChannels, outputWidth, outputHeight]);

    const inputChannelsWithoutAlpha = inputNdArray.hi(inputWidth, inputHeight, inputChannels - 1);

    ndarray.ops.assign(outputNdArray, inputChannelsWithoutAlpha.transpose(2, 0, 1));

    // Normalize
    const normalizedTensor = ndarray.ops.subseq(ndarray.ops.divseq(outputNdArray, 128.0), 1.0);

    // Return the preprocessed input tensor
    // return normalizedTensor.data;
    return new ort.Tensor('float32', normalizedTensor.data, [outputBatchSize, outputChannels, outputHeight, outputWidth]);
}

// Postprocessing function
function postprocess(outputData, imageSize) {
    // Reshape Array
    const [inputBatchSize, inputChannels, inputHeight, inputWidth] = [1, 3, imageSize, imageSize];
    const [outputWidth, outputHeight, outputChannels] = [imageSize, imageSize, 4];

    const inputNdArray = ndarray(outputData.data, [inputChannels, inputWidth, inputHeight]);
    const outputArray = new Float32Array(outputWidth * outputHeight * outputChannels);
    const outputNdArray = ndarray(outputArray, [outputWidth, outputHeight, outputChannels]);

    const inputChannelsTransposed = inputNdArray.transpose(1, 2, 0);

    ndarray.ops.assign(outputNdArray.hi(outputWidth, outputHeight, outputChannels - 1), inputChannelsTransposed);
    ndarray.ops.assigns(outputNdArray.pick(null, null, outputChannels - 1), 1);

    // Denormalize
    const denormalizedTensor = ndarray.ops.mulseq(ndarray.ops.addseq(outputNdArray, 1.0), 128.0);
    const outputDataArray = denormalizedTensor.data;

    // Create image data object to hold the postprocessed data
    const outputImageData = new ImageData(
        new Uint8ClampedArray(outputDataArray),
        denormalizedTensor.shape[0],
        denormalizedTensor.shape[1]
    );

    // Return the postprocessed image data
    return outputImageData;
}

async function anonymize(canvas_input, canvas_output, model, image_size, on_update_callback) {
    var ctxLandmarks = canvas_input.getContext('2d');
    var ctxAnonymized = canvas_output.getContext('2d');
    // Run GANonymization
    ctxAnonymized.save();
    // Extract landmarks from landmarks view
    const dx = canvas_output.width / 2 - image_size / 2;
    const dy = canvas_output.height / 2 - image_size / 2;
    const inputData = ctxLandmarks.getImageData(dx, dy, image_size, image_size)

    // Preprocess the input data
    const preprocessedData = preprocess(inputData, image_size);
    const input_tag = model.inputNames[0];
    const input_feed = {};
    input_feed[input_tag] = preprocessedData;

    // Run the ONNX model with the preprocessed input
    model.run(input_feed).then(output => {
        clearCanvas(ctxAnonymized);
        // Postprocess the output data
        const output_tag = model.outputNames[0];
        const postprocessedData = postprocess(output[output_tag], image_size);
        // Draw image
        ctxAnonymized.putImageData(postprocessedData, dx, dy);
        ctxAnonymized.restore();

        on_update_callback();
    }).catch(error => {
        console.log(error);
    });
}
