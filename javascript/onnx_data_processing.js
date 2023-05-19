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
