__kernel void feedForward(
    const int M, const int N, const int K, const int offsetInput,
    __global const float *input,
    __global const float *weights,
    __global float *results)
{
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of results (0..M)
    const int globalCol = get_global_id(1); // Col ID of results (0..N)
 
    // Compute a single element (loop over K)
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        sum += input[offsetInput + globalRow * K + i] * weights[i * N + globalCol];
    }
 
    // Calculate sigmoide
    results[globalRow * N + globalCol] = 1 / (1 + exp(-sum));
}

__kernel void calculateDeltaOutput(
    __global const float *outputs,
    __global const float *samples,
    __global float *results,
    uint const outputSize,
    uint const samplesOffset)
{
    uint outputId = get_global_id(0);
    if (outputId >= outputSize)
        return;

    results[outputId] = 
        outputs[outputId] * 
        (1 - outputs[outputId]) * 
        (samples[samplesOffset + outputId] - outputs[outputId]);
}

__kernel void calculateDeltaHidden(
    __global const float *hidden,
    __global const float *deltaOutput,
    __global const float *weights,
    __global float *results,
    uint const hiddenSize,
    uint const outputSize)
{
    uint hiddenId = get_global_id(0);
    if (hiddenId >= hiddenSize)
        return;

    float sum = 0.0f;
    uint id;
    for (uint i = 0; i < outputSize; i++)
    {
        id = hiddenId * outputSize + i;
        sum += deltaOutput[i] * weights[id];
    }

    results[hiddenId] = hidden[hiddenId] * (1 - hidden[hiddenId]) * sum;
}

__kernel void updateWeights(
    __global float* weights,
    __global const float *neurons,
    __global const float *delta,
    uint const neuronOffset,
    uint const neuronSize,
    uint const deltaSize,
    float const eta)
{
    uint neuronId = get_global_id(0);
    uint deltaId  = get_global_id(1);

    if (neuronId >= neuronSize)
        return;

    if (deltaId >= deltaSize)
        return;

    uint id = neuronId * deltaSize + deltaId;
    weights[id] += eta * delta[deltaId] * neurons[neuronOffset + neuronId];
}
